# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0.dev0
#   kernelspec:
#     display_name: flodym
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Example 6. Monte Carlo simulation of a recycling system
#
# This example propagates the uncertainty of two parameters through a small MFA system with a
# Monte Carlo simulation.
#
# The system has two processes besides the system environment:
# * `production` receives primary material from the system environment and recycled material
#   from `use`, and sends products to `use`.
# * `use` holds an in-use stock. The stock is prescribed over time; the inflow and the outflow
#   are computed with a stock-driven dynamic stock model. A share of the outflow, the recycling
#   rate, goes back to `production`; the rest leaves to the system environment.
#
# The future lifetime of the products and the recycling rate are uncertain. In each Monte Carlo
# run, we draw a value for each of them, compute the system, and add the results and their
# squares to two copies of the system. From these sums, we compute the mean and the standard
# deviation of all flows and stocks.
#
# All data is given in the script; no files are read.

# %% [markdown]
# ## 1. Load flodym and other useful packages

# %%
from copy import deepcopy

import graphviz
import numpy as np
import plotly.graph_objects as go
from IPython.display import SVG, display

import flodym as fd
import flodym.export as fde

# %% [markdown]
# ## 2. Define the MFA system
#
# The only dimension is time. Its items are given directly in the definition. The dynamic stock
# model also needs a cohort dimension with the same items, which holds the year of entry of the
# products into the stock.
#
# The recycling rate is the share of the flow from `use` to `production` in the outflow of the
# stock. The flow from `use` to the system environment gets the remainder. The flow from the
# system environment to `production`, the primary production, is the remainder on the inflow
# side of `production`.
#
# The lifetime is not part of the definition of the stock, so we pass it to the lifetime model
# in the function `compute` below. It is given by year of entry into the stock (cohort).

# %%
years = list(range(1950, 2101))

mfa_definition = fd.MFADefinition(
    dimensions=[
        fd.DimensionDefinition(letter="t", name="Time", dtype=int, items=years),
        fd.DimensionDefinition(letter="c", name="Cohort", dtype=int, items=years),
    ],
    parameters=[
        fd.ParameterDefinition(name="in-use stock", dim_letters=("t",)),
        fd.ParameterDefinition(name="lifetime", dim_letters=("t",)),
        fd.ParameterDefinition(name="recycling rate", dim_letters=("t",)),
    ],
    processes=[
        "sysenv",
        "production",
        fd.ProcessDefinition(name="use", outflow_shares={"production": "recycling rate"}),
    ],
    flows=[
        fd.FlowDefinition(from_process_name=origin, to_process_name=target, dim_letters=("t",))
        for origin, target in [
            ("sysenv", "production"),
            ("production", "use"),
            ("use", "production"),
            ("use", "sysenv"),
        ]
    ],
    stocks=[
        fd.StockDefinition(
            name="use",
            process="use",
            dim_letters=("t",),
            subclass=fd.StockDrivenDSM,
            lifetime_model_class=fd.LogNormalLifetime,
            cohort_letter="c",
            prescribed_stock="in-use stock",
        ),
    ],
)

# %% [markdown]
# ## 3. Set up the system and the parameter values
#
# We create the system from its definition.

# %%
mfa = fd.MFASystem.from_definition(mfa_definition)

# %% [markdown]
# The system is set up, but not yet computed. We can plot its structure: the processes, the
# flows between them, and the stock of `use`.
# Rendering the graph requires the Graphviz system binaries (see Howto 8). Without them, we
# print a message instead.

# %%
graph = fde.GraphvizProcessGraphPlotter(mfa=mfa).plot()
try:
    display(SVG(graph.pipe(format="svg")))
except graphviz.ExecutableNotFound:
    print("Install Graphviz (https://graphviz.org/download/) to show the process graph.")

# %% [markdown]
# The in-use stock follows a logistic curve, saturating at 100 units.
# The parameters `lifetime` and `recycling rate` get values in each Monte Carlo run below.

# %%
time = np.array(years)
mfa.set_parameter_values({"in-use stock": 100.0 / (1.0 + np.exp(-(time - 2000) / 12.0))})

# %% [markdown]
# ## 4. Prepare the sums
#
# We collect the sum and the sum of squares of all flows and stocks in two copies of the system.
# The helper function `stocks_and_flows` collects the arrays of a system that we need: all
# flows, and the stock, inflow and outflow of each stock. The labels are the same in copies of
# the system, so we can match the arrays of different copies by label.
#
# Flows and stocks are extensive (see `fd.FlodymArray.extensive`), but their squares are not.
# So we remove the extensivity from the arrays of the sum of squares. Otherwise, flodym warns
# when we combine them with extensive arrays below.


# %%
def stocks_and_flows(mfa: fd.MFASystem) -> dict[str, fd.FlodymArray]:
    """Collect the flows and stock arrays of an MFA system.

    Args:
        mfa: The MFA system.

    Returns:
        The arrays, keyed by a label such as ``"flow 'use => production'"``.
    """
    arrays = {f"flow '{name}'": flow for name, flow in mfa.flows.items()}
    for name, stock in mfa.stocks.items():
        for array_name in ("stock", "inflow", "outflow"):
            arrays[f"{array_name} of stock '{name}'"] = getattr(stock, array_name)
    return arrays


sums = stocks_and_flows(deepcopy(mfa))
sums_of_squares = stocks_and_flows(deepcopy(mfa))
for label in sums:
    sums[label][...] = 0.0
    sums_of_squares[label][...] = 0.0
    sums_of_squares[label].extensive = None

# %% [markdown]
# ## 5. Run the Monte Carlo simulation
#
# Until 2025, the lifetime is 20 years and the recycling rate is 0.4. From 2025, both move
# linearly to an uncertain target value, which they reach in 2100.
#
# The target lifetime is normally distributed with a mean of 25 years and a standard deviation
# of 4 years. We clip it at 5 years, since the lifetime model needs a positive mean.
# The target recycling rate is normally distributed with a mean of 0.6 and a standard deviation
# of 0.1. We clip it to the interval between 0 and 1, since it is a share.

# %%
historical_lifetime = 20.0
historical_recycling_rate = 0.4
transition_start = 2025
transition_end = 2100


def linear_transition(historical_value: float, target_value: float) -> np.ndarray:
    """Move linearly from the historical value to the target value over the transition period.

    Args:
        historical_value: The value until the start of the transition.
        target_value: The value from the end of the transition on.

    Returns:
        The values for each year.
    """
    return np.interp(time, [transition_start, transition_end], [historical_value, target_value])


# %% [markdown]
# The function `compute` computes one Monte Carlo run. It passes the lifetime to the lifetime
# model and computes all processes from the prescribed stock and the recycling rate.
# Before, it marks all flows and stocks as unset. Otherwise, the results of the previous run
# would count as given values in the next one.
#
# The standard deviation of the lifetime distribution is 30% of its mean.


# %%
def compute(mfa: fd.MFASystem) -> None:
    """Compute the flows and stocks of the MFA system from its current parameter values.

    Args:
        mfa: The MFA system.

    Raises:
        RuntimeError: If not all processes can be computed.
    """
    mfa.mark_all_unset()
    lifetime_model = mfa.stocks["use"].lifetime_model
    lifetime_model.mean = mfa.parameters["lifetime"].values
    lifetime_model.std = 0.3
    lifetime_model.relative_std = True
    not_computed = mfa.compute_all_possible()
    if not_computed:
        raise RuntimeError(f"Could not compute processes {not_computed}.")


# %%
n_samples = 100
rng = np.random.default_rng(seed=42)
lifetime_targets = np.clip(rng.normal(loc=25.0, scale=4.0, size=n_samples), 5.0, None)
recycling_rate_targets = np.clip(rng.normal(loc=0.6, scale=0.1, size=n_samples), 0.0, 1.0)

for lifetime_target, recycling_rate_target in zip(lifetime_targets, recycling_rate_targets):
    mfa.set_parameter_values(
        {
            "lifetime": linear_transition(historical_lifetime, lifetime_target),
            "recycling rate": linear_transition(historical_recycling_rate, recycling_rate_target),
        }
    )
    compute(mfa)
    for label, array in stocks_and_flows(mfa).items():
        sums[label][...] = sums[label] + array
        sums_of_squares[label][...] = sums_of_squares[label] + array**2

# %% [markdown]
# ## 6. Compute mean and standard deviation
#
# With the sum $S_1 = \sum_i x_i$ and the sum of squares $S_2 = \sum_i x_i^2$ over $N$ runs,
# the mean is $\bar{x} = S_1 / N$, and the (sample) standard deviation is
# $s = \sqrt{(S_2 - N \bar{x}^2) / (N - 1)}$.
# Rounding errors can make the variance slightly negative where it is zero, so we clip it.
#
# We store the results in two further copies of the system.

# %%
mean_mfa = deepcopy(mfa)
std_mfa = deepcopy(mfa)
means = stocks_and_flows(mean_mfa)
stds = stocks_and_flows(std_mfa)
for label in sums:
    means[label][...] = sums[label] / n_samples
    variance = (sums_of_squares[label] - n_samples * means[label] ** 2) / (n_samples - 1)
    stds[label][...] = variance.maximum(0.0) ** 0.5
    print(f"{label}: mean in {years[-1]} = {means[label].values[-1]:.2f}")

# %% [markdown]
# ## 7. Plot the recycling flow
#
# We plot the mean of the recycling flow and an approximate 95% confidence interval of the
# flow values, mean ± 1.96 standard deviations. This assumes that the flow values are about
# normally distributed. The interval describes the spread of the flow, not the uncertainty
# of its mean.

# %%
recycling_mean = mean_mfa.flows["use => production"].values
recycling_std = std_mfa.flows["use => production"].values
lower = np.clip(recycling_mean - 1.96 * recycling_std, 0.0, None)
upper = recycling_mean + 1.96 * recycling_std

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=np.concatenate([time, time[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill="toself",
        fillcolor="rgba(42, 120, 214, 0.2)",
        line={"width": 0},
        hoverinfo="skip",
        name="95% interval",
    )
)
fig.add_trace(
    go.Scatter(
        x=time,
        y=recycling_mean,
        mode="lines",
        line={"color": "#2a78d6", "width": 2},
        customdata=np.stack([lower, upper], axis=-1),
        hovertemplate=(
            "%{x}<br>mean: %{y:.2f}<br>95% interval: %{customdata[0]:.2f} – %{customdata[1]:.2f}"
            "<extra></extra>"
        ),
        name="Mean",
    )
)
fig.update_layout(
    title=f"Recycling flow from use to production ({n_samples} Monte Carlo runs)",
    xaxis_title="Year",
    yaxis_title="Flow (units per year)",
    hovermode="x",
    plot_bgcolor="#fcfcfb",
    yaxis={"rangemode": "tozero", "gridcolor": "#e8e7e4"},
    xaxis={"showgrid": False},
)
fig.show(renderer="notebook")
