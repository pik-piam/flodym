# %% [markdown]
# # Uncertainty and Monte Carlo
#
# Parameters of an MFA are rarely known exactly. flodym propagates their uncertainty through a
# model by Monte Carlo sampling:
#
# 1. Declare which parameters are uncertain, and how, in an `UncertaintyDefinition`.
# 2. Run the model repeatedly with sampled parameter values, with `run_monte_carlo()`.
# 3. Read mean, standard deviation and percentiles of any flow, stock or derived quantity.
#
# We use the `ExampleMFA` from `flodym.example_objects`: end-of-life machines and buildings are
# shredded or demolished, the scrap is remelted, and losses go to landfills and slag piles.
#
# ## A first run
#
# The values a parameter has in the model are its *nominal* values, the best estimate. A
# distribution adds the spread around them. Here, the amount of end-of-life machines is uncertain
# by 20 % (relative standard deviation of a lognormal distribution, whose mean is the nominal
# value):

# %%
import numpy as np
import plotly.graph_objects as go

import flodym as fd
from flodym.example_objects import get_example_mfa

mfa = get_example_mfa()

uncertainty = fd.UncertaintyDefinition(
    parameters={"eol machines": fd.LogNormalDistribution(relative_std=0.2)}
)

# %% [markdown]
# `run_monte_carlo()` runs a *model function*: any function which takes the parameters by name
# and returns the quantities of interest by name. `make_model_function()` turns an `MFASystem`
# into such a function. By default, it returns all flows, and each stock as `"<name>.stock"`,
# `"<name>.inflow"` and `"<name>.outflow"`. The seed makes the result reproducible.

# %%
model = fd.make_model_function(mfa)
result = fd.run_monte_carlo(model, mfa.parameters, uncertainty, n_samples=300, seed=42)
print(result)

# %% [markdown]
# Each output holds all samples. Mean, standard deviation and quantiles are computed for each
# value of the array. We compare them with the nominal run, the system computed with the nominal
# parameter values:

# %%
nominal = get_example_mfa()
nominal.compute()

scrap = result.outputs["shredder => remelting"]
print("Nominal in 2010:", nominal.flows["shredder => remelting"][{"t": 2010}].values)
print("Mean in 2010:", scrap.mean[{"t": 2010}].values)
print("Standard deviation in 2010:", scrap.std[{"t": 2010}].values)

# %% [markdown]
# ## Declaring uncertainty
#
# Mostly, several parameters are uncertain, each in its own way. Distributions are given
# relative to the nominal values, so that one line covers a whole array, and new data moves the
# distributions with it:
#
# - `NormalDistribution` and `LogNormalDistribution` have the nominal value as mean. Give `std`
#   or `relative_std`.
# - `UniformDistribution` lies between `lower` and `upper` (or `relative_lower` and
#   `relative_upper`).
# - `TriangularDistribution` lies between the bounds as well, with the nominal value as mode.
#
# `lower` and `upper` also truncate the normal and lognormal distribution, e.g. to keep a yield
# between zero and one. Truncation by bounds which are not symmetric around the nominal value
# shifts the mean away from the nominal value, e.g. a yield of 0.92 truncated at one has a lower
# mean.
#
# By default, all values of a parameter share one random draw: they are all at the same quantile
# of their distributions, so a time series is higher or lower as a whole. `independent_over`
# gives each item of a dimension its own draw. `where` restricts a distribution to a slice of the
# parameter.

# %%
uncertainty = fd.UncertaintyDefinition(
    parameters={
        # one draw per parameter: the whole time series is higher or lower
        "eol machines": fd.LogNormalDistribution(relative_std=0.2),
        "eol buildings": fd.LogNormalDistribution(relative_std=0.2),
        # yields are bounded; one draw per material
        "shredder yield": fd.ParameterUncertainty(
            distribution=fd.NormalDistribution(relative_std=0.05, lower=0.0, upper=1.0),
            independent_over="e",
        ),
        # only the copper yield of the demolition is uncertain
        "demolition yield": fd.ParameterUncertainty(
            distribution=fd.UniformDistribution(lower=0.01, upper=0.05), where={"e": "Cu"}
        ),
        # the composition keeps summing to one over the materials
        "composition eol machines": fd.TriangularDistribution(
            relative_lower=0.8, relative_upper=1.2
        ),
    },
    normalize_over={"composition eol machines": "e"},
)
result = fd.run_monte_carlo(model, mfa.parameters, uncertainty, n_samples=300, seed=42)

# %% [markdown]
# With a list of quantiles, the result gets a dimension `Quantile`:

# %%
steel = result.outputs["remelting => sysenv"]
bands = steel.quantile([0.05, 0.5, 0.95])
print("Dimensions:", bands.dims.letters)
bands[{"e": "Fe"}].to_df(dim_to_columns="Quantile").tail()

# %% [markdown]
# The 5 to 95 % band of the secondary iron, with the median and the nominal run. The median lies
# below the nominal run, since the lognormal distributions are skewed: their mean is the nominal
# value, their median is lower.

# %%
years = mfa.dims["t"].items
iron = bands[{"e": "Fe"}]
fig = go.Figure(
    [
        go.Scatter(x=years, y=iron[{"q": 0.95}].values, line={"width": 0}, showlegend=False),
        go.Scatter(
            x=years,
            y=iron[{"q": 0.05}].values,
            fill="tonexty",
            line={"width": 0},
            name="5 to 95 %",
        ),
        go.Scatter(x=years, y=iron[{"q": 0.5}].values, name="median"),
        go.Scatter(
            x=years,
            y=nominal.flows["remelting => sysenv"][{"e": "Fe"}].values,
            name="nominal run",
            line={"dash": "dash"},
        ),
    ]
)
fig.update_layout(title="Secondary iron", xaxis_title="Year", yaxis_title="kt/yr")
fig.show(renderer="notebook")

# %% [markdown]
# **Do not combine statistics.** The standard deviation or the 95th percentile of a sum or a ratio
# is not the sum or the ratio of the standard deviations or percentiles. Compute such quantities
# per sample: the samples can be sliced and summed, and the samples of outputs with the same sample
# dimension can be combined, as below; or compute them as derived quantities of the model (see
# below). For the same reason, the statistics keep the `extensive` flag of the output, but should
# not be summed over dimensions. For the copper content of the secondary steel in 2010:

# %%
copper = steel[{"e": "Cu", "t": 2010}]
total = steel[{"t": 2010}].sum_over(("e",))
copper_share = fd.SampledArray(samples=copper.samples / total.samples)
print(f"95th percentile of the copper content: {copper_share.quantile(0.95).values:.3%}")
ratio_of_percentiles = copper.quantile(0.95).values / total.quantile(0.95).values
print(f"Ratio of the 95th percentiles (wrong): {ratio_of_percentiles:.3%}")

# %% [markdown]
# ## One draw per parameter, or independent draws
#
# Whether values are drawn together or independently matters a lot for sums and cumulative
# quantities. Independent draws for each year average out over time, and give a much smaller
# uncertainty of the landfill stock, which accumulates the demolition losses of all years:

# %%
for label, independent_over in [("one draw", ()), ("independent draw per year", "t")]:
    definition = fd.UncertaintyDefinition(
        parameters={
            "eol buildings": fd.ParameterUncertainty(
                distribution=fd.LogNormalDistribution(relative_std=0.2),
                independent_over=independent_over,
            )
        }
    )
    runs = fd.run_monte_carlo(model, mfa.parameters, definition, n_samples=200, seed=1)
    final_stock = runs.outputs["landfills.stock"][{"t": 2010}].sum_over(("e",))
    relative_std = final_stock.std.values / final_stock.mean.values
    print(f"{label}: relative standard deviation of the landfill stock in 2010: {relative_std:.3f}")

# %% [markdown]
# Decide for each parameter which errors are shared: a systematic error of a data source is one
# draw; independent errors of separate measurements, e.g. per region, are independent draws.
# `independent_over` also accepts the letter of a parent dimension: with countries below world
# regions, `independent_over="w"` draws once per world region for all of its countries.
# `independent_over="all"` gives every value its own draw.
#
# ## Bounds and shares
#
# Sampled values stay within the bounds. `normalize_over` rescales the sampled values of a
# parameter so that their sum over the given dimensions keeps its nominal value, which is one for
# shares. Along these dimensions, values are drawn independently, since the rescaling would cancel
# a common draw. The bounds apply before the rescaling, so a rescaled value can lie outside them.
#
# `uncertainty.sample()` returns the sampled parameter values, exactly those used by
# `run_monte_carlo()` for the same seed:

# %%
inputs = uncertainty.sample(mfa.parameters, n_samples=300, seed=42)
print("Largest shredder yield:", inputs["shredder yield"].samples.values.max())
composition_sums = inputs["composition eol machines"].sum_over(("e",), check_extensivity=False)
print("Sums of the composition:", np.unique(composition_sums.samples.values.round(12)))

# %% [markdown]
# ## Tracking fewer outputs, and derived quantities
#
# All samples of all outputs are stored. For a large model, this needs a lot of memory; if the
# samples would need more than `max_memory_gib` (2 GiB by default), `run_monte_carlo()` raises an
# error which lists the largest outputs. Restrict the stored flows and stocks with `track`, and add
# derived quantities with `derived`, functions of the computed system. Here, we compute the copper
# content of the secondary steel per sample:


# %%
def copper_content(mfa: fd.MFASystem) -> fd.FlodymArray:
    secondary_steel = mfa.flows["remelting => sysenv"]
    return secondary_steel["Cu"] / secondary_steel.sum_over(("e",))


small_model = fd.make_model_function(
    mfa,
    track=["remelting => sysenv", "landfills.stock"],
    derived={"copper content": copper_content},
)
small_result = fd.run_monte_carlo(small_model, mfa.parameters, uncertainty, n_samples=300, seed=42)
print(list(small_result.outputs))
small_result.outputs["copper content"].quantile([0.05, 0.5, 0.95]).to_df(
    dim_to_columns="Quantile"
).tail()

# %% [markdown]
# `compute` replaces the call of `mfa.compute()`, e.g. by `mfa.compute_all_possible()` for systems
# computed from shares and prescribed flows.
#
# ## What the MFA system adapter does
#
# The function made by `make_model_function()` takes the parameters the system has when it is
# made, so make it after setting up the parameters. For every sample, it
#
# - writes all given parameter values into the parameters of the system, so that shares,
#   dimension splitters and prescribed flows linked to them use the sampled values; parameters
#   with integer values get float values during the run,
# - sets all flows and stocks to zero and unset, as in a new system, and computes the system,
# - returns copies of the tracked arrays and the derived quantities, and
# - restores the state the system had before, also if the computation fails.
#
# Lifetime model parameters are not reset. A lifetime model copies its parameters when they are
# assigned, so a lifetime assigned when the system is set up does not follow the sampled
# parameters. Assign lifetime parameters in `compute()`. If a dynamic stock is computed with the
# lifetime parameters it had before `compute()`, a warning is logged, or an error is raised with
# `flodym.config.error_behaviors.fixed_lifetime = "error"`.
#
# **Checks of the model.** The first sample is run twice; if the outputs differ, the model carries
# state from one run to the next or uses random numbers of its own, and `run_monte_carlo()` raises
# an error. Changes of the parameters in `compute()`, such as `self.parameters["x"] *= 1.5`, do not
# accumulate, since the adapter rewrites all parameters for every sample. If no output differs
# between the samples, e.g. because the model does not use the uncertain parameters, a warning is
# logged (see `flodym.config.error_behaviors.monte_carlo_no_variation`).
#
# ## Model functions without an MFA system
#
# Any function of the parameters is a model. It gets copies of all parameters, so it may modify
# them, derive new ones or build its own systems. Here, a single dynamic stock with an uncertain
# lifetime by region. The stock is created anew for each sample, so we pass its cohort dimension;
# otherwise, flodym would create one with a warning for each sample.

# %%
dims = fd.DimensionSet(
    dim_list=[
        fd.Dimension(name="Time", letter="t", dtype=int, items=list(range(2000, 2031))),
        fd.Dimension(name="Region", letter="r", dtype=str, items=["EUR", "USA", "CHN"]),
    ]
)
cohort = fd.Dimension(name="Cohort", letter="c", dtype=int, items=dims["t"].items)
stock_parameters = {
    "inflow": fd.Parameter(
        name="inflow", dims=dims, values=np.outer(np.linspace(1.0, 3.0, 31), [1.0, 1.5, 3.0])
    ),
    "lifetime": fd.Parameter(
        name="lifetime", dims=dims[("r",)], values=np.array([12.0, 15.0, 8.0])
    ),
}


def stock_model(parameters: dict[str, fd.FlodymArray]) -> dict[str, fd.FlodymArray]:
    lifetime = fd.NormalLifetime(
        dims=dims, mean=parameters["lifetime"], std=0.3 * parameters["lifetime"]
    )
    stock = fd.InflowDrivenDSM(
        dims=dims, inflow=parameters["inflow"], lifetime_model=lifetime, cohort_dim=cohort
    )
    stock.compute()
    return {"stock": stock.stock, "outflow": stock.outflow}


lifetime_uncertainty = fd.UncertaintyDefinition(
    parameters={
        "lifetime": fd.ParameterUncertainty(
            distribution=fd.NormalDistribution(relative_std=0.15, lower=1.0),
            independent_over="r",
        )
    }
)
stock_result = fd.run_monte_carlo(
    stock_model, stock_parameters, lifetime_uncertainty, n_samples=200, seed=1
)
stock_result.outputs["outflow"][{"t": 2030}].quantile([0.05, 0.95]).to_df(dim_to_columns="Quantile")

# %% [markdown]
# A model function is also the way to go for parameters derived outside `compute()`, e.g. a
# scenario factor applied to a parameter before the system is computed. Uncertain parameters are
# sampled before the model is called, so the derived parameter follows them:


# %%
def scenario_model(parameters: dict[str, fd.FlodymArray]) -> dict[str, fd.FlodymArray]:
    scenario_mfa = get_example_mfa()
    scenario_mfa.parameters.update(parameters)
    scenario_mfa.parameters["eol buildings"] = (
        parameters["eol buildings"] * parameters["building growth"]
    )
    scenario_mfa.compute()
    return {"landfills": scenario_mfa.stocks["landfills"].stock.sum_over(("e",))}


scenario_parameters = {
    **mfa.parameters,
    "building growth": fd.Parameter(
        name="building growth", dims=fd.DimensionSet(dim_list=[]), values=np.array(1.2)
    ),
}
scenario_uncertainty = fd.UncertaintyDefinition(
    parameters={"building growth": fd.UniformDistribution(lower=1.0, upper=1.4)}
)
scenario_result = fd.run_monte_carlo(
    scenario_model, scenario_parameters, scenario_uncertainty, n_samples=200, seed=3
)
print(
    "Landfill stock in 2010:",
    scenario_result.outputs["landfills"][{"t": 2010}].quantile([0.05, 0.95]).values,
)

# %% [markdown]
# ## Reproducibility
#
# The random numbers depend only on the seed, the parameter name, the entry of its uncertainty
# and the sample index. Adding, removing or reordering other uncertain parameters does not change
# the draws of a parameter, and a sample does not depend on which other samples are run. The
# random numbers between zero and one are the same on all platforms and numpy versions; sampled
# values and results agree up to floating-point differences of scipy and numpy functions.
#
# Without a seed, one is drawn and stored in the result as `result.seed`.
#
# To rerun a single sample, e.g. one with an unexpected result, give its index:

# %%
sample_17 = fd.run_monte_carlo(model, mfa.parameters, uncertainty, sample_indices=[17], seed=42)
np.testing.assert_array_equal(
    sample_17.outputs["remelting => sysenv"].samples.values[..., 0],
    steel.samples.values[..., 17],
)
inputs_17 = uncertainty.sample(mfa.parameters, sample_indices=[17], seed=42)
print("Shredder yield of sample 17:", inputs_17["shredder yield"].samples.values[:, 0])

# %% [markdown]
# ## Splitting a run
#
# flodym runs the samples one after another. To split a run, e.g. over the jobs of a job array,
# give each part its sample indices and the same seed, and join the parts. The joined result
# equals the one of a single run.

# %%
parts = [
    fd.run_monte_carlo(
        model, mfa.parameters, uncertainty, sample_indices=range(start, start + 100), seed=42
    )
    for start in (0, 100, 200)
]
joined = fd.MonteCarloResult.concat(parts)
np.testing.assert_array_equal(
    joined.outputs["landfills.stock"].samples.values,
    result.outputs["landfills.stock"].samples.values,
)

# %% [markdown]
# ## Failing samples
#
# By default, an error in a sample stops the run; the error is raised as it is, and the sample index
# and the seed are logged. With `on_error="skip"`, failing samples are left out. This is rejection
# sampling: the distribution of the remaining samples differs from the declared one. The result
# reports the rejection rate, and a warning is logged above 1 % (see
# `flodym.config.error_behaviors.monte_carlo_rejection`).


# %%
def strict_model(parameters: dict[str, fd.FlodymArray]) -> dict[str, fd.FlodymArray | float]:
    if parameters["shredder yield"].values.max() > 0.97:
        raise ValueError("shredder yield above 97 %")
    return model(parameters)


strict_result = fd.run_monte_carlo(
    strict_model, mfa.parameters, uncertainty, n_samples=100, seed=42, on_error="skip"
)
print(f"Rejection rate: {strict_result.rejection_rate:.0%}")
print("First failure:", next(iter(strict_result.failed_samples.items())))

# %% [markdown]
# ## Own distributions
#
# Other distributions are subclasses of `Distribution`. Declare their parameters as fields and
# implement `frozen()`, which returns a frozen `scipy.stats` distribution for the nominal values
# and the parameters as arrays. Bounds, broadcasting and sampling are done by flodym. Like the
# built-in distributions, an own distribution should be relative to the nominal value, here a
# beta distribution for shares with the nominal value as mean and a concentration parameter (the
# higher, the narrower). Optionally, `invalid_values()` checks the parameters and `no_spread()`
# keeps values at their nominal values:

# %%
import scipy.stats


class BetaDistribution(fd.Distribution):
    concentration: fd.FlodymArray | float

    def frozen(self, nominal, concentration):
        return scipy.stats.beta(nominal * concentration, (1 - nominal) * concentration)

    def invalid_values(self, nominal, concentration):
        return {"nominal value is not between 0 and 1": (nominal < 0) | (nominal > 1)}

    def no_spread(self, nominal, concentration):
        return (nominal == 0) | (nominal == 1)


beta_uncertainty = fd.UncertaintyDefinition(
    parameters={"remelting yield": BetaDistribution(concentration=50.0)}
)
beta_inputs = beta_uncertainty.sample(mfa.parameters, n_samples=200, seed=0)
print("Nominal:", mfa.parameters["remelting yield"].values)
beta_inputs["remelting yield"].quantile([0.05, 0.5, 0.95]).to_df(dim_to_columns="Quantile")
