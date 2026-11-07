# %% [markdown]
# # Stocks
#
# `Stock` objects are flodym's way to account for accumulating materials.
# Each `Stock` object has three arrays as attributes: `inflow`, `outflow`, and `stock`.
# They are instances of the `FlodymArray` subclass `StockArray`.
#
# There are different kinds of `Stock` (i.e. different subclasses), which differ in which of the above three quantities are given, which is calculated from the others, and how.
#
# ## Input requirements
#
# The first dimension of the dimension set `dims` must be time. If the time dimension does not have the dimension letter `t`, you must specify that via the `time_letter` argument.
# If you do not pass the three arrays to the constructor, they are created and initialized to zero.
# If you do pass them, they must have the same `dims` as the stock object.
#
# You can attach stocks to a process (which is needed for example for `MFASystem.check_mass_balance`), but you don't have to.
#
# ## Simple, from constructor
#
# The simplest stock is the `SimpleFlowDrivenStock`.
# Here, inflow and/or outflow are given, and the stock is calculated as the cumulative sum over time of inflow minus outflow.

# %%
from flodym import SimpleFlowDrivenStock, Dimension, DimensionSet

dims = DimensionSet(
    dim_list=[
        Dimension(letter="t", name="Time", dtype=int, items=list(range(2000, 2011))),
        Dimension(letter="p", name="Product", dtype=str, items=["Vehicles", "Buildings", "Other"]),
    ]
)

my_stock = SimpleFlowDrivenStock(dims=dims, name="in-use")

# example output
print("Inflow type:", type(my_stock.inflow))
print("Outflow shape:", my_stock.outflow.shape)
print("Stock sum:", my_stock.stock.sum_values())
print(f"Array names: {my_stock.stock.name}, {my_stock.inflow.name}, {my_stock.outflow.name}")


# %% [markdown]
# Now you can work with the stock.
# Every stock has a `compute()` routine, which depends on the type of stock you choose.
#
# As mentioned above, for `SimpleFlowDrivenStock`, the stock is calculated as the cumulative sum over time of inflow minus outflow.
#
# After `compute()`, the stock, inflow and outflow arrays are marked as set (see the HOWTO on arrays).
# Before the computation, flodym checks the arrays the computation is based on: a warning is logged if they are zero.
# What happens if they are not set is controlled by `flodym.config.error_behaviors.unset_array_used`.
# The check and the marking are added automatically to the `compute()` method of every `Stock` subclass, including your own.
#
# The `check_mass_balance()` method checks whether inflow, outflow and stock are consistent, i.e. whether the stock change in each time step equals the net inflow over the time step.
# The mass balance error itself is returned by `get_mass_balance()`.
# With `flodym.config.checks.mass_balance_after_compute = True`, the check is performed automatically after each computation.
# What happens if it fails, and the tolerance it uses, are controlled by `flodym.config` as well (see the HOWTO on working with the MFA system):

# %%
my_stock.inflow.values[...] = 0.1
my_stock.outflow.values[...] = 0.01
my_stock.compute()

print("Stock values for Vehicles:", my_stock.stock["Vehicles"].values)

my_stock.check_mass_balance()

# %% [markdown]
# ## Dynamic Stock Models (DSM)
#
# The other `Stock` subclasses are all dynamic stock models (DSM).
# This means they rely on a lifetime model to compute some of their quantities.
# There are different lifetime models available. There are two ways of passing the lifetime model to the stock.
# The first is to just specify the lifetime model class at initialization, and set its parameters later.
# Lifetimes can be set as `FlodymArray`s or as scalars. As `FlodymArray`s, they can have lower dimensionality than the stock, and will be cast to full dimensionality.

# %%
import numpy as np
from flodym import InflowDrivenDSM, NormalLifetime, Parameter

my_dsm = InflowDrivenDSM(
    dims=dims,
    name="in-use-dsm",
    lifetime_model=NormalLifetime,
)
lifetime_mean = Parameter(dims=dims[("p",)], values=np.array([10, 20, 5]))
my_dsm.inflow.values[...] = 0.1
my_dsm.lifetime_model.set_prms(mean=lifetime_mean, std=0.3 * lifetime_mean)
my_dsm.compute()

print("Stock values for Vehicles:", my_dsm.stock["Vehicles"].values)

# %% [markdown]
# Alternatively, the lifetime model can be initialized with its parameters, and the instance can be passed to the Stock.
# In this example, let's also try passing the inflows to the Stock at initialization.

# %%
from flodym import StockArray

lifetime_model = NormalLifetime(dims=dims, mean=lifetime_mean, std=0.3 * lifetime_mean)
inflow = StockArray(dims=dims, name="in-use-dsm_inflows", values=0.1 * np.ones(dims.shape))
my_dsm = InflowDrivenDSM(
    dims=dims,
    name="in-use-dsm",
    lifetime_model=lifetime_model,
    inflow=inflow,
)
my_dsm.compute()

print("Stock values for Vehicles:", my_dsm.stock["Vehicles"].values)

# %% [markdown]
# ## `StockDefinition` objects
#
# If you are working with definition objects for your MFA system, you can also use stock definitions:

# %%
from flodym import StockDefinition, make_empty_stocks, make_processes

stock_defs = [
    StockDefinition(
        dim_letters=("t", "p"),
        name="in-use",
        subclass=InflowDrivenDSM,
        lifetime_model_class=NormalLifetime,
        process_name="use_phase",
    )
]

processes = make_processes(["sysenv", "use_phase"])

stocks = make_empty_stocks(stock_definitions=stock_defs, processes=processes, dims=dims)

# still zero, as not yet computed
print("Stock values for Vehicles:", stocks["in-use"].stock["Vehicles"].values)


# %% [markdown]
# For an example of how to use stock definitions in the `MFASystem.from_data_reader()` method, see Example 5.

# %% [markdown]
# ## Computation modes
#
# `InflowDrivenDSM` and `StockDrivenDSM` fix whether `compute()` starts from the inflow or from the stock.
# The general `DynamicStockModel` instead has one method for each mode:
# `compute_inflow_driven()`, `compute_stock_driven()` and `compute_inflow_by_cohort_driven()`.
# The latter starts from `inflow_by_cohort`, an inflow by year and age-cohort, for which a `cohort_dim` must be given.
#
# These methods can be called repeatedly and in any order on the same object.
# For example, a stock-driven computation yields negative inflows if the stock decreases faster than items reach the end of their lifetime.
# These inflows can be set to zero, and the stock recomputed inflow-driven:

# %%
from flodym import DynamicStockModel

stock_values = np.array([1, 2, 3, 4, 5, 6, 6, 3, 2, 2, 2], dtype=float)[:, np.newaxis]
my_dsm = DynamicStockModel(
    dims=dims,
    name="in-use-dsm",
    lifetime_model=NormalLifetime(dims=dims, mean=lifetime_mean, std=0.3 * lifetime_mean),
    stock=StockArray(dims=dims, values=stock_values * np.ones(dims.shape)),
)
my_dsm.compute_stock_driven()
print("Smallest inflow, stock-driven:", my_dsm.inflow.values.min())

my_dsm.inflow[...] = np.maximum(my_dsm.inflow.values, 0.0)
my_dsm.compute_inflow_driven()
print("Smallest inflow, corrected:", my_dsm.inflow.values.min())
print("Stock values for Vehicles:", my_dsm.stock["Vehicles"].values)

# %% [markdown]
# `DynamicStockModel.compute()` chooses the mode itself: it uses whichever of stock, inflow and `inflow_by_cohort` is set (see `is_set` in the how-to on arrays).
# After a computation, stock, inflow and outflow are all set, so a second call to `compute()` raises an error.
# To recompute, use the explicit methods above.

# %% [markdown]
# ## Lifetime models
#
# The following lifetime models are available:
# - `FixedLifetime`: all items of a cohort leave the stock at the age `mean`.
# - `NormalLifetime`, `FoldedNormalLifetime`, `LogNormalLifetime`: distributions with `mean` and
#   standard deviation `std`.
# - `WeibullLifetime`: Weibull distribution with `weibull_scale` and `weibull_shape`.
# - `WeibullMeanStdLifetime`: Weibull distribution with `mean` and `std`, from which shape and
#   scale are computed internally.
# - `ExponentialLifetime`: constant hazard rate, i.e. the same share of the remaining items leaves
#   per time unit, independent of their age. Parameter `mean`.
# - `Leaching`: the same, but parametrized by `loss_rate`, the share of the stock leaving per
#   time unit (e.g. per year).
# - `TabulatedLifetime`: the cumulative survival is given directly by year and age-cohort, e.g.
#   from statistics or from an external model, as `cumulative_survival_table`.
#
# For the models with `mean` and `std`, setting `relative_std=True` makes `std` relative to the
# mean, i.e. the coefficient of variation std / mean:

# %%
from flodym import LogNormalLifetime

lifetime_model = LogNormalLifetime(dims=dims, mean=lifetime_mean, std=0.3, relative_std=True)

# %% [markdown]
# Several independent causes of leaving the stock can be combined with `CompetingRisksLifetime`,
# which takes a list of lifetime models with the same dimensions.
# The cumulative survival of the models is multiplied.
# For example, items can reach the end of their product lifetime, while some material is
# additionally lost from the stock every year, independent of the product age:

# %%
from flodym import CompetingRisksLifetime, Leaching

lifetime_model = CompetingRisksLifetime(
    models=[
        LogNormalLifetime(dims=dims, mean=lifetime_mean, std=0.3, relative_std=True),
        Leaching(dims=dims, loss_rate=0.05),
    ]
)
my_dsm = InflowDrivenDSM(dims=dims, lifetime_model=lifetime_model, inflow=inflow)
my_dsm.compute()

print("Stock values for Vehicles, with leaching:", my_dsm.stock["Vehicles"].values)

# %% [markdown]
# A lifetime model provides three arrays by year t and age-cohort c:
# - `cumulative_survival`: the share of the inflow of cohort c still present at the end of year t.
# - `conditional_survival`: the share of cohort c present at the end of year t-1 which is still
#   present at the end of year t.
# - `outflow_pmf`: the share of the inflow of cohort c leaving the stock during year t, i.e. the
#   probability mass function of the outflow time.
#
# They can be retrieved as `FlodymArray`s with `get_cumulative_survival()`,
# `get_conditional_survival()` and `get_outflow_pmf()`. By default, the cohort dimension has the
# letter `c`; a different one can be passed as `cohort_dim`.
#
# `get_mean_lifetime_by_cohort()` returns the mean time the items of each cohort stay in the
# stock, including all effects described in this how-to. As survival beyond the end of the time
# horizon is unknown, `tail` must be chosen: `"none"` ignores it (lower bound), and
# `"constant_rate"` extrapolates with the conditional survival of the last time step.

# %%
cumulative_survival = lifetime_model.get_cumulative_survival()
print("Dimensions of the cumulative survival:", cumulative_survival.dims.letters)

# With the short time horizon of this example, a large share of cohort 2000 is still present at
# the end, so the two options for the tail differ strongly.
print("Share of cohort 2000 present in 2010:", cumulative_survival[{"t": 2010, "c": 2000}].values)
for tail in ["none", "constant_rate"]:
    mean_lifetime = lifetime_model.get_mean_lifetime_by_cohort(tail=tail)
    print(f"Mean lifetime of cohort 2000, tail '{tail}':", mean_lifetime[{"t": 2000}].values)

# %% [markdown]
# ## Lifetime extension
#
# Lifetime extension can by modeled. To this end, a time-dependent extension factor can be specified.
# The lifetime distribution will then be scaled such that mean and standard deviation are scaled
# by this factor. The factor can be specified in the lifetime model class. There are two parameters
# for different kinds of lifetime extensions, which are also distinguished in
# Krych et al., 2024 - https://doi.org/10.1111/jiec.13586:
# - `lt_factor_by_cohort`: Applies the extension to all items of a cohort (which can be roughly
#   translated to one production year). Use this factor to model lifetime extension trough product
#   design, dubbed as "nature" lifetime extension in Krych et al.
# - `lt_factor_by_year`: Applies the extension in a given year to all items which are present in
#   the stock in that year. Use this factor to model lifetime extension trough repair and extended
#   use, dubbed as "nurture" lifetime extension in Krych et al.
# The following example illustrates the effect of both. For details, refer to the referenced
# publication.

# %%
import numpy as np

from flodym import Dimension, DimensionSet, StockArray, FlodymArray, WeibullLifetime, StockDrivenDSM

YEARS = list(range(2025, 2071))

# Initialize dimensions
# The "extension type" dimension is used to handle different types of extension in one stock object.
dim_list = [
    Dimension(
        name="time",
        letter="t",
        items=YEARS,
    ),
    Dimension(
        name="extension type",
        letter="p",
        items=["base", "extension by year", "extension by cohort"],
    ),
]
dims = DimensionSet(dim_list=dim_list)

# Initialize the prescribed stock
stock_over_time = 1 - np.exp(-0.1 * (np.array(YEARS) - 2025))
stock_values = np.ndarray((len(YEARS), 3))
stock_values[...] = stock_over_time[:, np.newaxis]
stock = StockArray(dims=dims, values=stock_values)

# Initialize prescribed extension factors
# linear increase from 1 to 2 between 2030 and 2040, then constant at 2
factor_np = 1 + np.clip((np.array(YEARS) - 2030) / 10, 0, 1)

factor = FlodymArray(dims=dims)
factor[...] = 1.0

factor_by_year = factor.copy()
factor_by_year["extension by year"] = factor_np

factor_by_cohort = factor.copy()
factor_by_cohort["extension by cohort"] = factor_np

lifetime_model = WeibullLifetime(
    dims=dims,
    time_letter="t",
    weibull_scale=10,
    weibull_shape=2,
    lt_factor_by_year=factor_by_year,
    lt_factor_by_cohort=factor_by_cohort,
)

dsm = StockDrivenDSM(
    dims=dims,
    stock=stock,
    lifetime_model=lifetime_model,
    time_letter="t",
)
dsm.compute()

# %% [markdown]
# The visualization shows how the extension by year immediately reduces the outflow, as it
# applies to all items in the stock. For the extension by cohort, the outflow is only reduced
# gradually, as only newly produced cohorts have the extended lifetimes, and their outflow
# becomes relevant only as they age.

# %%
import plotly.express as px

outflow = dsm.outflow.to_df().reset_index()

fig = px.line(
    y=factor_np,
    x=np.array(YEARS),
    title="Extension factor",
)
fig.show(renderer="notebook")

fig = px.line(
    outflow,
    y="value",
    x="time",
    color="extension type",
    title="Stock Outflow by Extension Type",
)
fig.show(renderer="notebook")

# %% [markdown]
# ## Period effects
#
# Period effects change the outflow in a given year for all age-cohorts present in that year.
# Two kinds are available as attributes of all lifetime models:
# - `hazard_factor_by_year`: multiplies the hazard, i.e. the intensity of leaving the stock.
#   The conditional survival s in that year becomes s ** factor. A factor of 1 has no effect,
#   0 means no outflow (e.g. a demolition moratorium), and values greater than 1 increase the
#   outflow (e.g. a scrappage scheme). For Weibull distributions, this is equivalent to
#   `lt_factor_by_year` with hazard factor = lt_factor ** (-weibull_shape).
# - `removal_share_by_year`: a share of the stock removed in a time step in addition to the
#   regular outflow, independent of age, e.g. by a disaster, or by a phase-out policy (share 1).
#
# Both are applied to the conditional survival, from which the cumulative survival is computed.

# %%
dims = DimensionSet(
    dim_list=[
        Dimension(name="time", letter="t", items=YEARS),
        Dimension(name="scenario", letter="s", items=["base", "scrappage scheme", "disaster"]),
    ]
)
stock = StockArray(dims=dims, values=stock_over_time[:, np.newaxis] * np.ones(dims.shape))

# triple exit intensity in 2035 and 2036
hazard_factor = FlodymArray(dims=dims)
hazard_factor[...] = 1.0
hazard_factor[{"s": "scrappage scheme", "t": [2035, 2036]}] = 3.0

# a fifth of the stock is destroyed in 2040
removal_share = FlodymArray(dims=dims)
removal_share[{"s": "disaster", "t": 2040}] = 0.2

lifetime_model = LogNormalLifetime(
    dims=dims,
    mean=10,
    std=0.3,
    relative_std=True,
    hazard_factor_by_year=hazard_factor,
    removal_share_by_year=removal_share,
)
dsm = StockDrivenDSM(dims=dims, stock=stock, lifetime_model=lifetime_model)
dsm.compute()

fig = px.line(
    dsm.outflow.to_df().reset_index(),
    y="value",
    x="time",
    color="scenario",
    title="Stock outflow with period effects",
)
fig.show(renderer="notebook")


# %% [markdown]
#
# ## Lifetime model inflow time
# The DSMs consist of discrete time steps (the items of the time dimension), leading to
# numerical errors (cf. Cencic and Frühwirth, 2025 - https://doi.org/10.1111/jiec.70102).
# The standard implementation assumes all inflow of a time step to occur at one point in time.
# The inflow can be set with the `inflow_at` attribute of the lifetime model to occur at the
# beginning, end, or middle of the time step.
# The default is the middle, which is generally the most accurate.
# For short lifetimes (similar length or shorter than the time step), all these methods will yield
# large discrepancies between the inflow and the stock.
# To avoid this, several points in time can be used per time interval.
# This corresponds to a numerical integration over the time step.
# It is more accurate than a single point, but also slower. The number can be set with the
# `n_pts_per_interval` attribute of the lifetime model. Up to 10 points can be used.
#
# ## Uneven time spacing
# The items of the time dimension can be non-contiguous and even unevenly spaced.
# For example, only select years can be used.
# If the time steps are not contiguous, the flows are assumed to be annual flows.
# So a stock inflow of 1 within a time step of 5 years will lead to a stock increase of 5.
#
# If the spacing is uneven, the interval lengths are calculated as follows:
# Each given time step is assumed to be within a time interval.
# The interval bounds are in the center between two time steps.
# The interval length of the first and last time step is assumed to be the same as the second and
# second-to-last time step, respectively.
# For example, if the given items of the time dimension are `[2000, 2005, 2010, 2020, 2030]`,
# Then the interval bounds are `[1997.5, 2002.5, 2007.5, 2015, 2025]`, and the interval lengths
# thus `[5, 5, 7.5, 10, 10]`.
# This means that a constant inflow of 1 (neglecting outflow for simplicity) will lead to a stock
# increase of 5, 5, 7.5, 10, and 10 in the respective time steps.
