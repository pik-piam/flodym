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
# There's a `check_stock_balance()` method, which checks whether inflow, outflow and stock are consistent:

# %%
my_stock.inflow.values[...] = 0.1
my_stock.outflow.values[...] = 0.01
my_stock.compute()

print("Stock values for Vehicles:", my_stock.stock["Vehicles"].values)

my_stock.check_stock_balance()

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
#
# For the models with `mean` and `std`, setting `relative_std=True` makes `std` relative to the
# mean, i.e. the coefficient of variation std / mean:

# %%
from flodym import LogNormalLifetime

lifetime_model = LogNormalLifetime(dims=dims, mean=lifetime_mean, std=0.3, relative_std=True)

# %% [markdown]
# Several independent causes of leaving the stock can be combined with `competing_risk`,
# which takes another lifetime model with the same dimensions.
# The survival rates of both models are multiplied.
# For example, items can reach the end of their product lifetime, while some material is
# additionally lost from the stock every year, independent of the product age:

# %%
from flodym import Leaching

lifetime_model = LogNormalLifetime(
    dims=dims,
    mean=lifetime_mean,
    std=0.3,
    relative_std=True,
    competing_risk=Leaching(dims=dims, loss_rate=0.05),
)
my_dsm = InflowDrivenDSM(dims=dims, lifetime_model=lifetime_model, inflow=inflow)
my_dsm.compute()

print("Stock values for Vehicles, with leaching:", my_dsm.stock["Vehicles"].values)

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
# The visualization shows how the extension by cohort immediately reduces the outflow,
# While for the extension by year, the outflow is only reduced after newly produced cohorts with
# the extended lifetimes have reached the end of their lifetime.

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
