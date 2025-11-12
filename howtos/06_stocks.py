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
# There's a `check_mass_balance()` method, which checks whether inflow, outflow and stock are consistent:

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
#
# ## Lifetime model inflow time
# The DSMs consist of discrete time steps (the items of the time dimension), leading to
# numerical errors.
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
# # Uneven time spacing
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
