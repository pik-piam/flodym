# %% [markdown]
# # Dimensions and DimensionSets
#
# ## Dimensions
#
# [Dimension](api.dimensions.html#flodym.Dimension) objects are what flodym datasets are differentiated by.
#
# They are defined by a name, index letter, and their items.
# The index letters must have length one, i.e. only be one letter.
#
# Optionally, a data type can be given. If doing so, it is ensured that all items have this data type.
# When loading items from file, it can be useful to check that for example years are integers and not converted to strings.

# %%
from flodym import Dimension

regions = Dimension(name="Region", letter="r", items=["EU", "US", "MEX"])

# %% [markdown]
#
# Never give Dimension objects with different item sets the same name or letter, even though it is tempting!

# %%
# historic and future time
time = Dimension(name="Time", letter="t", items=["2010", "2020", "2030"])
# Historic time - a subset of time

# WRONG! (same letter t)
historic_time = Dimension(name="Historic Time", letter="t", items=["2010", "2020"])

# Correct
historic_time = Dimension(name="Historic Time", letter="h", items=["2010", "2020"])

# %% [markdown]
# ## DimensionSets
#
# Multiple dimensions are grouped in a DimensionSet object:

# %%
from flodym import DimensionSet

dims = DimensionSet(
    dim_list=[
        Dimension(name="Region", letter="r", items=["EU", "US", "MEX"]),
        Dimension(name="Product", letter="p", items=["A", "B"]),
        Dimension(name="Time", letter="t", items=[2020]),
    ]
)

# %% [markdown]
# The MFASystem class has a dims attribute, which is a DimensionSet object containing all dimensions ever used in the system.
#
# Each FlodymArray object also has a dims object, with a subset of dimensions that this array is defined over.
#
# Subsets and single dimensions can be extracted by indexing:

# %%
item_by_letter = dims["r"]
print(f"Item by letter | type: {type(item_by_letter)}, name: {item_by_letter.name}")

item_by_name = dims["Region"]
print(f"Item by name | type: {type(item_by_name)}, name: {item_by_name.name}")

subset = dims["r", "p"]
print(f"Subset | type: {type(subset)}, names: {subset.names}")

# %% [markdown]
# DimensionSet objects have several attributes (like `names` or `letters`) and a range of methods to modify or combine them.
# For a full list and documentation, refer to the API Reference.
