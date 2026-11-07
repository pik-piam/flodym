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
# Each FlodymArray object also has a dims attribute, with a subset of dimensions that this array is defined over.
#
# Subsets and single dimensions can be extracted by indexing. A single index in square brackets returns a `Dimension`.
# Several indexes in the square brackets, or a tuple (even with a single item inside), return a `DimensionSet`.

# %%
item_by_letter = dims["r"]
print(f"Item by letter | type: {type(item_by_letter)}, name: {item_by_letter.name}")

item_by_name = dims["Region"]
print(f"Item by name | type: {type(item_by_name)}, name: {item_by_name.name}")

subset = dims["r", "p"]
print(f"Subset | type: {type(subset)}, names: {subset.names}")

subset_size_1_tuple = dims[("r",)]
print(f"Size 1 subset | type: {type(subset_size_1_tuple)}, names: {subset_size_1_tuple.names}")

# %% [markdown]
# DimensionSet objects have several attributes (like `names` or `letters`) and a range of methods to modify or combine them.
# For a full list and documentation, refer to the API Reference.
# Here we list the few most important ones.

# %%

# Transform single Dimension to DimensionSet
dim_as_set = dims["r"].as_dimset()
print(f"Dimension as DimensionSet | type: {type(dim_as_set)}, names: {dim_as_set.names}")

# Set union of two DimensionSets
combined_dims = dims["r", "p"] | dims["p", "t"]
print(f"Combined DimensionSet | type: {type(combined_dims)}, names: {combined_dims.names}")

# Set intersection of two DimensionSets
intersected_dims = dims["r", "p"] & dims["p", "t"]
print(f"Intersected DimensionSet | type: {type(intersected_dims)}, names: {intersected_dims.names}")

# Remove dimensions from DimensionSet
reduced_dims = dims - dims["t"]
print(f"Reduced DimensionSet | type: {type(reduced_dims)}, names: {reduced_dims.names}")


# %% [markdown]
# ## Sub-dimensions
#
# A dimension can be a finer resolution of another dimension, called its parent.
# Each item is assigned to exactly one item of the parent.
# Typical examples are countries assigned to world regions, or building types split into sub-types.
#
# A mapping from items to parent items can be given directly:

# %%
world_regions = Dimension(name="World region", letter="w", items=["EUR", "USA"])
countries = Dimension.from_mapping(
    name="Country",
    letter="c",
    parent=world_regions,
    mapping={"DEU": "EUR", "FRA": "EUR", "USA": "USA"},
)
print(countries)

# %% [markdown]
# Alternatively, some items of the parent can be split into the items of other dimensions.
# Items which are not split are kept. Several levels of splitting are possible.

# %%
building_types = Dimension(name="Building type", letter="b", items=["Com", "Res"])
residential_types = Dimension(name="Residential type", letter="y", items=["Sin", "Mul"])
building_sub_types = building_types.split(
    name="Building sub-type", letter="s", splitting={"Res": residential_types}
)
print(building_sub_types.items)

# %% [markdown]
# Parent and sub-dimension have different letters, so arrays can be defined over either of them.
#
# Arrays can be aggregated to a coarser dimension with `aggregate_to()` and disaggregated to a finer
# one with `disaggregate_to()`.
# Whether weights are needed depends on the kind of quantity:
#
# - Extensive quantities such as flows are summed when aggregating, but need shares or weights when
#   splitting.
# - Intensive quantities such as material intensities are copied when splitting, but need weights
#   for the weighted mean when aggregating.
#
# Weights are normalized within each parent item, so they can be shares or absolute values, e.g.
# production volumes.

# %%
import numpy as np

from flodym import FlodymArray

production = FlodymArray(
    dims=building_types.as_dimset(), values=np.array([1.0, 10.0]), name="production"
)
shares = FlodymArray(dims=building_sub_types.as_dimset(), values=np.array([1.0, 0.8, 0.2]))

production_by_sub_type = production.disaggregate_to(building_sub_types, weights=shares)
print(production_by_sub_type.values)

material_intensity = FlodymArray(
    dims=building_sub_types.as_dimset(), values=np.array([5.0, 10.0, 20.0])
)
mean_intensity = material_intensity.aggregate_to(building_types, weights=production_by_sub_type)
print(mean_intensity.values)

# %% [markdown]
# Operations between arrays over related dimensions follow the general flodym rule:
# Reducing dimensions sums, and expanding dimensions copies.
#
# - `+`, `-`, `minimum()`, `maximum()`, `sum_to()` and assignments such as `coarse[...] = fine`
#   aggregate to the coarser dimension by summation.
# - `*`, `/` and `cast_to()` disaggregate to the finer dimension by copying values.
#
# This is correct for the common cases of adding flows and multiplying them with intensive
# parameters. Splitting extensive quantities and aggregating intensive quantities is never done
# automatically; use the methods above with weights.

# %%
material_demand = production_by_sub_type * material_intensity
print(material_demand.dims.letters, material_demand.values)

material_demand_by_type = FlodymArray(dims=building_types.as_dimset())
material_demand_by_type[...] = material_demand
print(material_demand_by_type.values)

# %% [markdown]
# When setting values, an item of the parent addresses all items belonging to it.
# This is not possible when getting values, as no dimension is created implicitly for the result.

# %%
material_intensity[{"s": "Res"}] = 15.0
print(material_intensity.values)

# %% [markdown]
# When reading dimensions with a data reader, sub-dimensions are defined in the
# `DimensionDefinition` by the letter of their `parent`.
# The dimension file then has two columns: the items, and the parent item of each item.
# A split dimension is defined by `splitting`, which gives the dimension each parent item is split
# into. It is not read from a file, but created from the other dimensions.

# %%
from flodym import DimensionDefinition

dimension_definitions = [
    DimensionDefinition(name="World region", letter="w", dtype=str),
    DimensionDefinition(name="Country", letter="c", dtype=str, parent="w"),
    DimensionDefinition(name="Building type", letter="b", dtype=str),
    DimensionDefinition(name="Residential type", letter="y", dtype=str),
    DimensionDefinition(
        name="Building sub-type", letter="s", dtype=str, parent="b", splitting={"Res": "y"}
    ),
]
