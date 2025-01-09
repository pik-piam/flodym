# %% [markdown]
# # Working with NamedDimArrays
#
# ## Initializing arrays
#
# `NamedDimArray` objects require a `DimensionSet` at initialization. Optionally, a name can be given.
# If the values are not given, the array is initialized with zeros.
#
# There are several subclasses of `NamedDimArray`, often with little or no changes in functionality:
# See the API reference of `Flow`, `Parameter`, and `StockArray`.
#
# `Flow` object have to be passed the two `Process` objects they connect at initialization.
#
# In this HOWTO, only the `NamedDimArray` base class is used.
#
# Further options to initialize arrays are discussed in the HOWTO on data input.
#

# %%
import numpy as np
from sodym import Dimension, DimensionSet, NamedDimArray

# Create a dimension set
dims = DimensionSet(
    dim_list=[
        Dimension(name="Region", letter="r", items=["EU", "US", "MEX"]),
        Dimension(name="Product", letter="p", items=["A", "B"]),
        Dimension(name="Time", letter="t", items=[2020]),
    ]
)

flow_a = NamedDimArray(dims=dims["t", "p"], values=0.2 * np.ones((1, 2)))
flow_b = NamedDimArray(dims=dims["r", "t"], values=0.1 * np.ones((3, 1)))
parameter_a = NamedDimArray(dims=dims["r", "p"], values=0.5 * np.ones((3, 2)))


# %% [markdown]
#
# ## Math operations
#
# NamedDimArrays have the basic mathematical operations implemented.
# Let#s first create two arrays:

# %% [markdown]
# We write a small routine to print properties of the resulting array, and test it on the inputs:


# %%
def show_array(arr: NamedDimArray):
    print(f"  dimensions: {arr.dims.letters}")
    print(f"  shape: {arr.dims.shape()}")
    print(f"  name: {arr.name}")
    print(f"  values mean: {np.mean(arr.values):.3f}")
    print(f"  values sum: {arr.values.sum():.3f}")


print("flow_a:")
show_array(flow_a)
print("flow_b:")
show_array(flow_b)

# %% [markdown]
# Now let's try some operations.

# %%
summed = flow_a + flow_b
print("flow_a:")
show_array(summed)

# %% [markdown]
# What happened here?
# When adding the two flows, all dimensions that could be preserved were preserved.
# These are the dimensions that occur in both `flow_a` and `flow_b`, in this case only time.
#
# Since we wouldn't know how to split `flow_a` by region and `flow_b` by product, we have to sum the arrays to the set intersection of both arrays, and then perform the addition.
#
# The same goes for subtraction:

# %%
difference = flow_a - flow_b
print("difference:")
show_array(difference)

# %% [markdown]
# For multiplication and division, things are different.
# If we multiply a flow with a parameter, which splits it along a new dimension, the resulting flow can have that new dimension.
# Therefore, in multiplication and division, we keep all the dimensions that appear in either of the flows, i.e. the set union.

# %%
# recall:
print("flow_a dimensions: ", flow_a.dims.letters)
print("parameter_a dimensions: ", parameter_a.dims.letters, "\n")

multiplied = flow_a * parameter_a
print("multiplied:")
show_array(multiplied)

divided = flow_a / parameter_a
print("divided:")
show_array(divided)

# %% [markdown]
# This may not be the dimension we want, for example we might want to sum the result over products, keeping the dimensions tie and region. There are some class methods for these kinds of operations. See the API reference for the full documentation. For our example:

# %%
reduced = multiplied.sum_nda_to(result_dims=("t", "r"))
print("reduced:")
show_array(reduced)

# %% [markdown]
# ### With scalars
#
# Math operations can also be performed between a NamedDimArray and a scalar.
# The scalar is then expanded into the shape of the array before the operation is performed:

# %%
sum_with_scalar = flow_a + 0.4

print("SUm with scalar:")
show_array(sum_with_scalar)

# %% [markdown]
# ### Using just the `values` array
#
# When a mathematical operation is not implemented, you can still work with the `values` array manually, which is a numpy array. We recommend using either the numpy ellipsis slice `[...]` or the `NamedDimArray.set_values()` method, which both ensure keeping the correct shape of the array.

# %%
flow_a.values[...] = 0.3
print("flow_a:")
show_array(flow_a)

flow_a.set_values(flow_a.values**2)
print("flow_a:")
show_array(flow_a)

# %% [markdown]
# ## Computing values of existing arrays, such as flows
#
# In a sodym MFASystem, you have defined at initialization which arrays have which dimensionality.
# You can use that information to conveniently sum the result of an operation to the shape you defined, potentially re-ordering dimensions.
#
# This is done using the so-called ellipsis slice `[...]`:

# %%
# define and initialize values with zero
predefined_flow = NamedDimArray(name="predefined", dims=dims["r", "p"])
print("predefined_flow:")
show_array(predefined_flow)

# recall:
multiplied = flow_a * parameter_a
print("multiplied:")
show_array(multiplied)

# set values of predefined_flow to the values of multiplied
predefined_flow[...] = flow_a * parameter_a

print("predefined_flow:")
show_array(predefined_flow)

# %% [markdown]
# In a sodym MFASystem, this is a bit tricky, but quite important, as the flows are stored as a dictionary.
# (For simplicity, we only re-create these dictionaries, not the whole MFASystem)

# %%
flows = {
    "flow_a": flow_a,
    "flow_b": flow_b,
    "predefined_flow": predefined_flow,
}
parameters = {
    "parameter_a": parameter_a,
}

# %% [markdown]
# The correct way to perform an operation here, is using the ellipsis slice on the left side of an assignment, as this only affects the values of the `NamedDimArray` object:

# %%
flows["predefined_flow"][...] = flows["flow_a"] * parameters["parameter_a"]
print("predefined_flow:")
show_array(flows["predefined_flow"])

# %% [markdown]
# While the following __wrong__ code without the ellipsis slice will overwrite the whole object, with uncontrolled outcome:

# %%
flows["predefined_flow"] = flows["flow_a"] * parameters["parameter_a"]
print("WRONG predefined_flow:")
show_array(flows["predefined_flow"])

# %% [markdown]
# ## Slicing
#
# Sometimes, we don't want to access the whole array, but just a slice.
# We can do this with indexing.
#
# We can use indexing on the right-hand side of an assignment to only calculate with part of an array, and on the left-hand side, to only set the values of part of an array.
#
# Let's look at "getting" a slice first:

# %%
# recall
print("flow_a dimensions: ", flow_a.dims.letters)

slice_a1 = flow_a["A"]
print("slice_a1:")
show_array(slice_a1)

# %% [markdown]
# You can also slice along several dimensions at the same time.
# If you like to be more specific, you can also give the slice indexes as a dictionary.
# This is actually necessary if an item appears in several dimensions, such that giving only the item would be ambiguous.

# %%
slice_a2 = flow_a["A", 2020]
print("slice_a2:")
show_array(slice_a2)

slice_a3 = flow_a[{"t": 2020}]
print("slice_a3:")
show_array(slice_a3)

slice_a4 = flow_a[{"t": 2020, "p": "A"}]
print("slice_a4:")
show_array(slice_a4)

# %% [markdown]
# As you can see, zero-dimensional NamedDimArrays are possible.
#
# Note that numpy indexing of the whole object like `flow_a[0, :]` is not supported, as sodym wouldn't know if in `flow_a[2020]`, `2020` is an index or an item of the dimension.
#
# Of course, you can slice the values array: `flow_a.values[:,0]`.
# But we recommend not to do it. One major design goal of sodym is too keep the code flexible to changes in the dimensions, and `flow_a.values[:,0]` is quite inflexible with respect to the order and number of dimensions in the array, and to the order and number of items in the dimensions.

# %% [markdown]
# The slices we looked at just take one item along a dimension and drop that dimension in the process.
# If we want to access several items along one dimension, that creates a problem, as the dimension can't be dropped, but is changed, as it does not contain all items of the original one anymore. To cope with that, we have to create a new dimension object with a new name and letter, and pass it to the slice, along with the dimension letter we're taking a subset of:

# %%
regions_na = Dimension(name="RegionsNA", letter="n", items=["US", "MEX"])

slice_b1 = flow_b[{"r": regions_na}]
print("slice_a5:")
show_array(slice_b1)

# %% [markdown]
# As mentioned earlier, You can also use slicing to only access a par of the array on the left-hand side of an assignment:

# %%
flow_b["EU"] = flow_a["A"]
print("flow_b.values:\n", flow_b.values)

# %% [markdown]
# On the left-hand side, it is also possible to access several items along one dimension, with the same syntax. It does not change the shape of the flow.

# %%
flow_b[{"r": regions_na}] = flow_b[{"r": regions_na}] * 3
print("flow_b.values:\n", flow_b.values)
print("flow_b:")
show_array(flow_b)

# %% [markdown]
# ## Operation rules summary
#
# Let's summarize here the rules for dimension handling:
#
# - Additions and subtractions yield the set intersection of the two participating arrays.
# - Multiplications and divisions yield the set union of the participating arrays.
# - When setting the values of an existing array, the array on the right-hand side of the assignment is summed down to the dimensions of the left-hand side. Missing dimensions on the right-hand side will lead to an error
# - Scalars are converted to an array of equal dimensions before the operation is performed.
#
# ### Caveat
#
# We found these rules to yield the right behavior in almost all cases.
#
# There are exceptions: When adding two dimensionless parameters with different dimensions, it may be intended that the dimensions of both inputs are still used.
#
# A sodym extension is planned to account for this. In the meantime, we advise to use the `NamedDimArray.cast_to()` method on the arrays before performing the operation.
#
