# %% [markdown]
# # Data input
#
# Before, we have covered how to initialize the attributes of an MFASystem directly with their attributes.
# You can, of course, write your own data input routines and do this.
#
# Here, we will discuss how to directly read data into flodym objects.
#
# There are lots of different methods with different levels of integration vs customizability.
#
# ## From DataFrame
#
# FlodymArray objects provide the `from_df()` method.
#
# You can create your own data read-in to a csv file, and use this method.

# %%
import numpy as np
import pandas as pd

import flodym as fd

dims = fd.DimensionSet(
    dim_list=[
        fd.Dimension(letter="t", name="Time", dtype=int, items=[2010, 2020]),
        fd.Dimension(letter="e", name="Material", dtype=str, items=["Fe", "Cu", "Mn"]),
    ]
)
df = pd.DataFrame(
    {
        "Time": [2020, 2020, 2020, 2010, 2010, 2010],
        "Material": ["Mn", "Cu", "Fe", "Fe", "Cu", "Mn"],
        "value": [6.0, 5.0, 4.0, 1.0, 2.0, 3.0],
    }
)
my_array = fd.FlodymArray.from_df(dims=dims, df=df)

print("Shape:", my_array.shape)
print("Values:", my_array.values.flatten())

# %% [markdown]
# Notice how the entries were re-ordered according to the order of items in the dims.
# the `from_df()` methods performs lots of checks and can handle different input data formats.
#
# For details on allowed formats, see the API reference of the method.
#
# NB: `Dimension` objects also have `from_np()` and `from_df` methods, which can be combined with numpy and pandas read functions.
# Refer to the API reference for usage.

# %% [markdown]
#
# ## From file: CSV & Excel
#
# DimensionSets can be loaded using a `DimensionReader`.
#
# Why would you put the items of the dimension in an external file instead of putting them in the code directly?
# In order to change them later together with the other input data.
# Like this, you can easily switch between different regional resolutions, temporal scope, and so on, without changing the code.
#
#
# There are two dimension readers implemented, an `ExcelDimensionReader` and a `CSVDimensionReader`
# We only show the `ExcelDimensionReader` here. For the  `CSVDimensionReader`, refer to the API reference.
#
# When working with reader, the information needed in addition to the data (for dimensions, everything apart from the items) is given in a definition object:

# %%
dimension_definitions = [
    fd.DimensionDefinition(name="Time", letter="t", dtype=int),
    fd.DimensionDefinition(name="Material", letter="e", dtype=str),
]


# %% [markdown]
# We initialize the reader with dictionaries containing the paths and sheet names for each dimension.
# Here, we've put everything in the same file, and named the sheets the same as the dimensions.

# %%

dimension_file = "../examples/input_data/example2_dimensions.xlsx"
dimension_files = {d.name: dimension_file for d in dimension_definitions}
dimension_sheets = {d.name: d.name for d in dimension_definitions}
reader = fd.ExcelDimensionReader(
    dimension_files=dimension_files,
    dimension_sheets=dimension_sheets,
)

# %% [markdown]
# Every DimensionReader has a `read_dimensions` method, which takes the list of definitions as input and returns the DimensionSet:

# %%

dims = reader.read_dimensions(dimension_definitions)

# %% [markdown]
# ParameterReader objects work exactly the same: First, we create the definitions:

# %%
parameter_definitions = [
    fd.ParameterDefinition(name="eol machines", dim_letters=("t",)),
    fd.ParameterDefinition(name="eol buildings", dim_letters=("t",)),
    fd.ParameterDefinition(name="composition eol machines", dim_letters=("e",)),
    fd.ParameterDefinition(name="composition eol buildings", dim_letters=("e",)),
    fd.ParameterDefinition(name="shredder yield", dim_letters=("e",)),
    fd.ParameterDefinition(name="demolition yield", dim_letters=("e",)),
    fd.ParameterDefinition(name="remelting yield", dim_letters=("e",)),
]

# %% [markdown]
# We initialize the reader with dictionaries for file names and sheets names:

# %%
parameter_file = "../examples/input_data/example2_parameters.xlsx"
parameter_files = {p.name: parameter_file for p in parameter_definitions}
parameter_sheets = {p.name: p.name for p in parameter_definitions}
reader = fd.ExcelParameterReader(
    parameter_files=parameter_files,
    parameter_sheets=parameter_sheets,
)

# %% [markdown]
# Every ParameterReader has a `read_parameters()` method. Apart from the definitions, it also takes the DimensionSet object as input, as it needs information on the dimensions.

# %%
parameters = reader.read_parameters(parameter_definitions=parameter_definitions, dims=dims)

# %% [markdown]
# You can specify for csv and excel readers whether or not to throw an error if items are missing or not via the `allow_missing_values` argument.
# Setting it to False is good for avoiding errors such as typos in the input data. Setting it to True is necessary for sparse data.
# The default is False.
#
# Similarly, you can specify whether or not to throw an error if there are unknown dimension items in the input data via the `allow_extra_values` argument.
# Setting it to False is good for avoiding errors such as typos in the input data.
# Setting it to True is necessary if you'd like to be able to run the model with a subset of the items in the input data, e.g. a shorter time period, without changing the input data.
# The default is False.
#
# Instead of a boolean, both arguments also accept a list of parameter names. The setting is then only enabled for these parameters, e.g. `allow_missing_values=["p1"]`.

# %% [markdown]
# ## MFASystem `from_excel()` and `from_csv`
#
# If you wish to do your dat input using either excel or csv files as shown above, you can list all definitions, combine that into an MFADefinition object, and pass that into the `MFASystem.from_excel()` or `from_csv()` class methods together with the dictionaries for file paths (and sheet names for excel).
#
# It's a very clean, easy and quick way, but not very customizable.
# This method is used in example 2 and therefore not repeated here.

# %% [markdown]
# ## MFASystem `from_definition()`: without a data reader
#
# Some dimension items follow from the model configuration rather than from data, e.g. the
# years of the time dimension. They can be given in the dimension definition with `items`.
# Items which may change between model runs can instead be given in a dictionary to
# `MFASystem.from_definition()`, keyed by dimension letter or name.
# For a dimension with a parent, the dictionary maps each item to its parent item.

# %%
definition = fd.MFADefinition(
    dimensions=[
        fd.DimensionDefinition(name="Time", letter="t", dtype=int, items=[2020, 2030]),
        fd.DimensionDefinition(name="World region", letter="w", dtype=str),
        fd.DimensionDefinition(name="Country", letter="c", dtype=str, parent="w"),
    ],
    processes=[
        "sysenv",
        fd.ProcessDefinition(name="use", outflow_shares={"waste": "waste share"}),
        "waste",
    ],
    flows=[
        fd.FlowDefinition(
            from_process_name="sysenv",
            to_process_name="use",
            dim_letters=("t", "c"),
            prescribed="demand",
        ),
        fd.FlowDefinition(from_process_name="use", to_process_name="waste", dim_letters=("t", "c")),
        fd.FlowDefinition(
            from_process_name="use", to_process_name="sysenv", dim_letters=("t", "c")
        ),
        fd.FlowDefinition(
            from_process_name="waste", to_process_name="sysenv", dim_letters=("t", "c")
        ),
    ],
    parameters=[
        fd.ParameterDefinition(name="demand", dim_letters=("t", "c")),
        fd.ParameterDefinition(name="waste share", dim_letters=("t",)),
    ],
)
mfa = fd.MFASystem.from_definition(
    definition,
    dimension_items={"w": ["EUR", "USA"], "Country": {"DEU": "EUR", "FRA": "EUR", "USA": "USA"}},
)

# %% [markdown]
# The parameters are created with zero values and are not set yet (`is_set` is False).
# Using a parameter which is not set raises an error (see
# `flodym.config.error_behaviors.unset_parameter`).
# Their values are given with `set_parameter_values()`, as numpy arrays, `FlodymArray`s or
# DataFrames. The values are written into the existing parameters, so they are used by the
# processes and flows right away.

# %%
mfa.set_parameter_values(
    {
        "demand": np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
        "waste share": np.array([0.2, 0.3]),
    }
)
mfa.compute_all_possible()
mfa.flows["use => waste"].to_df()

# %% [markdown]
# Instead of dictionaries, both methods also take a function, with the signature of
# `read_dimension()` or `read_parameter_values()` of the readers described below. So the
# methods of a reader can be passed directly, e.g.
# `mfa.set_parameter_values(reader.read_parameter_values)`. The function is called for all
# parameters which are not set yet.

# %% [markdown]
#
# ## Write your own customized subclasses
#
# ### of `DimensionReader` and `ParameterReader`
#
# There are parent classes of `DimensionReader` and `ParameterReader` that you can write your own subclasses for.
# - In a subclass of the `DimensionReader`, you will have to provide the function `read_dimension` (singular, not plural!), which takes a `DimensionDefinition` object and returns a `DimensionObject`.
# - In a subclass of the `ParameterReader`, you will have to provide the function `read_parameter_values`, which takes a parameter name and the correct `DimensionSet` according to the letters given in the definition, and returns a `Parameter`.
# For both, you can provide additional information (such as file paths) in the `__init__` method.
#
# ### of `DataReader`
#
# There is also a combined `DataReader` class, which contains the methods described above for both dimension and parameter reading.
# If you have your own custom subclass of this, you can pass it to the `MFASystem.from_data_reader()`.
# This method is elegant, but required writing your own subclass, which may not be straightforward as other methods. It is shown in Example 5, so it is not demonstrated here.
#
# ###  Using `CompoundDataReader`
#
# As a final note, there is also a `CompoundDataReader`, which combines a `DimensionReader` and a `ParameterReader` into an integrated `DataReader`. You don't need a subclass, just pass the other reader instances at initialization. It's useful if you want to mix different methods without re-implementing them. Endless possibilities!
