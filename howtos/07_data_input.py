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
import pandas as pd
from flodym import Dimension, DimensionSet, FlodymArray

dims = DimensionSet(
    dim_list=[
        Dimension(letter="t", name="Time", dtype=int, items=[2010, 2020]),
        Dimension(letter="e", name="Material", dtype=str, items=["Fe", "Cu", "Mn"]),
    ]
)
df = pd.DataFrame(
    {
        "Time": [2020, 2020, 2020, 2010, 2010, 2010],
        "Material": ["Mn", "Cu", "Fe", "Fe", "Cu", "Mn"],
        "value": [6., 5., 4., 1., 2., 3.],
    }
)
my_array = FlodymArray.from_df(dims=dims, df=df)

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
from flodym import ExcelDimensionReader, DimensionDefinition

dimension_definitions = [
    DimensionDefinition(name="Time", letter="t", dtype=int),
    DimensionDefinition(name="Material", letter="e", dtype=str),
]


# %% [markdown]
# We initialize the reader with dictionaries containing the paths and sheet names for each dimension. 
# Here, we've put everything in the same file, and named the sheets the same as the dimensions. 

# %%

dimension_file = "../examples/input_data/example2_dimensions.xlsx"
dimension_files = {d.name: dimension_file for d in dimension_definitions}
dimension_sheets = {d.name: d.name for d in dimension_definitions}
reader = ExcelDimensionReader(
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
from flodym import ExcelParameterReader, ParameterDefinition

parameter_definitions = [
    ParameterDefinition(name="eol machines", dim_letters=("t",)),
    ParameterDefinition(name="eol buildings", dim_letters=("t",)),
    ParameterDefinition(name="composition eol machines", dim_letters=("e",)),
    ParameterDefinition(name="composition eol buildings", dim_letters=("e",)),
    ParameterDefinition(name="shredder yield", dim_letters=("e",)),
    ParameterDefinition(name="demolition yield", dim_letters=("e",)),
    ParameterDefinition(name="remelting yield", dim_letters=("e",)),
]

# %% [markdown]
# We initialize the reader with dictionaries for file names and sheets names: 

# %%
parameter_file = "../examples/input_data/example2_parameters.xlsx"
parameter_files = {p.name: parameter_file for p in parameter_definitions}
parameter_sheets = {p.name: p.name for p in parameter_definitions}
reader = ExcelParameterReader(
    parameter_files=parameter_files,
    parameter_sheets=parameter_sheets,
)

# %% [markdown]
# Every ParameterReader has a `read_parameters()` method. Apart from the definitions, it also takes the DimensionSet object as input, as it needs information on the dimensions.

# %%
parameters = reader.read_parameters(parameter_definitions=parameter_definitions, dims=dims)

# %% [markdown]
# ## MFASystem `from_excel()` and `from_csv`
#
# If you wish to do your dat input using either excel or csv files as shown above, you can list all definitions, combine that into an MFADefinition object, and pass that into the `MFASystem.from_excel()` or `from_csv()` class methods together with the dictionaries for file paths (and sheet names for excel). 
#
# It's a very clean, easy and quick way, but not very customizable. 
# This method is used in example 2 and therefore not repeated here.

# %% [markdown]
#
# ## Write your own customized subclasses
#
# There are parent classes of `DmensionReader` and `ParamaterReader` that you can write your own subclasses for.
# - In a subclass of the `DimensionReader`, you will have to provide the function `read_dimension` (singular, not plural!), which takes a `DimensionDefinition` object and returns a `DimensionObject`. 
# - In a subclass of the `ParameterReader`, you will have to provide the function `read_parameter_values`, which takes a parameter name and the correct `DimensionSet` according to the letters given in the definition, and returns a `Parameter`.
# For both, you can provide additional information (such as file paths) in the `__init__` method.
#
# There is also a combined `DataReader` class, which contains the methods described above for both dimension and parameter reading. 
# If you have your own custom subclass of this, you can pass it to the `MFASystem.from_data_reader()`. 
# This method is elegant, but required writing your own subclass, which may not be straightforward as other methods. It is shown in Example 5, so it is not demonstrated here.
#
# As a final note, there is also a `CompoundDatReader`, which combines a `DimensionReader` and a `ParamaterReader` into an integrated `DataReader`. It's useful if you want to mix different methods without re-implementing them. Endless possibilities!
