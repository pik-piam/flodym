# %% [markdown]
# # Export and Plotting
#
# ## Export
#

# %% [markdown]
# ### Array to data frame
#
# `NamedDimArray` objects have a `to_df()` method, which exports them to a pandas DataFrame:

# %%
from sodym.example_objects import get_example_array

array = get_example_array()
df = array.to_df()
df.head()

# %% [markdown]
#
# ### MFA system to dictionary and pickle file
#
# The easiest way to store an MFA object is to pickle it (see the pickle documentation for details).
# However, whoever loads it from the pickle file needs sodym to read it.
#
# sodym has a method to convert the MFASystem object to a nested dictionary, which does not need sodym.
# There are two data types the array values can be converted to: numpy arrays and pandas DataFrames.
# Let's try both:

# %%
from sodym.export import convert_to_dict
from sodym.example_objects import get_example_mfa


# printing function
def print_types(dict_in):
    first_items = {
        k: next(iter(v.values())) if isinstance(v, dict) else v[0] for k, v in dict_in.items()
    }
    type_strings = [f"  {k}: {type(v)}" for k, v in first_items.items()]
    print("Keys and data types:")
    print("\n".join(type_strings))


mfa = get_example_mfa()
mfa.compute()

print("Pandas:")
df_dict = convert_to_dict(mfa, type="pandas")
print_types(df_dict)
print()


print("Numpy:")
np_dict = convert_to_dict(mfa, type="numpy")
print_types(np_dict)

# %% [markdown]
# Note that parameters are not included, as they are normally not computed within the MFA.
#
# You can write this dictionary to a pickle file with the following command.
# (You can also pickle the mfa directly, of course, but we haven't included that - it's just one line of code. Refer to the pickle documentation for more information.)

# %%
# turn on logging
import logging

logging.basicConfig(level=logging.INFO)

from sodym.export import export_mfa_to_pickle

export_path = "output_data/mfa.pickle"
export_mfa_to_pickle(mfa=mfa, export_path=export_path)

# %% [markdown]
# ## To CSV
#
# sodym has functions to export MFA flows and stocks to csv files. Let's start with flows:

# %%
import os
from sodym.export import export_mfa_flows_to_csv, export_mfa_stocks_to_csv

export_dir = "output_data/flows"
export_mfa_flows_to_csv(mfa, export_directory=export_dir)

# print files list
print("\nFiles in folder:\n")
print("\n".join(os.listdir(export_dir)))

# print header of first file
print("\nFirst file Header:\n")
with open(os.path.join(export_dir, os.listdir(export_dir)[0])) as input_file:
    head = [next(input_file) for _ in range(5)]
print("".join(head))

# %% [markdown]
# As you can see, the flow names are modified to make for valid file names.
#
# The same can be done for stocks:

# %%

import os
from sodym.export import export_mfa_stocks_to_csv

export_dir = "output_data/stocks"
export_mfa_stocks_to_csv(mfa, export_directory=export_dir, with_in_and_out=True)

# print files list
print("\nFiles in folder:\n")
print("\n".join(os.listdir(export_dir)))

# %% [markdown]
# If inflow and outflow printing is switched on, 3 files are created per stock.

# %% [markdown]
# ## Sankey plotting
#
# sodym can print Sankey plots of mfa systems:
#

# %%
from sodym.export import PlotlySankeyPlotter

plotter = PlotlySankeyPlotter(
    mfa=mfa, split_flows_by="Material", slice_dict={"t": 2000}, color_scheme="antique"
)
fig = plotter.plot()
fig.show(renderer="notebook")

# %% [markdown]
# This isn't the best-suited example for Sankey plotting, and the plotly customizability is limited, but it gives you a first impression of the flows.

# %% [markdown]
# ## Plotting yourself
#
# Obviously, you can always export a `NamedDimArray` to a DataFrame and use your own plotly routine, or use the numpy array in the `values` attribute.
# This is done in the examples.
#
# ## Array plotting using `ArrayPlotter`
#
# sodym has some implemented functionality that is especially suited for plotting 3-dimensional data.
# It can be handy if you want a quick impression of three-dimensional data (which is then differentiated by x-axis index, subplot, and lines).
# Two versions exist: One for pyplot, and one for plotly.
#
# Example 2 shows how to use it. Refer to the API reference for details.
