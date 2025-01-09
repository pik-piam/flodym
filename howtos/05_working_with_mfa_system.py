# %% [markdown]
# # Work with an MFA System
#
# As you just saw, `DimensionSet` and `NamedDimArray` objects work without the `MFASystem` class.
#
# However, working with it brings a few advantages:
# - Having all attributes in one namespace, including the "parent" dimension set.
# - Integrated data read-in functions (see according HOWTO)
# - Some more export and plotting functions (see according HOWTO)
# - A mass-balance check
#
# In more complex MFA models, it may be advantageous to have one or even several MFASystem class instances, and then maybe also do some calculations outside of them.
#
# ### Write your own subclass
#
# sodym has implemented the MFASystem class as a parent class, of which you can write your own subclass.
# We recommend implementing your own `compute` function, where the array operations are performed.
#
# Of course, this requires knowledge about the flows, stocks and parameters you intend to put in, which are described later.

# %%
from sodym import MFASystem


class MyMFASystem(MFASystem):

    def compute(self):

        self.flows["sysenv => process_a"][...] = self.parameters["extraction"]
        product_flow = self.flows["sysenv => process_a"] * self.parameters["product_shares"]
        self.flows["process_a => process_b"][...] = (
            product_flow * self.parameters["process_a_yield"]
        )
        self.flows["process_a => sysenv"][...] = product_flow * (
            1.0 - self.parameters["process_a_yield"]
        )
        self.flows["process_b => sysenv"][...] = self.flows["process_a => process_b"]


# %% [markdown]
# Of course, you can add further methods!
#
# For example, you can add sub-methods, which you then call from the `compute()` method.

# %% [markdown]
#
# ## Initialize an MFA System
#
# There are many different ways to initialize and MFA system.
#
# In this HOWTO, we only show the most direct one, where we create all needed attributes ourselves and pass them to the system.
#
# Other ways are shown in the "data input" HOWTO. They are actually recommended over the one presented here, as they make full use of the flexibility with regards to dimensions and dimension items in sodym.
#
# Let's prepare the attributes we need. We start with the dimensions:

# %%
from sodym import DimensionSet, Dimension, Flow, Parameter, Process

dims = DimensionSet(
    dim_list=[
        Dimension(name="Region", letter="r", items=["EU", "US", "MEX"]),
        Dimension(name="Product", letter="p", items=["A", "B"]),
        Dimension(name="Time", letter="t", items=[2020]),
    ]
)

# %% [markdown]
# ### Processes
#
# `Process` objects are used to define the "nodes" of the MFA system. They are quite simple and only contain a name and ID:
# The first process, i.e. the one with ID 0 is always the system environment and has to be called "sysenv".
# Other names lead to an error when the MFASystem is created.
#
# Processes (as well as flows, stocks, and parameters) are given as a dictionary, the keys of which should correspond to the names of the objects for clarity, even though this is not a requirement.

# %%
processes = {
    "sysenv": Process(name="sysenv", id=0),
    "process_a": Process(name="process_a", id=1),
    "process_b": Process(name="process_b", id=2),
}

# %% [markdown]
# We now set up the flows dictionary.
# Normally, we do not initialize flows with values.
# Instead, their values enter their system through parameters, from which they are transferred to the flows.
# This makes data input easier, if data is read from files. If you're writing your own data input, you can deviate from this, it's not a requirement.
#
# The following code is a bit cumbersome and information for each flow is repeated.
# There are functions to shorten and ease setting up the flow dictionary.
# But let's do it manually first:

# %%
flows = {
    "sysenv => process_a": Flow(
        name="sysenv => process_a",
        dims=dims["r", "t"],
        from_process=processes["sysenv"],
        to_process=processes["process_a"],
    ),
    "process_a => process_b": Flow(
        name="process_a => process_b",
        dims=dims,
        from_process=processes["process_a"],
        to_process=processes["process_b"],
    ),
    "process_a => sysenv": Flow(
        name="process_a => sysenv",
        dims=dims,
        from_process=processes["process_a"],
        to_process=processes["sysenv"],
    ),
    "process_b => sysenv": Flow(
        name="process_b => sysenv",
        dims=dims["r", "t"],
        from_process=processes["process_b"],
        to_process=processes["sysenv"],
    ),
}

# %% [markdown]
# We initialize the parameters with data:

# %%
import numpy as np

parameters = {
    "extraction": Parameter(name="extraction", dims=dims["r", "t"], values=3.0 * np.ones((3, 1))),
    "product_shares": Parameter(
        name="product_shares", dims=dims[("p",)], values=np.array([0.6, 0.4])
    ),
    "process_a_yield": Parameter(
        name="process_a_yield", dims=dims[("p",)], values=np.array([0.8, 0.9])
    ),
}

# %% [markdown]
# We leave out stocks for simplicity here. We look at that in its own HOWTO.
#
# We have everything we need to initialize the system!

# %%
my_mfa_system = MyMFASystem(
    dims=dims,
    processes=processes,
    flows=flows,
    parameters=parameters,
)

# %% [markdown]
# ## Work with the system
#
# All that's left to do is call the compute function!

# %%
my_mfa_system.compute()

# print example output
for f in my_mfa_system.flows.values():
    print(f.name, "\n", f.to_df(), "\n")

# %% [markdown]
# We now call a method to check the mass balance of the system. If all is set up correctly, and no mass flows appear or disappear, this should not create errors. It's a nice safety check, but it's not a requirement. Thanks to Stefan Pauliuk, who implemented this in ODYM.
#
# The success message is implemented using an `info` of the `logging` package, so we have to configure it to show this message first. This is optional. The error is always shown if it fails.

# %%
import logging

# turn on success message
logging.basicConfig(level=logging.INFO)

my_mfa_system.check_mass_balance()

# %% [markdown]
# ## Generate attributes from Definition objects
#
# As you saw, some of the attributes were a bit cumbersome to create.
#
# sodym has definition objects, which store exactly the information you need to create them, and methods that take these definitions to produce the dictionaries.
#
# For processes, all that's needed for definition is a list of names

# %%
from sodym import make_processes

process_names = [
    "sysenv",
    "process_a",
    "process_b",
]
processes = make_processes(process_names)

# %% [markdown]
# For flows, there is a dedicated definition object:

# %%
from sodym import FlowDefinition

flow_definitions = [
    FlowDefinition(from_process_name="sysenv", to_process_name="process_a", dim_letters=("r", "t")),
    FlowDefinition(
        from_process_name="process_a", to_process_name="process_b", dim_letters=("r", "p", "t")
    ),
    FlowDefinition(
        from_process_name="process_a", to_process_name="sysenv", dim_letters=("r", "p", "t")
    ),
    FlowDefinition(from_process_name="process_b", to_process_name="sysenv", dim_letters=("r", "t")),
]

# %% [markdown]
# It can then be transformed, for which the actual `DimensionSet` and `Process` objects are needed:

# %%
from sodym import make_empty_flows

flows = make_empty_flows(processes=processes, flow_definitions=flow_definitions, dims=dims)

print("Flow names:\n ", "\n  ".join(flows))

# %% [markdown]
# As you can see, the flow names were automatically created from the process names.
# If you don't like this, you can either choose a different naming function, write your own (not shown), or override the names in the `FlowDefinition` with the `name_override` attribute:

# %%
from sodym.flow_naming import (
    # shown below
    process_ids,
    # the default
    process_names_with_arrow,
    # not shown here. Good to generate valid file and variable names.
    process_names_no_spaces,
)

# other naming function
flows_a = make_empty_flows(
    processes=processes,
    flow_definitions=flow_definitions,
    dims=dims,
    naming=process_ids,
)
print("Flow names with ids:\n ", "\n  ".join(flows_a), "\n")

# custom name
flow_definitions_b = [
    FlowDefinition(
        from_process_name="sysenv",
        to_process_name="process_a",
        dim_letters=("r", "t"),
        name_override="my_custom_name",
    )
]
flows_b = make_empty_flows(
    processes=processes,
    flow_definitions=flow_definitions_b,
    dims=dims,
)
print("Custom flow name(s):\n ", "\n  ".join(flows_b))

# %% [markdown]
# Dimensions and Parameters also have Definition objects.
# However, they are used with data read-in functions. They are discussed in the according HOWTO.
#
# Stock definition objects are discussed in the stock HOWTO.
