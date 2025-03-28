# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0rc1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Example 1. System with two processes, two parameters, one material.
# *ODYM example by Stefan Pauliuk, adapted for flodym*
#
# A simple MFA system with one material, a time horizon of 30 years (1980-2010), two processes, and a time-dependent parameter is analysed.
#
# <img src="pictures/MFAExample1.png" width="404" height="490" alt="Diagram of the MFA system">
#
# The model equations are as follows:
# * $a(t) = D(t)$ (exogenous input flow)
# * $d(t) = \alpha (t)\cdot b(t)$ (recovery efficiency parameter)
# * $a(t) + d(t) = b(t)$ (mass balance process 1)
# * $b(t) = c(t) + d(t)$ (mass balance process 2)
#
# From these equations the system solution follows:
# * $c(t) = a(t) = D(t)$
# * $b(t) = \frac{1}{1-\alpha (t)}\cdot D(t)$
# * $c(t) = \frac{\alpha}{1-\alpha (t)}\cdot D(t)$
#

# %% [markdown]
# ## 1. Load flodym and other useful packages

# %%
import numpy as np
import plotly.express as px

from flodym import (
    Dimension,
    DimensionSet,
    Process,
    Parameter,
    FlowDefinition,
    MFASystem,
    make_empty_flows,
)

# %% [markdown]
# ## 2. Load data
# Normally data would be loaded from a file / database, but since this is just a small example, the values are input directly into the code below.

# %%
time = Dimension(name="Time", letter="t", items=list(range(1980, 2011)))
elements = Dimension(
    name="Elements",
    letter="e",
    items=[
        "single material",
    ],
)
dimensions = DimensionSet(dim_list=[time, elements])

parameters = {
    "D": Parameter(name="inflow", dims=dimensions, values=np.arange(0, 31).reshape(31, 1)),
    "alpha": Parameter(
        name="recovery rate",
        dims=dimensions,
        values=np.arange(2, 33).reshape(31, 1) / 34,
    ),
}

processes = {
    "sysenv": Process(name="sysenv", id=0),
    "process 1": Process(name="process 1", id=1),
    "process 2": Process(name="process 2", id=2),
}

# %% [markdown]
# ## 3. Define flows and initialise them with zero values.
# By defining them with the shape specified by the relevant dimensions, we can later ensure that when we update the values, these have the correct dimensions.
# Note that flows are automatically assigned their names based on the names of the processes they are connecting.

# %%
flow_definitions = [
    FlowDefinition(
        from_process_name="sysenv", to_process_name="process 1", dim_letters=("t", "e")
    ),  # input
    FlowDefinition(
        from_process_name="process 1",
        to_process_name="process 2",
        dim_letters=("t", "e"),
    ),  # consumption
    FlowDefinition(
        from_process_name="process 2", to_process_name="sysenv", dim_letters=("t", "e")
    ),  # output
    FlowDefinition(
        from_process_name="process 2",
        to_process_name="process 1",
        dim_letters=("t", "e"),
    ),  # recovered material
]
flows = make_empty_flows(processes=processes, flow_definitions=flow_definitions, dims=dimensions)


# %% [markdown]
# ## 4. Define the MFA System equations
# We define a class with our system equations in the compute method. Afterwards we create an instance of this class, using the input data defined above. The class (system equations) can then easily be reused with different input data.
#
# We just need to define the compute method with our system equations, as all the other things we need are inherited from the MFASystem class.


# %%
class SimpleMFA(MFASystem):
    def compute(self):
        self.flows["sysenv => process 1"][...] = self.parameters[
            "D"
        ]  # the elipsis slice [...] ensures the dimensionality of the flow is not changed
        self.flows["process 1 => process 2"][...] = (
            1 / (1 - self.parameters["alpha"]) * self.parameters["D"]
        )
        self.flows["process 2 => sysenv"][...] = self.parameters["D"]
        self.flows["process 2 => process 1"][...] = (
            self.parameters["alpha"] / (1 - self.parameters["alpha"]) * self.parameters["D"]
        )


# %%
mfa_example = SimpleMFA(
    dims=dimensions, processes=processes, parameters=parameters, flows=flows, stocks={}
)
mfa_example.compute()

# %%
flow_a = mfa_example.flows["sysenv => process 1"].to_df()
flow_a = flow_a.reset_index(level="Elements", drop=True)
fig = px.line(flow_a, title="sysenv => process 1")
fig.show(renderer="notebook")

# %%
flow_b = mfa_example.flows["process 1 => process 2"].to_df()
flow_b = flow_b.reset_index(level="Elements", drop=True)
fig = px.line(flow_b, title="process 1 => process 2")
fig.show(renderer="notebook")

# %%
