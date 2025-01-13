# %% [markdown]
# # MFASystem
#
# ## The MFASystem attributes, explained
#
# The central class of flodym is the MFASystem class.
# In the following, its different parts are explained in more detail.
# If you are familiar with MFA modelling or ODYM, you may skim through or skip this HOWTO.
#
# ### Flows
#
# An MFA tracks mass flows of materials in a system (like the society of a defined region) though different stages of its lifecycle. In flodym, each of these stages is represented by one Flow object.
# It may be useful to track these flows differentiated by different dimensions, like different time steps and regions, but also different properties like elements, materials, or products they are contained in.
# The flows are therefore represented as multi-dimensional arrays (FlodymArrays in flodym).
# The dimensions can be different for different flows: During the product stage, the flows may be differentiated by the product they are contained in, while during the waste sorting stage, they may be differentiated by the waste category.
#
# ### Processes
#
# Processes are the transformations happening to the materials between one stage and another.
# In flodym, the process is a minimal object, containing only the name and ID of the process.
# If an MFA system is a network graph of flows, the processes are the nodes of this graph.
#
# Each flow therefore connects two processes, the source and the target process (The flow has a clear direction!).
#
# Counter-insuitively, whenever a transformation happens to a material, that is not represented by a flow, but only a process:
# For example, primary material production and fabrication could both be processes, while the flow connecting the two would be something like 'primary production to fabrication'.
# Think of a flow as something being transported from one facility to another, while a process is something happening in a facility.
#
# A process can have multiple input and/or output flows.
# For example, from the 'waste collection' process, one flow might go to 'recycling', one to 'landfilling' and another to 'incineration'.
# However, the different waste types might rather be represented by different dimensions of the flow.
#
# Each system has a system boundary.
# To represent flows leading in and out of this boundary, the first process in flodym is alwas the 'system environment' process.
# For example, there may be a flow from the system environment to a 'mining' process, where the materials enters the part of the material cycle represented in the MFA.
#
# ### Parameters
#
# In the above example, for calculating these flows, we have to know which share of which waste type enters recycling, landfilling and incineration. Possibly, these shares are different for different regions or time steps.
# So we need this information in multi-dimensional array form.
# This is what parameter objects are for. In flodym, they are also FlodymArray objects.
#
# ### Scalar parameters
#
# FlodymArrays can not be zero-dimensional.
# But parameters might be -- for example, if the share of incinerated waste is assumed to be constant across time, region and waste type.
#
# Therefore, there is an own attribute for scalar parameters in the MFASystem class.
#
# ### Stocks
#
# Sometimes materials will stay in a certain stage of their lifecycle for a longer time.
# To correctly account for this, stocks are used in dynamic MFAs.
#
# Stocks are defined by their inflow, outflow and amount of stock.
# In flodym, each of these quantities is represented by a FlodymArray object.
# Sometimes, lifetime distributions can be used to calculate the outflow from the stock.
# This is handled by the DynamicStockModel class in flodym.
# Details are given in the respective HOWTO.
#
# ### DimensionSet
#
# All different dimensions that at least one flow or stock is differentiated by are stored in a DimensionSet attribute of the MFASystem.
# Each dimension is defined by its name (like 'Element'), an index letter (like 'e') and a list of items (like ['Iron', 'Copper', 'Manganese']).
# A subset of these dimensions is also stored in each FlodymArray object.
# Further details are given in the following HOWTOs.
#
