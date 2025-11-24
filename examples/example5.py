# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0.dev0
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Example 5. Estimating the material content of the global vehicle fleet
# *ODYM example by Stefan Pauliuk, adapted for flodym*
#
# This example shows a fully-fledged MFA system, designed to estimate the material composition of the global passenger vehicle fleet in 2017, covering 130 countries, 25 age-cohorts, and 25 materials.
#
# The research questions asked are: How big is the material stock currently embodied in the global passenger vehicle fleet, and when will this material become available for recycling?
#
# To answer these questions a dynamic material flow analysis of the global passenger vehicle fleet and the waste management industries is performed.
#
# The dynamic MFA model has the following indices:
#
# * t: time (1990-2017)
# * c: age-cohort (1990-2017)
# * r: region (130 countries accounting for most of the global vehicle fleet)
# * g: good (passenger vehicle)
# * p: process (vehicle markets, use phase, waste management industries, scrap markets)
# * m: engineering materials (25)
# * e: chemical elements (all)
# * w: waste types (steel, Al, Cu scrap, plastics, glass, and other waste)
#
# The system definition of the model is given in the figure below. The data availability limits the temporal scope to 2017. The figure also shows the aspects of the different system variables. The total registration of vehicles, for example, is broken down into individual materials, whereas the flow of deregistered vehicles is broken down into regions, age-cohorts, and materials.
#
# <img src="pictures/System_example5.png" width="850" height="290" alt="MFA system">
#
# The model equations are as follows:
#
# 1) inflow-driven dynamic stock model, where $F_{1-2}$ is the historic inflow, $Sf$ is the survival function of the age-cohort (1-sum(pdf of discard)), and $S_2$ is the stock:
# $S_2(t,c,r,g) = F_{1-2}(c,r,g)\cdot Sf(t,c,r,g)$
# 2) Calculation of difference between inflow-driven stock (covering only the age-cohorts 2005-2017 due to data availability) and the 2015 reported stock and distribution of the difference to the years 1990-2005 (constant inflow assumed for these years)
# 3) Calculation of material composition of the fleet $S_2$ with
# $S_2(t,c,r,g,m) = \mu(c,r,g,m)\cdot S_2(t,c,r,g)$
# 4) Estimation of available future end-of-life vehicle scrap $F_{3-4}$ with
# $F_{3-4}(r,g,w,m) = \sum_{t,c}EoL_eff(r,g,m,w)\cdot M(t,c,r,g,m)$
#
# The remaining system variables are calculated by mass balance.

# %% [markdown]
# ## 1. Load flodym and other useful packages

# %%
from os.path import join

import numpy as np
import pandas as pd
import plotly.express as px

from flodym import (
    DimensionDefinition,
    Dimension,
    DimensionSet,
    ParameterDefinition,
    Parameter,
    FlowDefinition,
    StockDefinition,
    MFASystem,
    MFADefinition,
    DataReader,
    InflowDrivenDSM,
)
from flodym.lifetime_models import NormalLifetime

# %% [markdown]
# ## 2. Define the data requirements, flows, stocks and MFA system equations
#
# We define the dimensions that are relevant for our system and the model parameters, processes, stocks and flows. We further define a class with our system equations in the compute method.

# %%
dimension_definitions = [
    DimensionDefinition(letter="t", name="time", dtype=int),
    DimensionDefinition(letter="r", name="region", dtype=str),
    DimensionDefinition(letter="m", name="material", dtype=str),
    DimensionDefinition(letter="w", name="waste", dtype=str),
]

process_names = ["sysenv", "market", "use", "waste", "scrap"]

parameter_definitions = [
    ParameterDefinition(name="vehicle lifetime", dim_letters=("r",)),
    ParameterDefinition(name="vehicle material content", dim_letters=("m",)),
    ParameterDefinition(name="vehicle new registration", dim_letters=("t", "r")),
    ParameterDefinition(name="vehicle stock", dim_letters=("r",)),
    ParameterDefinition(name="eol recovery rate", dim_letters=("m", "w")),
]

flow_definitions = [
    FlowDefinition(from_process_name="sysenv", to_process_name="market", dim_letters=("t", "r")),
    FlowDefinition(from_process_name="market", to_process_name="use", dim_letters=("t", "m", "r")),
    FlowDefinition(from_process_name="use", to_process_name="waste", dim_letters=("t", "m", "r")),
    FlowDefinition(from_process_name="waste", to_process_name="scrap", dim_letters=("t", "w", "m")),
    FlowDefinition(from_process_name="waste", to_process_name="sysenv", dim_letters=("t", "m")),
    FlowDefinition(
        from_process_name="scrap", to_process_name="sysenv", dim_letters=("t", "w", "m")
    ),
]

stock_definitions = [
    StockDefinition(
        name="in use",
        process="use",
        dim_letters=("t", "r"),
        subclass=InflowDrivenDSM,
        lifetime_model_class=NormalLifetime,
        time_letter="t",
    )
]

mfa_definition = MFADefinition(
    dimensions=dimension_definitions,
    processes=process_names,
    flows=flow_definitions,
    stocks=stock_definitions,
    parameters=parameter_definitions,
)


# %%
class VehicleMFA(MFASystem):
    """We just need to define the compute method with our system equations,
    as all the other things we need are inherited from the MFASystem class."""

    def compute(self):
        stock_diff = self.compute_stock()
        self.compute_flows()
        return stock_diff

    def compute_stock(self):
        self.flows["sysenv => market"][...] = self.parameters["vehicle new registration"]
        self.stocks["in use"].inflow[...] = self.flows["sysenv => market"]
        self.stocks["in use"].lifetime_model.set_prms(
            mean=self.parameters["vehicle lifetime"],
            std=self.parameters["vehicle lifetime"] * 0.3,
        )
        self.stocks["in use"].compute()
        stock_diff = self.get_new_array(dim_letters=("r"))
        stock_diff[...] = (
            1000 * self.parameters["vehicle stock"] - self.stocks["in use"].stock[{"t": 2015}]
        )
        return (stock_diff * 1e-6).to_df()  # in millions

    def compute_flows(self):
        self.flows["market => use"][...] = (
            self.parameters["vehicle material content"]
            * self.parameters["vehicle new registration"]
            * 1e-9
        )
        self.flows["use => waste"][...] = (
            self.parameters["vehicle material content"] * self.stocks["in use"].outflow * 1e-9
        )
        self.flows["waste => scrap"][...] = (
            self.parameters["eol recovery rate"] * self.flows["use => waste"]
        )
        self.flows["scrap => sysenv"][...] = self.flows["waste => scrap"]
        self.flows["waste => sysenv"][...] = (
            self.flows["use => waste"] - self.flows["waste => scrap"]
        )


# %% [markdown]
# ## 3. Define our data reader
# Now that we have defined the MFA system and know what data we need, we can load the data.
# To do the data loading, we define a DataReader class. Such a class can be reused with different datasets of the same format by passing attributes, e.g. the directory where the data is stored, in the init function. In this example, we will also use this data reader in a following step.


# %%
class CustomDataReader(DataReader):
    """The methods `read_dimensions` and `read_parameters` are already defined in the parent
    DataReader class, and loop over the methods `read_dimension` and `read_parameter_values`
    that we specify for our usecase here.
    """

    def __init__(self, data_directory, years):
        self.data_directory = data_directory
        self.years = years

    def read_dimension(self, dimension_definition: DimensionDefinition) -> Dimension:
        if (dim_name := dimension_definition.name) == "region":
            data = pd.read_excel(
                join(self.data_directory, "example5_vehicle_lifetime.xlsx"), "Data"
            )
            other_data = pd.read_excel(
                join(self.data_directory, "example5_vehicle_stock.xlsx"), "Data"
            )
            data = (set(data[dim_name].unique())).intersection(set(other_data[dim_name].unique()))
            data = list(data)
            data.sort()
        elif (dim_name := dimension_definition.name) in ["waste", "material"]:
            data = pd.read_excel(
                join(self.data_directory, "example5_eol_recovery_rate.xlsx"), "Data"
            )
            data.columns = [x.lower() for x in data.columns]
            data = list(data[dim_name].unique())
            data.sort()
        elif dimension_definition.name == "time":
            data = self.years
        return Dimension(
            name=dimension_definition.name,
            letter=dimension_definition.letter,
            dtype=dimension_definition.dtype,
            items=data,
        )

    def read_parameter_values(self, parameter_name: str, dims: DimensionSet) -> Parameter:
        data = pd.read_excel(
            join(self.data_directory, f"example5_{parameter_name.replace(' ', '_')}.xlsx"), "Data"
        )
        data = data.fillna(0)
        if "r" in dims:  # remove unwanted regions
            data = data[data["region"].isin(dims["r"].items)]

        if parameter_name == "vehicle new registration":
            # add rows with missing years
            for year in self.years:
                if year not in data.columns:
                    data[year] = 0
            # remove unncessary columns
            data = data[["region"] + self.years]
        else:
            data.columns = [x.lower() for x in data.columns]
            # remove unncessary columns
            data = data[[dim.name for dim in dims] + ["value"]]
        if parameter_name == "eol recovery rate":
            # add rows with missing waste /material combinations
            waste_material_combinations = [
                (waste, material) for waste in dims["w"].items for material in dims["m"].items
            ]
            data = data.set_index(["waste", "material"])
            data = data.reindex(waste_material_combinations).reset_index()
            data = data.fillna(0)
        return Parameter.from_df(dims=dims, df=data)


# %% [markdown]
# ## 4. Put everything together
# We make an instance of our `CustomDataReader`, read in the data and use it to create an instance of our `VehicleMFA` class. Then we can run the calculations, and check what our estimate of vehicle stocks looks like compared to the data for 2015 in the `vehicle_stock.xlsx` file.

# %%
data_reader = CustomDataReader(data_directory="input_data", years=list(range(2012, 2018)))

vehicle_mfa = VehicleMFA.from_data_reader(
    definition=mfa_definition,
    data_reader=data_reader,
)
stock_diff = vehicle_mfa.compute_stock()  # compute 1st stock estimate and difference to data

# %%
stock_diff

# %% [markdown]
# ## 5. Improve the model
#
# We can see that the results from our inflow-driven dynamic stock model differ significantly from the data about vehicle stocks. One reason for this is the missing inflows for years before our model starts. In a next step, we extend the years back to 1990 and copy the inflow data (given by the `new_vehicle_registration` parameter) by region for each of these earlier years; assuming that the inflow stayed constant between 1990 and 2004.
#
# In addition to this and to answer our research questions, we will extend the timeseries out to 2050, with no more inflow after 2017.
#
# To make these changes in the model dimensions and parameters, we use the `CustomDataReader` class defined above, extending the time dimension. This will also extend the data for the `vehicle_new_registration` parameter, which is our only time-dependent parameter.
#
#

# %%
data_reader_2 = CustomDataReader(data_directory="input_data", years=list(range(1990, 2051)))

vehicle_mfa_2 = VehicleMFA.from_data_reader(
    definition=mfa_definition,
    data_reader=data_reader_2,
)
stock_diff_2 = (
    vehicle_mfa_2.compute()
)  # compute 2nd stock estimate and difference to data, as well as the MFA flows

# %%
stock_diff_2[stock_diff_2 <= stock_diff]

# %% [markdown]
# The stock values in 2015 have improved as compared to the data, but are still quite far away from the dataset.

# %% [markdown]
# ## 6. Answer research questions
# Approximately how big was the material stock embodied in the global passenger vehicle fleet in 2017?
# And when will this material become available for recycling?

# %%
stock_by_material_type = (
    vehicle_mfa_2.stocks["in use"].stock
    * vehicle_mfa_2.parameters["vehicle material content"]
    * 1e-9
)
global_stock_by_material_type = stock_by_material_type.sum_over(sum_over_dims=("r"))
global_stock_by_material_type_in_2017 = global_stock_by_material_type[{"t": 2017}]

stock_df = global_stock_by_material_type_in_2017.to_df(index=False)
fig = px.bar(stock_df, x="material", y="value")
fig.show(renderer="notebook")

# %%
np.nan_to_num(vehicle_mfa_2.flows["scrap => sysenv"].values, copy=False)
scrap_outflow = vehicle_mfa_2.flows["scrap => sysenv"].sum_over(sum_over_dims=("m"))
outflow_df = scrap_outflow.to_df(dim_to_columns="waste")
outflow_df = outflow_df[outflow_df.index > 2017]
fig = px.line(outflow_df, title="Scrap outflow")
fig.show(renderer="notebook")
fig.update_yaxes(type="log")
fig.show(renderer="notebook")

# %%
