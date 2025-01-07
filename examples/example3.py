# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Example 3. Dynamic Stock modelling
# *ODYM example by Stefan Pauliuk, adapted for sodym*
#
# sodym defines the class DynamicStockModel for handling the inflow-driven and stock driven model of in-use stocks, see methods section 3 of the [uni-freiburg industrial ecology course](http://www.teaching.industrialecology.uni-freiburg.de/). In this notebook, we show how the dynamic stock model is used in the sodym framework. Other methods of the dynamic_stock_modelling class can be used in a similar way.
#
# The research question is:
# * How large are in-use stocks of steel in selected countries?
# * What is the ration between steel in EoL (end-of-life) products to final steel consumption in selected countries?
# To answer that question the system definition is chosen as in the figure below.
#
# <img src="pictures/SimpleProcess.png" width="354" height="290" alt="Simple MFA system">
#
# Stocks S and outflow O are calculated from apparent final consumption i(t), which is obtained from statistics, cf. DOI 10.1016/j.resconrec.2012.11.008
# The model equations are as follows:
#
# First, we compute the outflow o_c(t,c) of each historic inflow/age-cohort i(c) in year t as
# $o\_c(t,c) = i(c) \cdot sf(t,c)$
# where sf is the survival function of the age cohort, which is 1-cdf, see the [wikipedia page on the survival function](https://en.wikipedia.org/wiki/Survival_function).
# The total outflow o(t) in a given year is then
# $o(t) = \sum_{c\leq t} o\_c(t,c)$
# The mass balance leads to the stock change $dS$:
# $dS(t) = i(t) - o(t)$
# And the stock finally is computed as
# $S(t) = \sum_{t'\leq t} ds(t')$

# %% [markdown]
# ## 1. Load sodym and useful packages

# %%
import os
import numpy as np
import pandas as pd
import plotly.express as px

from sodym import (
    DimensionDefinition,
    ParameterDefinition,
    Dimension,
    DimensionSet,
    Parameter,
    StockArray,
)
from sodym.data_reader import DataReader
from sodym.processes import Process
from sodym.survival_functions import NormalSurvival
from sodym.stocks import InflowDrivenDSM

# %% [markdown]
# ## 2. Define system dimensions and load data
#
# First, we specify the dimensions that are relevant to our system. These will get passed to our data reader class and thereby we can ensure that the data we are reading has the correct shape.
#
# Even though this is only a small system, we will load data from an excel file, as an example for more complex systems with larger datasets. As mentioned above, we define a data reader class to do read the data and put it into the desired python objects. Such a class can be reused with different datasets of the same format by passing attributes, e.g. the file path, in the init function.

# %%
dimension_definitions = [
    DimensionDefinition(letter="t", name="Time", dtype=int),
    DimensionDefinition(letter="r", name="Region", dtype=str),
]

parameter_definitions = [
    ParameterDefinition(
        name="inflow",
        dim_letters=(
            "t",
            "r",
        ),
    ),
    ParameterDefinition(name="tau", dim_letters=("r",)),
    ParameterDefinition(name="sigma", dim_letters=("r",)),
]


# %%
class LittleDataReader(DataReader):
    def __init__(self, country_lifetimes, steel_consumption_file):
        self.country_lifetimes = country_lifetimes
        self.steel_consumption = self.prepare_steel_consumption_data(steel_consumption_file)

    def prepare_steel_consumption_data(self, steel_consumption_file):
        steel_consumption = pd.read_excel(steel_consumption_file)
        steel_consumption = steel_consumption[["CS", "T", "V"]]
        return steel_consumption.rename(columns={"CS": "r", "T": "t", "V": "inflow"})

    def read_dimension(self, dimension_definition: DimensionDefinition) -> Dimension:
        if dimension_definition.letter == "t":
            data = list(self.steel_consumption["t"].unique())
        elif dimension_definition.letter == "r":
            data = list(self.country_lifetimes.keys())
        return Dimension(
            name=dimension_definition.name,
            letter=dimension_definition.letter,
            items=data,
        )

    def read_parameter_values(self, parameter_name: str, dims: DimensionSet) -> Parameter:
        if parameter_name == "tau":
            data = np.array(list(country_lifetimes.values()))
        elif parameter_name == "sigma":
            data = np.array([0.3 * lifetime for lifetime in country_lifetimes.values()])
        elif parameter_name == "inflow":
            multiindex = self.steel_consumption.set_index(["t", "r"])
            data = multiindex.unstack().values[:, :]
        return Parameter(dims=dims, values=data)


country_lifetimes = {
    "Argentina": 45,
    "Brazil": 25,
    "Canada": 35,
    "Denmark": 55,
    "Ethiopia": 70,
    "France": 45,
    "Greece": 70,
    "Hungary": 30,
    "Indonesia": 30,
}
data_reader = LittleDataReader(
    country_lifetimes=country_lifetimes,
    steel_consumption_file=os.path.join("input_data", "example3_steel_consumption.xlsx"),
)
dimensions = data_reader.read_dimensions(dimension_definitions)
parameters = data_reader.read_parameters(parameter_definitions, dimensions)

# %% [markdown]
# ## 3. Perform dynamic stock modelling
#
# In this example, we do not need to build a whole MFA system, since we are only considering one dynamic stock. To make a dynamic stock in sodym, we first need to define a survival model; in this case we assume a normal distribution of lifetimes. Then, we can initiate the dynamic stock model. Here we choose an inflow driven stock model, because we have data that specifies the inflow and from this and the survival model we want to calculate the stock and the outflow.

# %%
normal_survival_model = NormalSurvival(
    dims=dimensions,
    lifetime_mean=parameters["tau"],
    lifetime_std=parameters["sigma"],
)

inflow_stock = StockArray(dims=dimensions, values=parameters["inflow"].values)

dynamic_stock = InflowDrivenDSM(
    name="steel",
    process=Process(name="in use", id=1),
    survival_model=normal_survival_model,
    inflow=inflow_stock,
)
dynamic_stock.compute()

# %% [markdown]
# ## 4. Take a look at the results
# First, we plot the steel stock in the different countries over time.

# %%
stock_df = dynamic_stock.stock.to_df(dim_to_columns="Region")

fig = px.line(stock_df, title="In-use stocks of steel")
fig.show(renderer="notebook")

# %% [markdown]
# We then plot the ratio of outflow over inflow, which is a measure of the stationarity of a stock, and can be interpreted as one indicator for a circular economy.

# %%
inflow = dynamic_stock.inflow
outflow = dynamic_stock.outflow
with np.errstate(divide="ignore"):
    ratio_df = (outflow / inflow).to_df(dim_to_columns="Region")

fig = px.line(ratio_df, title="Ratio outflow:inflow")
fig.show(renderer="notebook")

# %% [markdown]
# We see that for the rich countries France and Canada the share has been steadily growing since WW2. Upheavals such as wars and major economic crises can also be seen, in particular for Hungary.

# %%
