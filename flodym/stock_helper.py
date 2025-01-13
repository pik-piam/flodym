"""Home to helper functions for the `Stock` class."""

import logging
from typing import Optional


from .processes import Process
from .survival_functions import (
    SurvivalModel,
    FixedSurvival,
    FoldedNormalSurvival,
    NormalSurvival,
    LogNormalSurvival,
    WeibullSurvival,
)
from .flodym_arrays import StockArray, Parameter
from .flodym_array_helper import named_dim_array_stack
from .dimensions import Dimension, DimensionSet
from .mfa_definition import StockDefinition
from .stocks import DynamicStockModel, InflowDrivenDSM, StockDrivenDSM, FlowDrivenStock, Stock


def stock_stack(stocks: list[Stock], dimension: Dimension) -> FlowDrivenStock:
    """Make a `FlowDrivenStock` object as a combination of `Stock` objects,
    by combining them on a new dimension.
    For example, we could have one stock for each material of interest, and
    with this function we can combine them to a stock object that contains
    information about all the materials.
    """
    stacked_stock = named_dim_array_stack([stock.stock for stock in stocks], dimension=dimension)
    stacked_inflow = named_dim_array_stack([stock.inflow for stock in stocks], dimension=dimension)
    stacked_outflow = named_dim_array_stack(
        [stock.outflow for stock in stocks], dimension=dimension
    )
    return FlowDrivenStock(
        stock=stacked_stock,
        inflow=stacked_inflow,
        outflow=stacked_outflow,
        name=stocks[0].name,
        process=stocks[0].process,
    )


def make_empty_stock(
    stock_definition: StockDefinition, dims: DimensionSet, process: Process
) -> FlowDrivenStock:
    """Initialises a FlowDrivenStock object with zero values for the stock, inflow and outflow.
    The stock, inflow and outflow have the specified dimensions,
    which ensures that when values are set later, the shape of the data is correct.
    """
    name = stock_definition.name
    stock = StockArray(dims=dims, name=f"{name}_stock")
    inflow = StockArray(dims=dims, name=f"{name}_inflow")
    outflow = StockArray(dims=dims, name=f"{name}_outflow")
    return FlowDrivenStock(name=name, stock=stock, inflow=inflow, outflow=outflow, process=process)


def make_empty_stocks(
    stock_definitions: list[StockDefinition], processes: dict[str, Process], dims: DimensionSet
) -> FlowDrivenStock:
    """Initialise empty FlowDrivenStock objects for each of the stocks listed in stock definitions."""
    empty_stocks = {}
    for stock_definition in stock_definitions:
        dim_subset = dims.get_subset(stock_definition.dim_letters)
        try:
            process = processes[stock_definition.process_name]
        except KeyError:
            raise KeyError(f"Missing process required by stock definition {stock_definition}.")
        stock = make_empty_stock(stock_definition, dims=dim_subset, process=process)
        empty_stocks[stock.name] = stock
    return empty_stocks


def create_dynamic_stock(
    name: str,
    process: Process,
    time_letter: str = "t",
    stock: Optional[StockArray] = None,
    inflow: Optional[StockArray] = None,
    process_name: Optional[str] = None,
    ldf_type: Optional[str] = None,
    lifetime_mean: Optional[Parameter] = None,
    lifetime_std: Optional[Parameter] = None,
) -> DynamicStockModel:
    """Initialise either a StockDrivenDSM or an InflowDrivenDSM,
    depending on whether the user passes stock or inflow.
    The survival function of the dynamic stock model depends on the lifetime distribution function
    (ldf_type), the lifetime mean and the lifetime standard deviation.
    """
    if stock is None and inflow is None:
        raise ValueError("Either stock or inflow must be passed to create a dynamic stock object.")
    dims = stock.dims if stock is not None else inflow.dims
    survival_model = get_survival_model(ldf_type)(
        dims=dims, lifetime_mean=lifetime_mean, lifetime_std=lifetime_std, time_letter=time_letter
    )
    if stock is not None:
        logging.info("Creating StockDrivenDSM object")
        return StockDrivenDSM(
            name=name,
            process=process,
            process_name=process_name,
            stock=stock,
            survival_model=survival_model,
        )
    elif inflow is not None:
        logging.info("Creating InflowDrivenDSM object")
        return InflowDrivenDSM(
            name=name,
            process=process,
            process_name=process_name,
            inflow=inflow,
            survival_model=survival_model,
        )


def get_survival_model(ldf_type: str) -> SurvivalModel:
    """Provides a map whereby the user passes a string and a surival model (class, not instance)
    is returned."""
    survival_map = {
        "Fixed": FixedSurvival,
        "Normal": NormalSurvival,
        "FoldedNormal": FoldedNormalSurvival,
        "LogNormal": LogNormalSurvival,
        "Weibull": WeibullSurvival,
    }
    if ldf_type not in survival_map:
        raise ValueError(f"ldf_type must be one of {list(survival_map.keys())}.")
    return survival_map[ldf_type]
