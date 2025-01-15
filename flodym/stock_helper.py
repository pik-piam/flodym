"""Home to helper functions for the `Stock` class."""

from .processes import Process
from .flodym_array_helper import flodym_array_stack
from .dimensions import Dimension, DimensionSet
from .mfa_definition import StockDefinition
from .stocks import Stock


def stock_stack(stocks: list[Stock], dimension: Dimension) -> Stock:
    """Make a `FlowDrivenStock` object as a combination of `Stock` objects,
    by combining them on a new dimension.
    For example, we could have one stock for each material of interest, and
    with this function we can combine them to a stock object that contains
    information about all the materials.
    """
    stacked_stock = flodym_array_stack([stock.stock for stock in stocks], dimension=dimension)
    stacked_inflow = flodym_array_stack([stock.inflow for stock in stocks], dimension=dimension)
    stacked_outflow = flodym_array_stack([stock.outflow for stock in stocks], dimension=dimension)
    return stocks[0].__class__(
        stock=stacked_stock,
        inflow=stacked_inflow,
        outflow=stacked_outflow,
        name=stocks[0].name,
        process=stocks[0].process,
    )


def make_empty_stocks(
    stock_definitions: list[StockDefinition],
    processes: dict[str, Process],
    dims: DimensionSet,
) -> dict[str, Stock]:
    """Initialise empty Stock objects for each of the stocks listed in stock definitions."""
    empty_stocks = {}
    for stock_definition in stock_definitions:
        dim_subset = dims.get_subset(stock_definition.dim_letters)
        if stock_definition.process_name is None:
            process = None
        else:
            try:
                process = processes[stock_definition.process_name]
            except KeyError:
                raise KeyError(f"Process {stock_definition.process_name} not in processes.")

        init_args = dict(
            dims=dim_subset,
            time_letter=stock_definition.time_letter,
            name=stock_definition.name,
            process=process,
        )
        if stock_definition.lifetime_model_class is not None:
            lifetime_model = stock_definition.lifetime_model_class(
                dims=dim_subset, time_letter=stock_definition.time_letter
            )
            init_args["lifetime_model"] = lifetime_model

        stock = stock_definition.subclass(**init_args)
        empty_stocks[stock.name] = stock
    return empty_stocks
