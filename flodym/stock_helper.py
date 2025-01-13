"""Home to helper functions for the `Stock` class."""

from .processes import Process
from .survival_models import get_survival_model_by_type
from .flodym_array_helper import named_dim_array_stack
from .dimensions import Dimension, DimensionSet
from .mfa_definition import StockDefinition
from .stocks import get_stock_by_type, Stock


def stock_stack(stocks: list[Stock], dimension: Dimension) -> Stock:
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
        subclass = get_stock_by_type(stock_definition.type)
        survival_model = get_survival_model_by_type(stock_definition.survival_model)(
            dims=dim_subset, time_letter=stock_definition.time_letter
        )

        stock = subclass.from_dims(
            dims=dim_subset,
            time_letter=stock_definition.time_letter,
            name=stock_definition.name,
            process=process,
            survival_model=survival_model,
        )

        empty_stocks[stock.name] = stock
    return empty_stocks
