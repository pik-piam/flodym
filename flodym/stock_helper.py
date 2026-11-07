"""Home to helper functions for the `Stock` class."""

from typing import Literal, overload

from ._attribute_dict import AttributeDict
from .dimensions import Dimension, DimensionSet
from .flodym_array_helper import flodym_array_stack
from .mfa_definition import StockDefinition
from .processes import Process
from .stocks import Stock


def stock_stack(stocks: list[Stock], dimension: Dimension) -> Stock:
    """Combine `Stock` objects into one stock of the class of the first one, by stacking their
    stock, inflow and outflow arrays along a new dimension.

    For example, we could have one stock for each material of interest, and
    with this function we can combine them to a stock object that contains
    information about all the materials.

    Args:
        stocks: Stocks to combine. Name and process are taken from the first one.
        dimension: The new dimension to stack along, see :py:func:`flodym.flodym_array_stack`.

    Returns:
        The combined stock.
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


@overload
def make_empty_stocks(
    stock_definitions: list[StockDefinition],
    processes: dict[str, Process],
    dims: DimensionSet,
    as_attribute_dict: Literal[False] = ...,
) -> dict[str, Stock]: ...
@overload
def make_empty_stocks(
    stock_definitions: list[StockDefinition],
    processes: dict[str, Process],
    dims: DimensionSet,
    as_attribute_dict: Literal[True],
) -> AttributeDict[Stock]: ...
def make_empty_stocks(
    stock_definitions: list[StockDefinition],
    processes: dict[str, Process],
    dims: DimensionSet,
    as_attribute_dict: bool = False,
) -> dict[str, Stock] | AttributeDict[Stock]:
    """Initialise empty Stock objects for each of the stocks listed in stock definitions.

    The parameter names of prescribed stock arrays (see :py:class:`flodym.StockDefinition`) are
    stored in the arrays; the parameters are linked by
    :py:meth:`flodym.MFASystem.link_parameters`.

    Args:
        stock_definitions: List of stock definitions.
        processes: Dictionary of processes, with process names as keys.
        dims: DimensionSet object containing all dimensions.
        as_attribute_dict: If True, return an :py:class:`flodym.AttributeDict`, whose entries
            can also be accessed as attributes. Default is False, returning a plain dict.

    Returns:
        Dictionary of stocks, with stock names as keys.

    Raises:
        KeyError: If the process of a stock definition is not found in processes.
        ValueError: If a prescribed array does not exist in the created stock.
    """
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

        init_args = {
            "dims": dim_subset,
            "time_letter": stock_definition.time_letter,
            "name": stock_definition.name,
            "process": process,
        }
        if stock_definition.cohort_letter is not None:
            init_args["cohort_dim"] = dims[stock_definition.cohort_letter]
        if stock_definition.lifetime_model_class is not None:
            lifetime_model = stock_definition.lifetime_model_class(
                dims=dim_subset, time_letter=stock_definition.time_letter
            )
            init_args["lifetime_model"] = lifetime_model

        stock = stock_definition.subclass(**init_args)
        for array_name, parameter_name in stock_definition.prescribed_arrays.items():
            array = getattr(stock, array_name, None)
            if array is None:
                raise ValueError(
                    f"Stock '{stock.name}' has no array '{array_name}', which is prescribed in its "
                    "definition. inflow_by_cohort requires a dynamic stock model with a cohort "
                    "dimension."
                )
            array._set_prescribed(parameter_name)
        empty_stocks[stock.name] = stock
    if as_attribute_dict:
        return AttributeDict(empty_stocks)
    return empty_stocks
