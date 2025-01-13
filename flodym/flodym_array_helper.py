"""Home to helper functions for working with `FlodymArray`s."""

from .flodym_arrays import FlodymArray
from .dimensions import Dimension


def named_dim_array_stack(named_dim_arrays: list[FlodymArray], dimension: Dimension) -> FlodymArray:
    """Stack a list of FlodymArray objects using a new dimension.
    Like numpy.stack with axis=-1, but for `FlodymArray`s.
    Method can be applied to `FlodymArray`s, `StockArray`s, `Parameter`s and `Flow`s.
    """
    named_dim_array0 = named_dim_arrays[0]
    extended_dimensions = named_dim_array0.dims.expand_by([dimension])
    extended = FlodymArray(dims=extended_dimensions)
    for item, nda in zip(dimension.items, named_dim_arrays):
        extended[{dimension.letter: item}] = nda
    return extended
