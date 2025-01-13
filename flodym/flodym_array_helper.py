"""Home to helper functions for working with `FlodymArray`s."""

from .flodym_arrays import FlodymArray
from .dimensions import Dimension


def flodym_array_stack(flodym_arrays: list[FlodymArray], dimension: Dimension) -> FlodymArray:
    """Stack a list of FlodymArray objects using a new dimension.
    Like numpy.stack with axis=-1, but for `FlodymArray`s.
    Method can be applied to `FlodymArray`s, `StockArray`s, `Parameter`s and `Flow`s.
    """
    flodym_array0 = flodym_arrays[0]
    extended_dimensions = flodym_array0.dims.expand_by([dimension])
    extended = FlodymArray(dims=extended_dimensions)
    for item, flodym_array in zip(dimension.items, flodym_arrays):
        extended[{dimension.letter: item}] = flodym_array
    return extended
