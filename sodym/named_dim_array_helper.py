"""Home to helper functions for working with `NamedDimArray`s."""

from .named_dim_arrays import NamedDimArray
from .dimensions import Dimension


def named_dim_array_stack(
    named_dim_arrays: list[NamedDimArray], dimension: Dimension
) -> NamedDimArray:
    """Stack a list of NamedDimArray objects using a new dimension.
    Like numpy.stack with axis=-1, but for `NamedDimArray`s.
    Method can be applied to `NamedDimArray`s, `StockArray`s, `Parameter`s and `Flow`s.
    """
    named_dim_array0 = named_dim_arrays[0]
    extended_dimensions = named_dim_array0.dims.expand_by([dimension])
    extended = NamedDimArray(dims=extended_dimensions)
    for item, nda in zip(dimension.items, named_dim_arrays):
        extended[{dimension.letter: item}] = nda
    return extended
