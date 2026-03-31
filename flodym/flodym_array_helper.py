"""Home to helper functions for working with `FlodymArray`s."""

from .flodym_arrays import FlodymArray
from .dimensions import Dimension


def flodym_array_stack(flodym_arrays: list[FlodymArray], dimension: Dimension, inplace: bool = False) -> FlodymArray:
    """Stack a list of FlodymArray objects using a new dimension.
    Like numpy.stack, but for `FlodymArray`s. New dimension is added at the end.

    Args:
        flodym_arrays (list[FlodymArray]): List of FlodymArray objects to stack. Must have the same dimensions and shape.
        dimension (Dimension): The new dimension to stack along. Must not be present in the FlodymArrays to stack.
        inplace (bool, optional): If True, modify the first FlodymArray in place. If False, return a new FlodymArray object.
    """
    if len(flodym_arrays) != dimension.len:
        raise ValueError(f"Length of flodym_arrays ({len(flodym_arrays)}) must match length of dimension ({dimension.len})")
    for flodym_array in flodym_arrays:
        if flodym_array.dims != flodym_arrays[0].dims:
            raise ValueError("All FlodymArrays to stack must have the same dimensions")
    if dimension.letter in flodym_array.dims:
        raise ValueError(f"Dimension {dimension.letter} already present in FlodymArrays to stack")
    flodym_array0 = flodym_arrays[0]
    extended_dimensions = flodym_array0.dims.append(dimension)
    extended = flodym_array0.cast_to(extended_dimensions, inplace=inplace)
    for item, flodym_array in zip(dimension.items, flodym_arrays):
        extended[{dimension.letter: item}] = flodym_array
    return extended
