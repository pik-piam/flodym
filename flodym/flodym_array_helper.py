"""Home to helper functions for working with `FlodymArray`s."""

import numpy as np

from .dimensions import Dimension
from .flodym_arrays import FlodymArray


def flodym_array_stack(
    flodym_arrays: list[FlodymArray], dimension: Dimension, inplace: bool = False
) -> FlodymArray:
    """Stack a list of FlodymArray objects using a new dimension.

    Like numpy.stack, but for `FlodymArray` objects. New dimension is added at the end.

    Args:
        flodym_arrays: List of FlodymArray objects to stack. Must have the same dimensions and
            shape.
        dimension: The new dimension to stack along. Must not be present in the FlodymArrays to
            stack.
        inplace: If True, modify the first FlodymArray in place. If False, return a new
            FlodymArray object.

    Returns:
        The stacked array, with ``dimension`` appended to the dimensions of the inputs.

    Raises:
        ValueError: If the number of arrays does not match the length of ``dimension``, if the
            arrays have different dimensions, or if ``dimension`` is already present in them.
    """
    if len(flodym_arrays) != dimension.len:
        raise ValueError(
            f"Length of flodym_arrays ({len(flodym_arrays)}) must match length of dimension ({dimension.len})"
        )
    for flodym_array in flodym_arrays:
        if flodym_array.dims != flodym_arrays[0].dims:
            raise ValueError("All FlodymArrays to stack must have the same dimensions")
    if dimension.letter in flodym_array.dims:
        raise ValueError(f"Dimension {dimension.letter} already present in FlodymArrays to stack")
    flodym_array0 = flodym_arrays[0]
    extended_dimensions = flodym_array0.dims.append(dimension)
    # no cast_to: values are overwritten below, and casting extensive arrays would be reported
    values = np.zeros(extended_dimensions.shape, dtype=flodym_array0.values.dtype)
    extended = FlodymArray(dims=extended_dimensions, values=values, name=flodym_array0.name)
    extended._set_degree(flodym_array0._degree)
    for item, flodym_array in zip(dimension.items, flodym_arrays):
        extended[{dimension.letter: item}] = flodym_array
    if inplace:
        flodym_array0.dims = extended.dims
        flodym_array0.values = extended.values
        return flodym_array0
    return extended
