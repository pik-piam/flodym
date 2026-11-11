"""Weights for converting `FlodymArray` objects between coarser and finer dimensions.

These are used by :py:meth:`FlodymArray.aggregate_to` and :py:meth:`FlodymArray.disaggregate_to`.
"""

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from .dimensions import Dimension, DimensionSet

if TYPE_CHECKING:
    from .flodym_arrays import FlodymArray


def _group_sums(values: np.ndarray, indices: np.ndarray, n_groups: int, axis: int) -> np.ndarray:
    """Sum values along an axis within groups, given the group index of each entry on the axis."""
    groups = [np.flatnonzero(indices == group) for group in range(n_groups)]
    return np.stack([values.take(group, axis=axis).sum(axis=axis) for group in groups], axis=axis)


def _normalized_weights(
    weights: "FlodymArray | dict[Any, FlodymArray]", fine_dim: Dimension, coarse_dim: Dimension
) -> "FlodymArray":
    """Normalize weights over fine_dim such that they sum to one within each coarse_dim item.

    If the weights within a coarse item sum to zero, all its sub-items get the same weight.

    Args:
        weights: Array containing fine_dim, or dictionary of arrays, see
            :py:meth:`FlodymArray.aggregate_to`.
        fine_dim: The finer dimension.
        coarse_dim: The ancestor of fine_dim within whose items the weights are normalized.

    Returns:
        The normalized weights, with the dimensions of the given weights.

    Raises:
        ValueError: If the weights do not contain fine_dim.
    """
    from .flodym_arrays import FlodymArray

    if isinstance(weights, dict):
        weights = _weights_from_dict(weights, fine_dim)
    if fine_dim.letter not in weights.dims:
        raise ValueError(
            f"Weights '{weights.name}' must have the dimension '{fine_dim.name}' to be weighted."
        )
    weights._check_is_set("weighting")
    axis = weights.dims.index(fine_dim.letter)
    indices = fine_dim._indices_in_ancestor(coarse_dim)
    totals = _group_sums(weights.values, indices, coarse_dim.len, axis).take(indices, axis=axis)
    group_sizes = np.bincount(indices, minlength=coarse_dim.len)[indices]
    group_sizes = group_sizes.reshape([-1 if i == axis else 1 for i in range(weights.dims.ndim)])
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = np.where(totals != 0, weights.values / totals, 1.0 / group_sizes)
    return FlodymArray(dims=weights.dims, values=shares, name="weights", extensive=False)


def _weights_from_dict(weights: dict[Any, "FlodymArray"], split_dim: Dimension) -> "FlodymArray":
    """Combine weights given per split parent item into one array over the split dimension.

    Items which are not split get the weight one.

    Args:
        weights: For each split parent item, an array over the dimension it is split into
            (see :py:attr:`flodym.Dimension.sub_dims`), possibly with further dimensions.
        split_dim: Dimension created by :py:meth:`flodym.Dimension.split`.

    Returns:
        Array over split_dim and all further dimensions of the given weights.

    Raises:
        ValueError: If split_dim was not created by splitting, or if weights are missing for a
            split item or lack the dimension it is split into.
    """
    from .flodym_arrays import FlodymArray

    if split_dim.sub_dims is None or split_dim.parent is None or split_dim.parent_items is None:
        raise ValueError(
            f"Weights can only be given as a dictionary for dimensions created by "
            f"Dimension.split(), not for '{split_dim.name}'."
        )
    split_items = {k: v for k, v in split_dim.sub_dims.items() if k in split_dim.parent.items}
    if set(weights) != set(split_items):
        raise ValueError(
            f"Keys of the weights dictionary must be the split items of '{split_dim.name}' "
            f"{sorted(map(str, split_items))}, but are {sorted(map(str, weights))}."
        )
    extra_dims = DimensionSet.empty()
    for parent_item, item_weights in weights.items():
        sub_dim = split_items[parent_item]
        if sub_dim.letter not in item_weights.dims:
            raise ValueError(
                f"Weights for '{parent_item}' must have the dimension '{sub_dim.name}'."
            )
        extra_dims = extra_dims | item_weights.dims.drop(sub_dim.letter)
    combined = FlodymArray(
        dims=split_dim.as_dimset() + extra_dims,
        values=np.ones(split_dim.as_dimset().shape + extra_dims.shape),
        name="weights",
    )
    positions_in_group: dict[Any, int] = defaultdict(int)
    for item, parent_item in zip(split_dim.items, split_dim.parent_items):
        if parent_item not in weights:
            continue
        sub_dim = split_items[parent_item]
        sub_item = sub_dim.items[positions_in_group[parent_item]]
        positions_in_group[parent_item] += 1
        item_weights = weights[parent_item][{sub_dim.letter: sub_item}]
        combined.values[split_dim.index(item)] = item_weights.cast_values_to(extra_dims)
    return combined
