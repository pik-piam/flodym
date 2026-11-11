"""Slicing of `FlodymArray` objects along their dimensions."""

from collections import defaultdict
from collections.abc import Hashable, Iterable
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from .dimensions import Dimension

if TYPE_CHECKING:
    from .flodym_arrays import FlodymArray


def _is_iterable(arg: object) -> bool:
    """Whether ``arg`` is an iterable of items, i.e. not a string or a :py:class:`Dimension`."""
    return isinstance(arg, Iterable) and not isinstance(arg, (str, Dimension))


class _SubArrayHandler:
    """Handler for subsets of the 'values' numpy array of a FlodymArray object.

    The subsets are created by slicing along one or several dimensions. This class specifies the
    behavior of `foo[definition] = bar` and `foo = bar[definition]`, where `foo` and `bar` are
    FlodymArray objects. This is done via the `__getitem__` and `__setitem__` methods of the
    FlodymArray class.

    It returns either

    - a new FlodymArray object (via the `to_flodym_array()` function), or
    - a pointer to a subset of the values array of the parent FlodymArray object, via the
      `values_pointer` attribute.

    The allowed syntaxes for the definition of the subset are described in the docstring of
    :py:class:`FlodymArray`. Several comma-separated item names appear in `__getitem__` and
    `__setitem__` as a tuple.

    Note that does not inherit from FlodymArray, so it is not a FlodymArray object itself.
    However, one can use it to create a FlodymArray object with the `to_flodym_array()` method.

    A split item of the parent of a dimension created by :py:meth:`flodym.Dimension.split` is
    resolved to its sub-items, and the dimension is replaced by the one the item is split into.
    Other items of ancestors are resolved to the items belonging to them only if
    ``allow_ancestor_items`` is True, i.e. when setting values, as no dimension is created
    implicitly for the result of slicing.
    """

    def __init__(
        self,
        flodym_array: "FlodymArray",
        definition: dict[str, Any] | Hashable,
        allow_ancestor_items: bool = False,
    ) -> None:
        self.flodym_array = flodym_array
        self._get_def_dict(definition)
        self._sub_dims_out: dict[str, Dimension] = {}
        self._resolve_ancestor_items(allow_ancestor_items)
        self.invalid_flodym_array = any(
            _is_iterable(v) for k, v in self.def_dict.items() if k not in self._sub_dims_out
        )
        self._init_dims_out()
        self._init_ids()

    @property
    def ids(self) -> tuple[slice | int | list[int] | np.ndarray, ...]:
        """Indices used for slicing the values array."""
        return tuple(self._ids_all_dims)

    @property
    def values_pointer(self) -> np.ndarray:
        """Pointer to the subset of the values array of the parent FlodymArray object."""
        return self.flodym_array.values[self.ids]

    @property
    def dim_letters(self) -> tuple[str, ...]:
        """Updated dimension letters.

        Sliced dimensions with only one item along that direction are removed.
        """
        return self.dims_out.letters

    def to_flodym_array(self) -> "FlodymArray":
        """Return a FlodymArray object that is a slice of the original FlodymArray object.

        Attention: This creates a new FlodymArray object, which is not linked to the original one.
        """
        if self.invalid_flodym_array:
            raise ValueError(
                "Cannot convert to FlodymArray if there are dimension slices with several items."
                "Use a new dimension object with the subset as values instead"
            )

        from .flodym_arrays import FlodymArray

        return FlodymArray(
            dims=self.dims_out, values=self.values_pointer, name=self.flodym_array.name
        )._with_degree(self.flodym_array._degree)

    def _get_def_dict(self, definition: dict[str, Any] | Hashable) -> None:
        self.def_dict: dict[str, Any]
        if isinstance(definition, type(Ellipsis)):
            self.def_dict = {}
        elif isinstance(definition, dict):
            self.def_dict = definition
        elif isinstance(definition, tuple):
            self.def_dict = self._to_dict_tuple(definition)
        else:
            self.def_dict = {self._get_key_single_item(definition): definition}

    def _get_key_single_item(self, item: Hashable) -> str:
        if isinstance(item, slice):
            raise TypeError(
                "Numpy indexing of FlodymArrays is not supported. Details are given in the FlodymArray class "
                "docstring."
            )
        key = None
        for d in self.flodym_array.dims:
            if item in d.items or d._items_below(item) is not None:
                if key is not None:
                    raise ValueError(
                        f"Ambiguous slicing: Item '{item}' is found in multiple dimensions. Please specify the "
                        "dimension by using a slicing dict instead."
                    )
                key = d.letter
        if key is None:
            raise ValueError(f"Slicing item '{item}' not found in any dimension.")
        return key

    def _resolve_ancestor_items(self, allowed: bool) -> None:
        """Replace items of ancestors in the slicing definition by the items belonging to them.

        A single split item of the parent is replaced by its sub-items, and the dimension it is
        split into is stored to replace the sliced dimension.

        Raises:
            ValueError: If an ancestor item is found, but ancestor items are not allowed.
        """
        self.def_dict = dict(self.def_dict)
        for letter, value in list(self.def_dict.items()):
            if isinstance(value, Dimension):
                continue
            dim = self.flodym_array.dims[letter]
            if not _is_iterable(value) and value not in dim.items:
                sub_dim = (dim.sub_dims or {}).get(value)
                if sub_dim is not None and sub_dim.letter not in self.flodym_array.dims:
                    self._sub_dims_out[letter] = sub_dim
                    self.def_dict[letter] = dim._items_below(value)
                    continue
            values = value if _is_iterable(value) else [value]
            resolved: list = []
            found_ancestor_item = False
            for item in values:
                items_below = None if item in dim.items else dim._items_below(item)
                if items_below is None:
                    resolved.append(item)
                    continue
                if not allowed:
                    raise ValueError(
                        f"'{item}' is an item of an ancestor of dimension '{dim.name}'. Except for "
                        "single split items of the parent, such items can only be used to set "
                        "values. Use aggregate_to() or the items of the dimension instead."
                    )
                found_ancestor_item = True
                resolved.extend(items_below)
            if found_ancestor_item:
                self.def_dict[letter] = resolved

    def _to_dict_tuple(self, slice_def: tuple[Hashable, ...]) -> dict[str, Any]:
        dict_out: defaultdict[str, list[Hashable]] = defaultdict(list)
        for item in slice_def:
            key = self._get_key_single_item(item)
            dict_out[key].append(item)
        # if there is only one item along a dimension, convert list to single item
        return {k: v if len(v) > 1 else v[0] for k, v in dict_out.items()}

    def _init_dims_out(self) -> None:
        self.dims_out = deepcopy(self.flodym_array.dims)
        for letter, value in self.def_dict.items():
            if isinstance(value, Dimension):
                self.dims_out.replace(letter, value, inplace=True)
            elif not _is_iterable(value):
                self.dims_out.drop(letter, inplace=True)
        for letter, sub_dim in self._sub_dims_out.items():
            self.dims_out.replace(letter, sub_dim, inplace=True)

    def _init_ids(self) -> None:
        """Initialize the internal list of indices used for slicing the values array.

        - Init the internal list of index slices to slice(None) (i.e. no slicing, keep all items
          along that dimension)
        - For each dimension that is sliced, get the corresponding item index (or list of several
          indexes along one dimension) and replace the slice(None) with it.
        - convert lists of indexes to meshgrid arrays, if there are several dimensions with lists
        """
        self._ids_all_dims: list[slice | int | list[int] | np.ndarray] = [
            slice(None) for _ in self.flodym_array.dims.letters
        ]
        for dim_letter, item_or_items in self.def_dict.items():
            self._set_ids_single_dim(dim_letter, item_or_items)
        self._convert_lists_to_meshgrid()

    def _convert_lists_to_meshgrid(self) -> None:
        """Convert index lists along several dimensions to broadcastable numpy arrays.

        If there are several dimensions with lists as indexes, numpy will try to broadcast the
        lists together.
        To avoid this, the lists are converted to numpy arrays which can be broadcast together:
        Size-one dimensions are prepended and appended to their shape for each of the other
        dimensions of the resulting sub-array. Example:

        original array shape: (3,4,5,6)
        _id_list: [[0,2], slice(None), [1,3,4], 2]
        resulting sub-array shape: (2,4,3)

        conversion:
        - [0,2] -> np.array with shape (2,1,1) and entries 0 and 2
        - slice(None) -> np.array with shape (1,4,1)
        - [1,3,4] -> np.array with shape (1,1,3) and entries 1, 3, 4
        - 2 -> no conversion
        """
        requires_conversion = sum(isinstance(ids, list) for ids in self._ids_all_dims) > 1
        if not requires_conversion:
            return
        # convert slices to lists of all ids to include them in the meshgrid
        for i_axis, ids in enumerate(self._ids_all_dims):
            if isinstance(ids, slice):
                dim_len = self.flodym_array.dims[i_axis].len
                self._ids_all_dims[i_axis] = list(range(dim_len))
        axes_with_id_list = [i for i, ids in enumerate(self._ids_all_dims) if isinstance(ids, list)]
        id_lists = [ids for ids in self._ids_all_dims if isinstance(ids, list)]
        mesh_of_ids = np.ix_(*id_lists)
        for i_axis, ids_mesh in zip(axes_with_id_list, mesh_of_ids):
            self._ids_all_dims[i_axis] = ids_mesh

    def _set_ids_single_dim(self, dim_letter: str, item_or_items: Any) -> None:
        """Set the item IDs along dimension 'dim_letter' for one or several item names.

        Given either a single item name or a list of item names, the corresponding item IDs along
        dimension 'dim_letter' are stored in the internal list of indices.
        """
        items_ids: int | list[int]
        if isinstance(item_or_items, Dimension):
            if item_or_items.is_subset(self.flodym_array.dims[dim_letter]):
                items_ids = [
                    self._get_single_item_id(dim_letter, item) for item in item_or_items.items
                ]
            else:
                raise ValueError(
                    "Dimension item given in array index must be a subset of the dimension it replaces"
                )
        elif _is_iterable(item_or_items):
            items_ids = [self._get_single_item_id(dim_letter, item) for item in item_or_items]
        else:
            items_ids = self._get_single_item_id(dim_letter, item_or_items)  # single item
        self._ids_all_dims[self.flodym_array.dims.index(dim_letter)] = items_ids

    def _get_single_item_id(self, dim_letter: str, item_name: Hashable) -> int:
        return self.flodym_array.dims[dim_letter].items.index(item_name)
