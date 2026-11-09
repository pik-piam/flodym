"""Dimensions and sets of dimensions over which flodym arrays are defined."""

from __future__ import annotations

from collections.abc import Iterator
from copy import copy
from typing import Any, Literal, overload

import numpy as np
import pandas as pd
from pydantic import AliasChoices, Field, model_validator

from ._base_model import FlodymBaseModel
from .mfa_definition import DimensionDefinition


class Dimension(FlodymBaseModel):
    """One of multiple dimensions over which MFA arrays are defined.

    Defined by a name, a letter for shorter addressing, and a list of items.

    **Example**

    >>> import flodym as fd
    >>> regions = fd.Dimension(name="Region", letter="r", items=["Earth", "Moon", "Sun"])

    The list of items can be loaded using a :py:class:`flodym.DataReader` object,
    or set directly, for example if a subset of an existing dimension is formed.

    **Sub-dimensions**

    A dimension can be a finer resolution of another dimension, its ``parent``: Each item is
    assigned to exactly one item of the parent, given in ``parent_items``. Examples are countries
    assigned to world regions, or building types split into sub-types. Several levels are
    possible, as the parent can have a parent itself.

    >>> world_regions = fd.Dimension(name="World region", letter="w", items=["EUR", "USA"])
    >>> countries = fd.Dimension(
    ...     name="Country",
    ...     letter="c",
    ...     items=["DEU", "FRA", "USA"],
    ...     parent=world_regions,
    ...     parent_items=["EUR", "EUR", "USA"],
    ... )

    Items of the parent can be split into the items of other dimensions with :py:meth:`split`.
    Arrays defined over related dimensions are aggregated and disaggregated automatically in
    operations, see :py:meth:`flodym.FlodymArray.aggregate_to` and
    :py:meth:`flodym.FlodymArray.disaggregate_to`.
    """

    name: str = Field(..., min_length=2)
    """The full name of the dimension"""
    letter: str = Field(
        ..., min_length=1, max_length=1, validation_alias=AliasChoices("letter", "dim_letter")
    )
    """A single index letter for shorter addressing of the dimension"""
    items: list
    """A list of items that are resolved along the dimension, e.g. years or regions"""
    dtype: type | None = None
    """If given, a check is performed that all items have this datatype. Recommended for safety,
    especially in ambiguous cases such as calendar years (int or str)
    """
    parent: Dimension | None = None
    """Coarser dimension of which this dimension is a finer resolution."""
    parent_items: list | None = None
    """For each item, the item of the parent it belongs to. Every parent item must occur at least
    once. Can be given without ``parent`` when reading the dimension, before the parent is
    linked by :py:meth:`flodym.DataReader.read_dimensions`."""
    sub_dims: dict[Any, Dimension] | None = None
    """Only for dimensions created by :py:meth:`split`: The dimension each split parent item is
    split into. For dimensions created by :py:meth:`combine_splits`, also the lower split for
    the items of the further ancestor which are split. Slicing an array by one of the keys returns
    an array over the corresponding dimension."""

    @model_validator(mode="after")
    def _items_have_datatype(self) -> Dimension:
        """If a datatype is specified, check that all items have this datatype."""
        if self.dtype is not None and any(not isinstance(i, self.dtype) for i in self.items):
            raise ValueError("All items must have the same datatype as specified in dtype.")
        return self

    @model_validator(mode="after")
    def _check_parent(self) -> Dimension:
        """Check that the parent items are consistent with the items and the parent."""
        if self.parent is None:
            if self.sub_dims is not None:
                raise ValueError("sub_dims can only be given together with a parent.")
            return self
        if self.parent_items is None:
            raise ValueError(f"Dimension '{self.name}' has a parent, but no parent_items.")
        if len(self.parent_items) != len(self.items):
            raise ValueError(
                f"parent_items of dimension '{self.name}' must have the same length as its items."
            )
        if self.letter in [dim.letter for dim in self.ancestors]:
            raise ValueError(
                f"Dimension '{self.name}' must not have the same letter as one of its ancestors."
            )
        unknown_items = set(self.parent_items) - set(self.parent.items)
        if unknown_items:
            raise ValueError(
                f"parent_items of dimension '{self.name}' are not items of the parent "
                f"'{self.parent.name}': {sorted(map(str, unknown_items))}"
            )
        uncovered_items = set(self.parent.items) - set(self.parent_items)
        if uncovered_items:
            raise ValueError(
                f"Items of parent '{self.parent.name}' are not assigned to any item of dimension "
                f"'{self.name}': {sorted(map(str, uncovered_items))}"
            )
        for ancestor_item, sub_dim in (self.sub_dims or {}).items():
            items_below = self._items_below(ancestor_item)
            if items_below is None or len(items_below) != sub_dim.len:
                raise ValueError(
                    f"Dimension '{sub_dim.name}' in sub_dims of '{self.name}' must have one item "
                    f"for each item below '{ancestor_item}'."
                )
        return self

    def __add__(self, other: Dimension | DimensionSet) -> DimensionSet:
        """Addition operator to create a DimensionSet from two Dimensions.

        If ``other`` is a DimensionSet, this Dimension is prepended to it.

        Raises:
            TypeError: If other is neither a Dimension nor a DimensionSet.
        """
        if isinstance(other, Dimension):
            return DimensionSet(dim_list=[self, other])
        elif isinstance(other, DimensionSet):
            return self.as_dimset() + other
        else:
            raise TypeError("Can only add Dimension to Dimension or DimensionSet.")

    def __str__(self) -> str:
        base = f"Dimension '{self.name}' ('{self.letter}'); "
        item_base = f"{self.len} items"
        type_info = f" (type {self.dtype})" if self.dtype is not None else ""
        if self.len <= 3:
            list_str = f": {self.items!s}"
        else:
            list_str = f": ['{self.items[0]}', ..., '{self.items[-1]}']"
        parent_info = f"; parent '{self.parent.letter}'" if self.parent is not None else ""
        return base + item_base + type_info + list_str + parent_info

    @classmethod
    def from_df(cls, df: pd.DataFrame, definition: DimensionDefinition) -> Dimension:
        """Create Dimension object from definition and a pandas DataFrame for the items.

        Args:
            df: A single-column or single-row data frame, which is transformed to the dimension
                items.
            definition: The definition of the dimension.

        Returns:
            The dimension object.
        """
        data = df.to_numpy()
        return cls.from_np(data, definition)

    @classmethod
    def from_np(cls, data: np.ndarray, definition: DimensionDefinition) -> Dimension:
        """Create Dimension object from definition and a numpy array for the items.

        Performs checks on the array and transforms to list of items.
        A leading item equal to the dimension name is treated as header and removed.

        If the definition has a :py:attr:`flodym.DimensionDefinition.parent`, the array must have
        two columns: the items, and the parent item of each item. The parent items are stored in
        :py:attr:`parent_items`; the parent is linked by
        :py:meth:`flodym.DataReader.read_dimensions`.

        Args:
            data: A numpy array where only one dimension exceeds size 1, or a two-column array if
                the definition has a parent.
            definition: The definition object

        Returns:
            The dimension object.

        Raises:
            ValueError: If more than one axis of the array has a size larger than one, or if the
                definition has a parent and the array does not have two columns.
        """
        if definition.parent is not None:
            return cls._from_np_with_parent_items(data, definition)
        if sum([s > 1 for s in data.shape]) > 1:
            raise ValueError(
                f"Dimension data for {definition.name} must have only one row or column."
            )
        if data.ndim != 1:
            data = data.flatten()
        # delete header for items if present
        if data[0] == definition.name:
            data = data[1:]
        data = [definition.dtype(item) for item in data]
        return cls(
            name=definition.name, letter=definition.letter, items=data, dtype=definition.dtype
        )

    @classmethod
    def _from_np_with_parent_items(
        cls, data: np.ndarray, definition: DimensionDefinition
    ) -> Dimension:
        """Create a Dimension with unlinked parent items from a two-column numpy array.

        A leading row whose first entry is the dimension name is treated as header and removed.

        Raises:
            ValueError: If the array does not have two columns.
        """
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(
                f"Dimension data for {definition.name} must have two columns: the items, and the "
                f"items of the parent dimension '{definition.parent}' they belong to."
            )
        if data[0, 0] == definition.name:
            data = data[1:]
        items = [definition.dtype(item) for item in data[:, 0]]
        parent_items = data[:, 1].tolist()
        return cls(
            name=definition.name,
            letter=definition.letter,
            items=items,
            dtype=definition.dtype,
            parent_items=parent_items,
        )

    @classmethod
    def from_mapping(
        cls,
        name: str,
        letter: str,
        parent: Dimension,
        mapping: dict[Any, Any],
        dtype: type | None = None,
    ) -> Dimension:
        """Create a sub-dimension of ``parent`` from a mapping of items to parent items.

        Args:
            name: Name of the new dimension.
            letter: Letter of the new dimension.
            parent: The coarser dimension.
            mapping: The parent item for each item of the new dimension, e.g.
                ``{"DEU": "EUR", "FRA": "EUR", "USA": "USA"}``. The order of the keys is the
                order of the items.
            dtype: Datatype of the items.

        Returns:
            The new dimension.
        """
        return cls(
            name=name,
            letter=letter,
            items=list(mapping.keys()),
            dtype=dtype,
            parent=parent,
            parent_items=list(mapping.values()),
        )

    @classmethod
    def combine_splits(
        cls,
        name: str,
        letter: str,
        upper: Dimension,
        lower: Dimension | list[Dimension],
        separator: str = "|",
    ) -> Dimension:
        """Combine splits into a dimension with two levels of splitting.

        ``upper`` splits items of a dimension into the items of other dimensions, and each
        dimension in ``lower`` splits items of one of these dimensions further (see
        :py:meth:`split`). The result is ``upper``, with the items split further as in ``lower``.
        Its parent is ``upper``, so arrays over it can be aggregated to ``upper`` and its
        ancestors. Slicing an array by an item of the parent of ``upper`` returns an array over
        the dimension in ``lower`` it is split as, or over the dimension it is split into in
        ``upper``, if it is not split further.

        **Example**

        >>> import flodym as fd
        >>> coarse = fd.Dimension(name="Construction type", letter="i", items=["Infr", "Build"])
        >>> medium = fd.Dimension(name="Building type", letter="b", items=["Com", "Res"])
        >>> fine = fd.Dimension(name="Residential type", letter="y", items=["Sin", "Mul"])
        >>> coarse_and_medium = coarse.split(
        ...     name="Level 1", letter="l", splitting={"Build": medium}
        ... )
        >>> medium_and_fine = medium.split(
        ...     name="Building sub-type", letter="s", splitting={"Res": fine}
        ... )
        >>> combined = fd.Dimension.combine_splits(
        ...     name="Level 2", letter="m", upper=coarse_and_medium, lower=medium_and_fine
        ... )
        >>> combined.items
        ['Infr', 'Build|Com', 'Build|Res|Sin', 'Build|Res|Mul']

        Args:
            name: Name of the new dimension.
            letter: Letter of the new dimension.
            upper: Dimension created by :py:meth:`split`.
            lower: Dimension created by :py:meth:`split` from a dimension into which items of
                ``upper`` are split, or a list of such dimensions for different such dimensions.
            separator: String between parent item and sub-item in the new item names.

        Returns:
            The new dimension, with ``upper`` as parent.

        Raises:
            ValueError: If a dimension in ``lower`` is not created by splitting a dimension into
                which items of ``upper`` are split.
        """
        splitting: dict[Any, Dimension] = {}
        lower_sub_dims: dict[Any, Dimension] = {}
        for lower_dim in [lower] if isinstance(lower, Dimension) else lower:
            lower_parent = lower_dim.parent
            split_items = [
                item
                for item, sub_dim in (upper.sub_dims or {}).items()
                if lower_parent is not None and sub_dim.letter == lower_parent.letter
            ]
            if lower_parent is None or lower_dim.sub_dims is None or not split_items:
                raise ValueError(
                    f"Dimension '{lower_dim.name}' must be created by split() from a dimension "
                    f"into which items of '{upper.name}' are split."
                )
            splitting |= {
                item: lower_dim.sub_dims[lower_parent_item]
                for upper_item in split_items
                for item, lower_parent_item in zip(
                    upper._items_below(upper_item) or [], lower_parent.items
                )
                if lower_parent_item in lower_dim.sub_dims
            }
            lower_sub_dims |= dict.fromkeys(split_items, lower_dim)
        combined = upper.split(name=name, letter=letter, splitting=splitting, separator=separator)
        unchanged_sub_dims = {
            item: sub_dim
            for item, sub_dim in (upper.sub_dims or {}).items()
            if combined._items_below(item) == upper._items_below(item)
        }
        sub_dims = {**unchanged_sub_dims, **lower_sub_dims, **splitting}
        return cls.model_validate({**dict(combined), "sub_dims": sub_dims})

    @property
    def len(self) -> int:
        """The number of items in the dimension."""
        return len(self.items)

    @property
    def ancestors(self) -> list[Dimension]:
        """The parent, the parent of the parent, and so on."""
        ancestors = []
        dim = self.parent
        while dim is not None:
            ancestors.append(dim)
            dim = dim.parent
        return ancestors

    def split(
        self,
        name: str,
        letter: str,
        splitting: dict[Any, Dimension],
        separator: str = "|",
    ) -> Dimension:
        """Create a sub-dimension by splitting some items into the items of other dimensions.

        The name of a split item is the parent item and the sub-item, joined by ``separator``.
        Items which are not split are kept.

        Arrays over the new dimension can be sliced by a split item, which returns an array over
        the dimension it is split into (see :py:class:`flodym.FlodymArray`). To split several
        levels, see :py:meth:`combine_splits`.

        **Example**

        >>> import flodym as fd
        >>> building_types = fd.Dimension(name="Building type", letter="b", items=["Com", "Res"])
        >>> residential_types = fd.Dimension(
        ...     name="Residential type", letter="r", items=["Sin", "Mul"]
        ... )
        >>> sub_types = building_types.split(
        ...     name="Building sub-type", letter="s", splitting={"Res": residential_types}
        ... )
        >>> sub_types.items
        ['Com', 'Res|Sin', 'Res|Mul']

        Args:
            name: Name of the new dimension.
            letter: Letter of the new dimension.
            splitting: Dictionary of the items to split, and the dimension each is split into.
            separator: String between parent item and sub-item in the new item names. It may
                occur in the sub-items, e.g. if they result from splitting themselves.

        Returns:
            The new dimension, with this dimension as parent.

        Raises:
            ValueError: If a key of ``splitting`` is not an item, or if the new item names are not
                unique.
        """
        unknown_items = set(splitting) - set(self.items)
        if unknown_items:
            raise ValueError(
                f"Items to split are not items of dimension '{self.name}': "
                f"{sorted(map(str, unknown_items))}"
            )
        items: list = []
        parent_items: list = []
        for item in self.items:
            if item not in splitting:
                items.append(item)
                parent_items.append(item)
                continue
            for sub_item in splitting[item].items:
                items.append(f"{item}{separator}{sub_item}")
                parent_items.append(item)
        if len(set(items)) != len(items):
            raise ValueError(f"Items of the split dimension '{name}' are not unique: {items}")
        return Dimension(
            name=name,
            letter=letter,
            items=items,
            dtype=str if self.dtype is str else None,
            parent=self,
            parent_items=parent_items,
            sub_dims=dict(splitting),
        )

    def is_descendant_of(self, other: Dimension) -> bool:
        """Check whether ``other`` is the parent, the parent of the parent, etc.

        Dimensions are compared by letter.
        """
        return other.letter in [dim.letter for dim in self.ancestors]

    def common_ancestor(self, other: Dimension) -> Dimension | None:
        """Return the finest dimension that both this and ``other`` are or descend from.

        Dimensions are compared by letter. If one of the two dimensions is an ancestor of the
        other, it is returned itself.

        Returns:
            The common ancestor, or None if the dimensions are not related.
        """
        if other.letter == self.letter or self.is_descendant_of(other):
            return other
        if other.is_descendant_of(self):
            return self
        other_lineage_letters = [dim.letter for dim in other.ancestors]
        for dim in self.ancestors:
            if dim.letter in other_lineage_letters:
                return dim
        return None

    def ancestor_items(self, ancestor: Dimension) -> list:
        """Return, for each item, the item of an ancestor it belongs to.

        Args:
            ancestor: The ancestor dimension; it is identified by its letter.

        Returns:
            The ancestor items, in the order of the items of this dimension.

        Raises:
            ValueError: If ``ancestor`` is not an ancestor of this dimension, or if it is not
                linked yet.
        """
        if self.parent is None or self.parent_items is None:
            raise ValueError(f"Dimension '{self.name}' has no linked parent.")
        current_items = self.parent_items
        dim = self.parent
        while dim.letter != ancestor.letter:
            if dim.parent is None or dim.parent_items is None:
                raise ValueError(
                    f"Dimension '{ancestor.name}' is not an ancestor of dimension '{self.name}'."
                )
            to_parent = dict(zip(dim.items, dim.parent_items))
            current_items = [to_parent[item] for item in current_items]
            dim = dim.parent
        return current_items

    def index(self, item: Any) -> int:
        """Return the index of an item in the dimension.

        Raises:
            ValueError: If the item is not in the dimension.
        """
        return self.items.index(item)

    def is_subset(self, other: Dimension) -> bool:
        """Check if the items of this dimension are a subset of the items of another dimension."""
        return set(self.items).issubset(other.items)

    def is_superset(self, other: Dimension) -> bool:
        """Check if the items of this dimension are a superset of the items of another dimension."""
        return set(self.items).issuperset(other.items)

    def as_dimset(self) -> DimensionSet:
        """Convert the Dimension to a DimensionSet containing only this Dimension."""
        return DimensionSet(dim_list=[self])

    def _items_below(self, ancestor_item: Any) -> list | None:
        """Return the items belonging to an item of the nearest ancestor which has this item.

        Returns:
            The items, or None if no ancestor has the item.
        """
        for ancestor in self.ancestors:
            if ancestor_item in ancestor.items:
                ancestor_items = self.ancestor_items(ancestor)
                return [
                    item
                    for item, other in zip(self.items, ancestor_items)
                    if other == ancestor_item
                ]
        return None

    def _indices_in_ancestor(self, ancestor: Dimension) -> np.ndarray:
        """For each item, the index of the ancestor item it belongs to, in ``ancestor.items``.

        Raises:
            ValueError: If items of the ancestor in the parent chain are missing in ``ancestor``.
        """
        ancestor_items = self.ancestor_items(ancestor)
        index_of = {item: index for index, item in enumerate(ancestor.items)}
        missing_items = set(ancestor_items) - set(index_of)
        if missing_items:
            raise ValueError(
                f"Items {sorted(map(str, missing_items))} of the ancestor of '{self.name}' are "
                f"missing in dimension '{ancestor.name}'."
            )
        return np.array([index_of[item] for item in ancestor_items], dtype=int)


class DimensionSet(FlodymBaseModel):
    """A set of Dimension objects which MFA arrays are defined over.

    **Example**

        >>> import flodym as fd
        >>> regions = fd.Dimension(name="Region", letter="r", items=["Earth", "Moon", "Sun"])
        >>> time = fd.Dimension(name="Time", letter="t", items=[1990, 2000, 2010, 2020, 2030])
        >>> dimensions = fd.DimensionSet(dim_list=[regions, time])

    It is expected that DimensionSet instances are created via the :py:class:`flodym.DataReader`.

        >>> import flodym as fd
        >>> class MyDataReader(fd.DataReader):
        ...     def read_dimension(
        ...         self, dimension_definition: fd.DimensionDefinition
        ...     ) -> fd.Dimension:
        ...         if dimension_definition.letter == "t":
        ...             return fd.Dimension(
        ...                 name="Time", letter="t", items=[1990, 2000, 2010, 2020, 2030]
        ...             )
        ...         elif dimension_definition.letter == "r":
        ...             return fd.Dimension(
        ...                 name="Region", letter="r", items=["Earth", "Moon", "Sun"]
        ...             )
        ...         raise ValueError("No data available for desired dimension")
        ...
        ...     def read_parameter_values(
        ...         self, parameter_name: str, dims: fd.DimensionSet
        ...     ) -> fd.Parameter:
        ...         raise NotImplementedError("This reader only provides dimensions")
        >>> data_reader = MyDataReader()
        >>> time_definition = fd.DimensionDefinition(name="Time", letter="t", dtype=int)
        >>> region_definition = fd.DimensionDefinition(name="Region", letter="r", dtype=str)
        >>> definitions = [time_definition, region_definition]
        >>> dimensions = data_reader.read_dimensions(dimension_definitions=definitions)

    """

    dim_list: list[Dimension]
    """A list of Dimension objects defining the set"""

    @model_validator(mode="after")
    def _no_repeated_dimensions(self) -> DimensionSet:
        """Check that all dimensions have unique letters."""
        letters = self.letters
        if len(letters) != len(set(letters)):
            raise ValueError("Dimensions must have unique letters in DimensionSet.")
        return self

    @model_validator(mode="after")
    def _copy_dim_list(self) -> DimensionSet:
        """Ensure each DimensionSet has its own copy of the dim_list.

        This avoids inplace operations affecting other DimensionSets.
        """
        self.dim_list = copy(self.dim_list)
        return self

    @overload
    def __getitem__(self, key: tuple[str, ...]) -> DimensionSet: ...
    @overload
    def __getitem__(self, key: str | int) -> Dimension: ...
    def __getitem__(self, key: tuple[str, ...] | str | int) -> DimensionSet | Dimension:
        """Get a dimension by its name, letter or index with the [] operator.

        Args:
            key: The name, letter or index of the dimension to get; If a tuple is passed, a subset
                of the dimensions is returned as a new DimensionSet.

        Returns:
            The selected dimension, or a DimensionSet if a tuple was
            passed.

        Raises:
            TypeError: If the key is not a string, int or tuple.

        Example:
        >>> dimensions["Region"]
        >>> dimensions["r"]
        >>> dimensions[0]
        >>> dimensions["Region", "Time"]
        """
        if isinstance(key, tuple):
            return self.get_subset(key)
        if isinstance(key, str):
            return self._full_mapping[key]
        elif isinstance(key, int):
            return self.dim_list[key]
        else:
            raise TypeError("Key must be string or int")

    def __iter__(self) -> Iterator[Dimension]:
        return iter(self.dim_list)

    def __contains__(self, key: str | Dimension) -> bool:
        """Check if a dimension name or letter is in the set; Dimensions are compared by letter."""
        if isinstance(key, Dimension):
            key = key.letter
        return key in self._full_mapping

    def __len__(self) -> int:
        """Return the number of dimensions in the set."""
        return len(self.dim_list)

    def __bool__(self) -> bool:
        """Return True if the set is not empty."""
        return len(self.dim_list) > 0

    def __str__(self) -> str:
        base = f"DimensionSet ({','.join(self.letters)}) with shape {self.shape}:"
        dim_strs = [
            f"\n  '{dim.letter}': '{dim.name}' with length {dim.len}" for dim in self.dim_list
        ]
        return base + "".join(dim_strs)

    @classmethod
    def empty(cls) -> DimensionSet:
        """Return an empty DimensionSet."""
        return cls(dim_list=[])

    @property
    def ndim(self) -> int:
        """The number of dimensions in the set."""
        return len(self.dim_list)

    @property
    def names(self) -> tuple[str, ...]:
        """A tuple of the names of the dimensions in the set."""
        return tuple([dim.name for dim in self.dim_list])

    @property
    def letters(self) -> tuple[str, ...]:
        """A tuple of the letters of the dimensions in the set."""
        return tuple([dim.letter for dim in self.dim_list])

    @property
    def string(self) -> str:
        """The letters of the dimensions in the set concatenated to a single string."""
        return "".join(self.letters)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the array that would be created with the dimensions in the set."""
        return tuple(self.size(dim) for dim in self.letters)

    @property
    def total_size(self) -> int:
        """Size (total number of elements) of an array created with the dimensions in the set."""
        return int(np.prod(self.shape))

    @property
    def _full_mapping(self) -> dict[str, Dimension]:
        """Contains mappings.

        letter --> dim object and name --> dim object
        """
        return {dim.name: dim for dim in self.dim_list} | {dim.letter: dim for dim in self.dim_list}

    def size(self, key: str) -> int:
        """Get the number of items in a dimension.

        Args:
            key: the name or letter of the dimension to get the size of

        Returns:
            The number of items in the dimension.
        """
        return self._full_mapping[key].len

    def index(self, key: str) -> int:
        """Return the index of a dimension in the set.

        Args:
            key: The name or letter of the dimension to get the index of

        Returns:
            The position of the dimension in the set.
        """
        dim = self._full_mapping[key]
        return self.dim_list.index(dim)

    def copy(self) -> DimensionSet:
        """Return a copy of the DimensionSet."""
        return self.model_copy(update={"dim_list": copy(self.dim_list)})

    def get_subset(self, dims: tuple[str, ...] | None = None) -> DimensionSet:
        """Select a subset of the dimensions, given by their letters or names.

        Selects :py:class:`Dimension` objects from the object attribute dim_list,
        according to the dims passed, which can be either letters or names.
        Returns a copy if dims are not given.

        Args:
            dims: A tuple of dimension letters or names to select. Defaults to None.

        Returns:
            A new DimensionSet with the selected dimensions in the given order.
        """
        subset = self.model_copy()
        if dims is not None:
            subset.dim_list = [self._full_mapping[dim_key] for dim_key in dims]
        return subset

    @overload
    def expand_by(
        self, added_dims: list[Dimension], inplace: Literal[False] = False
    ) -> DimensionSet: ...
    @overload
    def expand_by(self, added_dims: list[Dimension], inplace: Literal[True]) -> None: ...
    @overload
    def expand_by(
        self, added_dims: list[Dimension], inplace: bool = False
    ) -> DimensionSet | None: ...
    def expand_by(self, added_dims: list[Dimension], inplace: bool = False) -> DimensionSet | None:
        """Expands the DimensionSet by adding new dimensions to it.

        Args:
            added_dims: A list of Dimension objects to add
            inplace: If True, the operation is performed in place, otherwise a new DimensionSet is
                returned. Defaults to False.

        Returns:
            None if inplace=True, otherwise a new DimensionSet with the new dimensions added

        Raises:
            ValueError: If one of the added dimensions has a letter already present in the set.
        """
        if not all(dim.letter not in self.letters for dim in added_dims):
            raise ValueError(
                "DimensionSet already contains one or more of the dimensions to be added."
            )
        if inplace:
            self.dim_list.extend(added_dims)
            return
        else:
            return DimensionSet(dim_list=self.dim_list + added_dims)

    extend = expand_by

    @overload
    def append(self, new_dim: Dimension, inplace: Literal[False] = False) -> DimensionSet: ...
    @overload
    def append(self, new_dim: Dimension, inplace: Literal[True]) -> None: ...
    @overload
    def append(self, new_dim: Dimension, inplace: bool = False) -> DimensionSet | None: ...
    def append(self, new_dim: Dimension, inplace: bool = False) -> DimensionSet | None:
        """Add a new dimension to the set at the end.

        Args:
            new_dim: The new dimension to add
            inplace: If True, the operation is performed in place, otherwise a new DimensionSet is
                returned. Defaults to False.

        Returns:
            None if inplace=True, otherwise a new DimensionSet with the new dimension added

        Raises:
            TypeError: If new_dim is not a Dimension.
            ValueError: If the letter of new_dim is already present in the set.
        """
        self._check_additional_dim(new_dim)
        if inplace:
            self.dim_list.append(new_dim)
            return
        else:
            return self + new_dim

    @overload
    def prepend(self, new_dim: Dimension, inplace: Literal[False] = False) -> DimensionSet: ...
    @overload
    def prepend(self, new_dim: Dimension, inplace: Literal[True]) -> None: ...
    @overload
    def prepend(self, new_dim: Dimension, inplace: bool = False) -> DimensionSet | None: ...
    def prepend(self, new_dim: Dimension, inplace: bool = False) -> DimensionSet | None:
        """Add a new dimension to the set at the beginning.

        Args:
            new_dim: The new dimension to add
            inplace: If True, the operation is performed in place, otherwise a new DimensionSet is
                returned. Defaults to False.

        Returns:
            None if inplace=True, otherwise a new DimensionSet with the new dimension added

        Raises:
            TypeError: If new_dim is not a Dimension.
            ValueError: If the letter of new_dim is already present in the set.
        """
        self._check_additional_dim(new_dim)
        if inplace:
            self.dim_list.insert(0, new_dim)
            return
        else:
            return new_dim + self

    @overload
    def insert(
        self, index: int, new_dim: Dimension, inplace: Literal[False] = False
    ) -> DimensionSet: ...
    @overload
    def insert(self, index: int, new_dim: Dimension, inplace: Literal[True]) -> None: ...
    @overload
    def insert(
        self, index: int, new_dim: Dimension, inplace: bool = False
    ) -> DimensionSet | None: ...
    def insert(self, index: int, new_dim: Dimension, inplace: bool = False) -> DimensionSet | None:
        """Insert a new dimension to the set at the given index.

        Args:
            index: The index at which to insert the new dimension
            new_dim: The new dimension to add
            inplace: If True, the operation is performed in place, otherwise a new DimensionSet is
                returned. Defaults to False.

        Returns:
            None if inplace=True, otherwise a new DimensionSet with the new dimension added

        Raises:
            TypeError: If new_dim is not a Dimension.
            ValueError: If the letter of new_dim is already present in the set.
        """
        self._check_additional_dim(new_dim)
        if inplace:
            self.dim_list.insert(index, new_dim)
            return
        else:
            dim_list = copy(self.dim_list)
            dim_list.insert(index, new_dim)
            return DimensionSet(dim_list=dim_list)

    @overload
    def drop(self, key: str, inplace: Literal[False] = False) -> DimensionSet: ...
    @overload
    def drop(self, key: str, inplace: Literal[True]) -> None: ...
    @overload
    def drop(self, key: str, inplace: bool = False) -> DimensionSet | None: ...
    def drop(self, key: str, inplace: bool = False) -> DimensionSet | None:
        """Remove a dimension from the set.

        Args:
            key: The name or letter of the dimension to drop
            inplace: If True, the operation is performed in place, otherwise a new DimensionSet is
                returned. Defaults to False.

        Returns:
            None if inplace=True, otherwise a new DimensionSet with the dimension removed
        """
        dim_to_drop = self._full_mapping[key]
        if inplace:
            self.dim_list.remove(dim_to_drop)
            return
        else:
            dimensions = copy(self.dim_list)
            dimensions.remove(dim_to_drop)
            return DimensionSet(dim_list=dimensions)

    remove = drop

    @overload
    def replace(
        self, key: str, new_dim: Dimension, inplace: Literal[False] = False
    ) -> DimensionSet: ...
    @overload
    def replace(self, key: str, new_dim: Dimension, inplace: Literal[True]) -> None: ...
    @overload
    def replace(
        self, key: str, new_dim: Dimension, inplace: bool = False
    ) -> DimensionSet | None: ...
    def replace(self, key: str, new_dim: Dimension, inplace: bool = False) -> DimensionSet | None:
        """Replace a dimension in the set with a new one.

        Args:
            key: The name or letter of the dimension to replace
            new_dim: The new dimension to replace the old one
            inplace: If True, the operation is performed in place, otherwise a new DimensionSet is
                returned. Defaults to False.

        Returns:
            None if inplace=True, otherwise a new DimensionSet with the dimension replaced

        Raises:
            ValueError: If the letter of new_dim is already present in the set.
        """
        if new_dim.letter in self.letters:
            raise ValueError(
                "New dimension can't have same letter as any of those already in DimensionSet, "
                "as that would create ambiguity"
            )
        if inplace:
            self.dim_list[self.index(key)] = new_dim
            return
        else:
            dim_list = copy(self.dim_list)
            dim_list[self.index(key)] = new_dim
            return DimensionSet(dim_list=dim_list)

    def _prepare_other(self, other: DimensionSet | Dimension) -> DimensionSet:
        """Check that the other object is a DimensionSet or Dimension, and convert if necessary.

        Raises:
            TypeError: If other is neither a DimensionSet nor a Dimension.
        """
        if isinstance(other, DimensionSet):
            return other
        elif isinstance(other, Dimension):
            return other.as_dimset()
        else:
            raise TypeError("Operation of DimensionSet must be with DimensionSet or Dimension")

    def intersect_with(self, other: DimensionSet | Dimension) -> DimensionSet:
        """Get the intersection of two DimensionSets.

        Args:
            other: The other DimensionSet or Dimension to intersect with

        Returns:
            The intersection of the two DimensionSets
        """
        other = self._prepare_other(other)
        intersection_letters = tuple(letter for letter in self.letters if letter in other.letters)
        return self.get_subset(intersection_letters)

    def __and__(self, other: DimensionSet | Dimension) -> DimensionSet:
        """Intersection operator for two DimensionSets."""
        return self.intersect_with(other)

    def union_with(self, other: DimensionSet | Dimension) -> DimensionSet:
        """Get the union of two DimensionSets.

        Args:
            other: The other DimensionSet or Dimension to unite with

        Returns:
            The union of the two DimensionSets
        """
        other = self._prepare_other(other)
        added_dims = [dim for dim in other.dim_list if dim.letter not in self.letters]
        return self.expand_by(added_dims)

    def __or__(self, other: DimensionSet | Dimension) -> DimensionSet:
        """Union operator for two DimensionSets."""
        return self.union_with(other)

    def __add__(self, other: DimensionSet | Dimension) -> DimensionSet:
        """Addition operator for two DimensionSets (union of non-overlapping sets).

        Raises:
            ValueError: If the two sets share a dimension.
        """
        if self & other:
            raise ValueError("Dimensions of DimensionSets overlap. Use union '|' operator instead.")
        return self.union_with(other)

    def difference_with(self, other: DimensionSet | Dimension) -> DimensionSet:
        """Get the set difference of two DimensionSets.

        Args:
            other: The other DimensionSet or Dimension to compare with

        Returns:
            The difference of the two DimensionSets
        """
        other = self._prepare_other(other)
        difference_letters = tuple(letter for letter in self.letters if letter not in other.letters)
        return self.get_subset(difference_letters)

    def __sub__(self, other: DimensionSet | Dimension) -> DimensionSet:
        """Difference operator for two DimensionSets."""
        return self.difference_with(other)

    def __xor__(self, other: DimensionSet) -> DimensionSet:
        """Symmetric difference operator for two DimensionSets."""
        return (self - other) | (other - self)

    def _check_additional_dim(self, new_dim: Dimension) -> None:
        """Checks the new_dim is a Dimension and that it is not already in the DimensionSet."""
        if not isinstance(new_dim, Dimension):
            raise TypeError("new_dim must be a Dimension object.")
        if new_dim in self:
            raise ValueError(
                "New dimension can't have same letter as any of those already in DimensionSet, "
                "as that would create ambiguity."
            )
