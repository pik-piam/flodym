from __future__ import annotations
from copy import copy
from pydantic import BaseModel as PydanticBaseModel, Field, AliasChoices, model_validator
from typing import Dict, Iterator, Optional
import numpy as np
import pandas as pd

from .mfa_definition import DimensionDefinition


class Dimension(PydanticBaseModel):
    """One of multiple dimensions over which MFA arrays are defined.

    Defined by a name, a letter for shorter addressing, and a list of items.

    **Example**

    >>> from flodym import Dimension
    >>> regions = Dimension(name='Region', letter='r', items=['Earth', 'Moon', 'Sun'])

    The list of items can be loaded using a :py:class:`flodym.DataReader` object,
    or set directly, for example if a subset of an existing dimension is formed.
    """

    name: str = Field(..., min_length=2)
    """The full name of the dimension"""
    letter: str = Field(
        ..., min_length=1, max_length=1, validation_alias=AliasChoices("letter", "dim_letter")
    )
    """A single index letter for shorter addressing of the dimension"""
    items: list
    """A list of items that are resolved along the dimension, e.g. years or regions"""
    dtype: Optional[type] = None
    """If given, a check is performed that all items have this datatype. Recommended for safety,
    especially in ambiguous cases such as calendar years (int or str)
    """

    @model_validator(mode="after")
    def items_have_datatype(self):
        """If a datatype is specified, check that all items have this datatype."""
        if self.dtype is not None and any([not isinstance(i, self.dtype) for i in self.items]):
            raise ValueError("All items must have the same datatype as specified in dtype.")
        return self

    @classmethod
    def from_df(cls, df: pd.DataFrame, definition: DimensionDefinition) -> "Dimension":
        """Create Dimension object from definition and a pandas DataFrame for the items.

        Args:
            df (pd.DataFrame): A single-column or single-row data frame, which is transformed to the dimension items.
            definition (DimensionDefinition): The definition of the dimension.

        Returns:
            Dimension: The dimension object.
        """
        data = df.to_numpy()
        return cls.from_np(data, definition)

    @classmethod
    def from_np(cls, data: np.ndarray, definition: DimensionDefinition) -> "Dimension":
        """Create Dimension object from definition and a numpy array for the items.
        Performs checks on the array and transforms to list of items.

        Args:
            data (np.ndarray): A numpy array where only one dimension exceeds size 1.
            definition (DimensionDefinition): The definition object

        Returns:
            Dimension: The dimension object.
        """
        if sum([s > 1 for s in data.shape]) > 1:
            raise ValueError(
                f"Dimension data for {definition.name} must have only one row or column."
            )
        data = data.flatten().tolist()
        # delete header for items if present
        if data[0] == definition.name:
            data = data[1:]
        data = [definition.dtype(item) for item in data]
        return cls(
            name=definition.name, letter=definition.letter, items=data, dtype=definition.dtype
        )

    @property
    def len(self) -> int:
        """The number of items in the dimension."""
        return len(self.items)

    def index(self, item) -> int:
        """Return the index of an item in the dimension."""
        return self.items.index(item)

    def is_subset(self, other: "Dimension"):
        """Check if the items of this dimension are a subset of the items of another dimension."""
        return set(self.items).issubset(other.items)

    def is_superset(self, other: "Dimension"):
        """Check if the items of this dimension are a superset of the items of another dimension."""
        return set(self.items).issuperset(other.items)

    def as_dimset(self) -> "DimensionSet":
        """Convert the Dimension to a DimensionSet containing only this Dimension."""
        return DimensionSet(dim_list=[self])

    def __add__(self, other: "Dimension") -> "DimensionSet":
        """Addition operator to create a DimensionSet from two Dimensions."""
        if isinstance(other, Dimension):
            return DimensionSet(dim_list=[self, other])
        elif isinstance(other, DimensionSet):
            return self.as_dimset() + other
        else:
            raise TypeError("Can only add Dimension to Dimension or DimensionSet.")

    def __str__(self):
        base = f"Dimension '{self.name}' ('{self.letter}'); "
        item_base = f"{self.len} items"
        type_info = f" (type {self.dtype})" if self.dtype is not None else ""
        if self.len <= 3:
            list_str = f": {str(self.items)}"
        else:
            list_str = f": ['{self.items[0]}', ..., '{self.items[-1]}']"
        return base + item_base + type_info + list_str


class DimensionSet(PydanticBaseModel):
    """A set of Dimension objects which MFA arrays are defined over.

    **Example**

        >>> from flodym import Dimension, DimensionSet
        >>> regions = Dimension(name='Region', letter='r', items=['Earth', 'Moon', 'Sun'])
        >>> time = Dimension(name='Time', letter='t', items=[1990, 2000, 2010, 2020, 2030])
        >>> dimensions = DimensionSet([regions, time])

    It is expected that DimensionSet instances are created via the :py:class:`flodym.DataReader`.

        >>> from flodym import DataReader, DimensionDefinition, Dimension
        >>> class MyDataReader(DataReader):
        >>>    def read_dimension(self, dimension_definition: DimensionDefinition) -> Dimension:
        >>>        if dimension_definition.letter == 't':
        >>>            return Dimension(name='Time', letter='t', items=[1990, 2000, 2010, 2020, 2030])
        >>>        elif dimension_definition.letter == 'r':
        >>>            return Dimension(name='Region', letter='r', items=['Earth', 'Moon', 'Sun'])
        >>>        raise ValueError('No data available for desired dimension')
        >>> data_reader = MyDataReader()
        >>> time_definition = DimensionDefinition(name='Time', letter='t', dtype=int)
        >>> region_definition = DimensionDefinition(name='Region', letter='r', dtype=str)
        >>> definitions = [time_definition, region_definition]
        >>> dimensions = data_reader.read_dimensions(dimension_definitions=definitions)

    """

    dim_list: list[Dimension]
    """A list of Dimension objects defining the set"""

    @model_validator(mode="after")
    def no_repeated_dimensions(self):
        """Check that all dimensions have unique letters."""
        letters = self.letters
        if len(letters) != len(set(letters)):
            raise ValueError("Dimensions must have unique letters in DimensionSet.")
        return self

    @model_validator(mode="after")
    def copy_dim_list(self):
        """Ensure each DimensionSet has its own copy of the dim_list to avoid inplace operations
        affecting other DimensionSets."""
        self.dim_list = copy(self.dim_list)
        return self

    @classmethod
    def empty(cls) -> "DimensionSet":
        """Return an empty DimensionSet."""
        return cls(dim_list=[])

    @property
    def _full_mapping(self) -> Dict[str, Dimension]:
        """Contains mappings.

        letter --> dim object and name --> dim object
        """
        return {dim.name: dim for dim in self.dim_list} | {dim.letter: dim for dim in self.dim_list}

    def __getitem__(self, key) -> Dimension:
        """Get a dimension by its name, letter or index with the [] operator.

        Args:
            key (str, int, tuple): The name, letter or index of the dimension to get; If a tuple is passed, a subset of the dimensions is returned as a new DimensionSet.

        Example:
        >>> dimensions['Region']
        >>> dimensions['r']
        >>> dimensions[0]
        >>> dimensions['Region', 'Time']
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
        if isinstance(key, Dimension):
            key = key.letter
        return key in self._full_mapping

    def size(self, key: str):
        """get the number of items in a dimension

        Args:
            key (str): the name or letter of the dimension to get the size of
        """
        return self._full_mapping[key].len

    @property
    def shape(self) -> tuple[int]:
        """shape of the array that would be created with the dimensions in the set"""
        return tuple(self.size(dim) for dim in self.letters)

    @property
    def total_size(self) -> int:
        """size (total number of elements) of the array that would be created with the dimensions in the set"""
        return int(np.prod(self.shape))

    def copy(self) -> "DimensionSet":
        """Return a copy of the DimensionSet."""
        return self.model_copy(update={"dim_list": copy(self.dim_list)})

    def get_subset(self, dims: tuple = None) -> "DimensionSet":
        """Selects :py:class:`Dimension` objects from the object attribute dim_list,
        according to the dims passed, which can be either letters or names.
        Returns a copy if dims are not given.

        Args:
            dims (tuple, optional): A tuple of dimension letters or names to select. Defaults to None.
        """
        subset = self.model_copy()
        if dims is not None:
            subset.dim_list = [self._full_mapping[dim_key] for dim_key in dims]
        return subset

    def expand_by(self, added_dims: list[Dimension], inplace: bool = False) -> "DimensionSet":
        """Expands the DimensionSet by adding new dimensions to it.

        Args:
            added_dims (list[Dimension]): A list of Dimension objects to add
            inplace (bool, optional): If True, the operation is performed in place, otherwise a new DimensionSet is returned. Defaults to False.

        Returns:
            None if inplace=True, otherwise a new DimensionSet with the new dimensions added
        """
        if not all([dim.letter not in self.letters for dim in added_dims]):
            raise ValueError(
                "DimensionSet already contains one or more of the dimensions to be added."
            )
        if inplace:
            self.dim_list.extend(added_dims)
            return
        else:
            return DimensionSet(dim_list=self.dim_list + added_dims)

    extend = expand_by

    def _check_additional_dim(self, new_dim: Dimension):
        """Checks the new_dim is a Dimension and that it is not already in the DimensionSet."""
        if not isinstance(new_dim, Dimension):
            raise TypeError("new_dim must be a Dimension object.")
        if new_dim in self:
            raise ValueError(
                "New dimension can't have same letter as any of those already in DimensionSet, "
                "as that would create ambiguity."
            )

    def append(self, new_dim: Dimension, inplace: bool = False) -> Optional["DimensionSet"]:
        """Add a new dimension to the set at the end.

        Args:
            new_dim (Dimension): The new dimension to add
            inplace (bool, optional): If True, the operation is performed in place, otherwise a new DimensionSet is returned. Defaults to False.

        Returns:
            None if inplace=True, otherwise a new DimensionSet with the new dimension added
        """
        self._check_additional_dim(new_dim)
        if inplace:
            self.dim_list.append(new_dim)
            return
        else:
            return self + new_dim

    def prepend(self, new_dim: Dimension, inplace: bool = False) -> Optional["DimensionSet"]:
        """Add a new dimension to the set at the beginning.

        Args:
            new_dim (Dimension): The new dimension to add
            inplace (bool, optional): If True, the operation is performed in place, otherwise a new DimensionSet is returned. Defaults to False.

        Returns:
            None if inplace=True, otherwise a new DimensionSet with the new dimension added
        """
        self._check_additional_dim(new_dim)
        if inplace:
            self.dim_list.insert(0, new_dim)
            return
        else:
            return new_dim + self

    def insert(
        self, index: int, new_dim: Dimension, inplace: bool = False
    ) -> Optional["DimensionSet"]:
        """Insert a new dimension to the set at the given index.

        Args:
            index (int): The index at which to insert the new dimension
            new_dim (Dimension): The new dimension to add
            inplace (bool, optional): If True, the operation is performed in place, otherwise a new DimensionSet is returned. Defaults to False.

        Returns:
            None if inplace=True, otherwise a new DimensionSet with the new dimension added
        """
        self._check_additional_dim(new_dim)
        if inplace:
            self.dim_list.insert(index, new_dim)
            return
        else:
            dim_list = copy(self.dim_list)
            dim_list.insert(index, new_dim)
            return DimensionSet(dim_list=dim_list)

    def drop(self, key: str, inplace: bool = False) -> Optional["DimensionSet"]:
        """Remove a dimension from the set.

        Args:
            key (str): The name, index or letter of the dimension to drop
            inplace (bool, optional): If True, the operation is performed in place, otherwise a new DimensionSet is returned. Defaults to False.

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

    def replace(self, key: str, new_dim: Dimension, inplace: bool = False):
        """Replace a dimension in the set with a new one.

        Args:
            key (str): The name, index or letter of the dimension to replace
            new_dim (Dimension): The new dimension to replace the old one
            inplace (bool, optional): If True, the operation is performed in place, otherwise a new DimensionSet is returned. Defaults to False.

        Returns:
            None if inplace=True, otherwise a new DimensionSet with the dimension replaced
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

    def prepare_other(self, other: "DimensionSet" | Dimension) -> "DimensionSet":
        """Check that the other object is a DimensionSet or Dimension, and convert if necessary."""
        if isinstance(other, DimensionSet):
            return other
        elif isinstance(other, Dimension):
            return other.as_dimset()
        else:
            raise TypeError("Operation of DimensionSet must be with DimensionSet or Dimension")

    def intersect_with(self, other: "DimensionSet" | Dimension) -> "DimensionSet":
        """Get the intersection of two DimensionSets.

        Args:
            other (DimensionSet): The other DimensionSet to intersect with

        Returns:
            DimensionSet: The intersection of the two DimensionSets
        """
        other = self.prepare_other(other)
        intersection_letters = [dim.letter for dim in self.dim_list if dim.letter in other.letters]
        return self.get_subset(intersection_letters)

    def __and__(self, other: "DimensionSet" | Dimension) -> "DimensionSet":
        """Intersection operator for two DimensionSets."""
        return self.intersect_with(other)

    def union_with(self, other: "DimensionSet" | Dimension) -> "DimensionSet":
        """Get the union of two DimensionSets.

        Args:
            other (DimensionSet): The other DimensionSet to unite with

        Returns:
            DimensionSet: The union of the two DimensionSets
        """
        other = self.prepare_other(other)
        added_dims = [dim for dim in other.dim_list if dim.letter not in self.letters]
        return self.expand_by(added_dims)

    def __or__(self, other: "DimensionSet" | Dimension) -> "DimensionSet":
        """Union operator for two DimensionSets."""
        return self.union_with(other)

    def __add__(self, other: "DimensionSet" | Dimension) -> "DimensionSet":
        """Addition operator for two DimensionSets (alias for union)."""
        if self & other:
            raise ValueError("Dimensions of DimensionSets overlap. Use union '|' operator instead.")
        return self.union_with(other)

    def difference_with(self, other: "DimensionSet" | Dimension) -> "DimensionSet":
        """Get the set difference of two DimensionSets.

        Args:
            other (DimensionSet): The other DimensionSet to compare with

        Returns:
            DimensionSet: The difference of the two DimensionSets
        """
        other = self.prepare_other(other)
        difference_letters = [
            dim.letter for dim in self.dim_list if dim.letter not in other.letters
        ]
        return self.get_subset(difference_letters)

    def __sub__(self, other: "DimensionSet" | Dimension) -> "DimensionSet":
        """Difference operator for two DimensionSets."""
        return self.difference_with(other)

    def __xor__(self, other: "DimensionSet" | Dimension) -> "DimensionSet":
        """Symmetric difference operator for two DimensionSets."""
        return (self - other) | (other - self)

    @property
    def ndim(self):
        """the number of dimensions in the set"""
        return len(self.dim_list)

    def __len__(self) -> int:
        """Return the number of dimensions in the set."""
        return len(self.dim_list)

    def __bool__(self) -> bool:
        """Return True if the set is not empty."""
        return len(self.dim_list) > 0

    @property
    def names(self):
        """A tuple of the names of the dimensions in the set."""
        return tuple([dim.name for dim in self.dim_list])

    @property
    def letters(self):
        """A tuple of the letters of the dimensions in the set."""
        return tuple([dim.letter for dim in self.dim_list])

    @property
    def string(self):
        """The letters of the dimensions in the set concatenated to a single string."""
        return "".join(self.letters)

    def index(self, key):
        """Return the index of a dimension in the set.

        Args:
            key (str): The name or letter of the dimension to get the index of
        """
        dim = self._full_mapping[key]
        return self.dim_list.index(dim)

    def __str__(self):
        base = f"DimensionSet ({','.join(self.letters)}) with shape {self.shape}:"
        dim_strs = [
            f"\n  '{dim.letter}': '{dim.name}' with length {dim.len}" for dim in self.dim_list
        ]
        return base + "".join(dim_strs)
