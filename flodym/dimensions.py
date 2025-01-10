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

    @property
    def _dict(self) -> Dict[str, Dimension]:
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
            return self._dict[key]
        elif isinstance(key, int):
            return self.dim_list[key]
        else:
            raise TypeError("Key must be string or int")

    def __iter__(self) -> Iterator[Dimension]:
        return iter(self.dim_list)

    def __contains__(self, key: str) -> bool:
        return key in self._dict

    def size(self, key: str):
        """get the number of items in a dimension

        Args:
            key (str): the name or letter of the dimension to get the size of
        """
        return self._dict[key].len

    def shape(self, keys: tuple = None):
        """get the shape of the array that would be created with the dimensions in the set

        Args:
            keys (tuple, optional): the names or letters of the dimensions to get the shape of. If None, all dimensions are used. Defaults to None.
        """
        keys = keys if keys else self.letters
        return tuple(self.size(key) for key in keys)

    @property
    def ndim(self):
        """the number of dimensions in the set"""
        return len(self.dim_list)

    def get_subset(self, dims: tuple = None) -> "DimensionSet":
        """Selects :py:class:`Dimension` objects from the object attribute dim_list,
        according to the dims passed, which can be either letters or names.
        Returns a copy if dims are not given.
        """
        subset = copy(self)
        if dims is not None:
            subset.dim_list = [self._dict[dim_key] for dim_key in dims]
        return subset

    def expand_by(self, added_dims: list[Dimension]) -> "DimensionSet":
        """Expands the DimensionSet by adding new dimensions to it."""
        if not all([dim.letter not in self.letters for dim in added_dims]):
            raise ValueError(
                "DimensionSet already contains one or more of the dimensions to be added."
            )
        return DimensionSet(dim_list=self.dim_list + added_dims)

    def drop(self, key: str, inplace: bool = False) -> Optional["DimensionSet"]:
        """Remove a dimension from the set.

        Args:
            key (str): The name, index or letter of the dimension to drop
            inplace (bool, optional): If True, the operation is performed in place, otherwise a new DimensionSet is returned. Defaults to False.

        Returns:
            None if inplace=True, otherwise a new DimensionSet with the dimension removed
        """
        dim_to_drop = self._dict[key]
        if inplace:
            self.dim_list.remove(dim_to_drop)
            return
        else:
            dimensions = copy(self.dim_list)
            dimensions.remove(dim_to_drop)
            return DimensionSet(dim_list=dimensions)

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

    def intersect_with(self, other: "DimensionSet") -> "DimensionSet":
        """Get the intersection of two DimensionSets.

        Args:
            other (DimensionSet): The other DimensionSet to intersect with

        Returns:
            DimensionSet: The intersection of the two DimensionSets
        """
        intersection_letters = [dim.letter for dim in self.dim_list if dim.letter in other.letters]
        return self.get_subset(intersection_letters)

    def union_with(self, other: "DimensionSet") -> "DimensionSet":
        """Get the union of two DimensionSets.

        Args:
            other (DimensionSet): The other DimensionSet to unite with

        Returns:
            DimensionSet: The union of the two DimensionSets
        """
        added_dims = [dim for dim in other.dim_list if dim.letter not in self.letters]
        return self.expand_by(added_dims)

    def difference_with(self, other: "DimensionSet") -> "DimensionSet":
        """Get the set difference of two DimensionSets.

        Args:
            other (DimensionSet): The other DimensionSet to compare with

        Returns:
            DimensionSet: The difference of the two DimensionSets
        """
        difference_letters = [
            dim.letter for dim in self.dim_list if dim.letter not in other.letters
        ]
        return self.get_subset(difference_letters)

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
        """Return the index of a dimension in the set."""
        return [d.letter for d in self.dim_list].index(key)
