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

    >>> from sodym import Dimension
    >>> regions = Dimension(name='Region', letter='r', items=['Earth', 'Moon', 'Sun'])

    The list of items can be loaded using a :py:class:`sodym.DataReader` object,
    or set directly, for example if a subset of an existing dimension is formed.
    """

    name: str = Field(..., min_length=2)
    letter: str = Field(
        ..., min_length=1, max_length=1, validation_alias=AliasChoices("letter", "dim_letter")
    )
    items: list
    dtype: Optional[type] = None
    """If given, a check is performed that all items have this datatype. Recommended for safety,
    especially in ambiguous cases such as calendar years (int or str)
    """

    @model_validator(mode="after")
    def items_have_datatype(self):
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
        return len(self.items)

    def index(self, item) -> int:
        return self.items.index(item)

    def is_subset(self, other: "Dimension"):
        return set(self.items).issubset(other.items)

    def is_superset(self, other: "Dimension"):
        return set(self.items).issuperset(other.items)


class DimensionSet(PydanticBaseModel):
    """A set of Dimension objects which MFA arrays are defined over.

    **Example**

        >>> from sodym import Dimension, DimensionSet
        >>> regions = Dimension(name='Region', letter='r', items=['Earth', 'Moon', 'Sun'])
        >>> time = Dimension(name='Time', letter='t', items=[1990, 2000, 2010, 2020, 2030])
        >>> dimensions = DimensionSet([regions, time])

    It is expected that DimensionSet instances are created via the :py:class:`sodym.DataReader`.

        >>> from sodym import DataReader, DimensionDefinition, Dimension
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

    @model_validator(mode="after")
    def no_repeated_dimensions(self):
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
        if isinstance(key, str):
            return self._dict[key]
        elif isinstance(key, int):
            return self.dim_list[key]
        else:
            raise TypeError("Key must be string or int")

    def __iter__(self) -> Iterator[Dimension]:
        return iter(self.dim_list)

    def size(self, key: str):
        return self._dict[key].len

    def shape(self, keys: tuple = None):
        keys = keys if keys else self.letters
        return tuple(self.size(key) for key in keys)

    @property
    def ndim(self):
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

    def drop(self, key: str, inplace: bool = False):
        dim_to_drop = self._dict[key]
        if inplace:
            self.dim_list.remove(dim_to_drop)
            return
        else:
            dimensions = copy(self.dim_list)
            dimensions.remove(dim_to_drop)
            return DimensionSet(dim_list=dimensions)

    def replace(self, key: str, new_dim: Dimension, inplace: bool = False):
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
        intersection_letters = [dim.letter for dim in self.dim_list if dim.letter in other.letters]
        return self.get_subset(intersection_letters)

    def union_with(self, other: "DimensionSet") -> "DimensionSet":
        added_dims = [dim for dim in other.dim_list if dim.letter not in self.letters]
        return self.expand_by(added_dims)

    def difference_with(self, other: "DimensionSet") -> "DimensionSet":
        difference_letters = [
            dim.letter for dim in self.dim_list if dim.letter not in other.letters
        ]
        return self.get_subset(difference_letters)

    @property
    def names(self):
        return tuple([dim.name for dim in self.dim_list])

    @property
    def letters(self):
        return tuple([dim.letter for dim in self.dim_list])

    @property
    def string(self):
        return "".join(self.letters)

    def index(self, key):
        return [d.letter for d in self.dim_list].index(key)
