"""The classes and methods defined here are building blocks for creating MFA systems.
This includes the base `FlodymArray` class and its helper the `SubArrayHandler`,
as well as applications of the `FlodymArray` for specific model components.
"""

from collections.abc import Iterable
from copy import deepcopy
from collections import defaultdict
import numpy as np
import pandas as pd
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, model_validator
from typing import Optional, Union

from .processes import Process
from .dimensions import DimensionSet, Dimension
from ._df_to_flodym_array import DataFrameToFlodymDataConverter


def _is_iterable(arg):
    return isinstance(arg, Iterable) and not isinstance(arg, (str, Dimension))


class FlodymArray(PydanticBaseModel):
    """Parent class for an array with pre-defined dimensions, which are addressed by name. Operations between
    different multi-dimensional arrays can than be performed conveniently, as the dimensions are automatically matched.

    In order to 'fix' the dimensions of the array, the array has to be 'declared' by calling the FlodymArray object
    constructor with a set of dimensions before working with it.
    Basic mathematical operations between FlodymArrays are defined, which return a FlodymArray object as a result.

    In order to set the values of a FlodymArray object to that of another one, the ellipsis slice ('[...]') can be
    used, e.g.
    foo[...] = bar.
    This ensures that the dimensionality of the array (foo) is not changed, and that the dimensionality of the
    right-hand side FlodymArray (bar) is consistent.
    While the syntaxes like of 'foo = bar + baz' are also possible (where 'bar' and 'baz' are FlodymArrays),
    it is not recommended, as it provides no control over the dimensionality of 'foo'. Use foo[...] = bar + baz instead.

    The values of the FlodymArray object are stored in a numpy array, and can be accessed directly via the 'values'
    attribute.
    So if type(bar) is np.ndarray, the operation
    foo.values[...] = bar
    is also possible.
    It is not recommended to use 'foo.values = bar' without the slice, as this might change the dimensionality of
    foo.values.

    Subsets of arrays can be set or retrieved.
    Here, slicing information is passed instead of the ellipsis to the square brackets of the FlodymArray, i.e.
    foo[keys] = bar or foo = bar[keys]. For details on the allowed values of 'keys', see the docstring of the
    SubArrayHandler class.

    The dimensions of a FlodymArray stored as a :py:class:`flodym.DimensionSet` object in the 'dims' attribute.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    dims: DimensionSet
    """Dimensions of the FlodymArray."""
    values: Optional[Union[np.ndarray, np.generic]] = None
    """Values of the FlodymArray. Must have the same shape as the dimensions of the FlodymArray. If None, an array of zeros is created."""
    name: Optional[str] = "unnamed"
    """Name of the FlodymArray."""

    @model_validator(mode="after")
    def validate_values(self):
        if self.values is None:
            self.values = np.zeros(self.dims.shape())
        self._check_value_format()
        return self

    def _check_value_format(self):
        if not isinstance(self.values, (np.ndarray, np.generic)):
            raise ValueError("Values must be a numpy array or numpy generic.")
        if self.dims.ndim > 0 and not isinstance(self.values, np.ndarray):
            raise ValueError("Values must be a numpy array, except for 0-dimensional arrays.")
        elif self.dims.ndim == 0 and isinstance(self.values, np.generic):
            self.values = np.array(self.values)

        if self.values.shape != self.dims.shape():
            raise ValueError(
                f"Values passed to {self.__class__.__name__} must have the same shape as the DimensionSet.\n"
                f"Array shape: {self.dims.shape()}\n"
                f"Values shape: {self.values.shape}\n"
            )

    @classmethod
    def from_dims_superset(
        cls, dims_superset: DimensionSet, dim_letters: tuple = None, **kwargs
    ) -> "FlodymArray":
        """Create a FlodymArray object from a superset of dimensions, by specifying which
        dimensions to take.

        Parameters:
            dims_superset: DimensionSet from which the objects dimensions are derived
            dim_letters: specify which dimensions to take from dims_superset
            kwargs: additional keyword arguments passed to the FlodymArray constructor

        Returns:
            cls instance
        """
        dims = dims_superset.get_subset(dim_letters)
        return cls(dims=dims, **kwargs)

    @classmethod
    def from_df(
        cls,
        dims: DimensionSet,
        df: pd.DataFrame,
        allow_missing_values: bool = False,
        allow_extra_values: bool = False,
        **kwargs,
    ) -> "FlodymArray":
        """Create a FlodymArray object from a DataFrame.

        Parameters:
            dims (DimensionSet): Dimensions of the FlodymArray
            df (DataFrame): pandas DataFrame containing the values of the FlodymArray.
                Dimensions of the named dim array can be given in DataFrame columns or the index.
                The DataFrame can be in long or wide format, that is there can either be one value column,
                or the value columns are named by items of one FlodymArray dimension.
                If dimension names are not given in the respective index or column, they are inferred from the
                items of the dimensions of the FlodymArray.
                It is advisable to give the dimension names in the DataFrame, as this makes the error messages
                more informative if there are typos in the items or if items are missing.
                Ordering of rows and columns is arbitrary, but the items across each dimension must be given,
                must be complete and exactly match those of the FlodymArray.
                Dimensions with only one item do not need to be given in the DataFrame.
                Supersets of dimensions (i.e. additional values) will lead to an error.
            allow_missing_values (bool, optional): Whether to allow missing values in the DataFrame.
                This includes both missing rows, and NaN values in the value column.
                Defaults to False.
            allow_extra_values (bool, optional): Whether to allow extra rows in the DataFrame,
                i.e. tows with index items not present in the FlodymArray dimension items.
                Defaults to False.

        Returns:
            FlodymArray: FlodymArray object with the values from the DataFrame
        """
        flodym_array = cls(dims=dims, **kwargs)
        flodym_array.set_values_from_df(
            df, allow_missing_values=allow_missing_values, allow_extra_values=allow_extra_values
        )
        return flodym_array

    def _sub_array_handler(self, definition) -> "SubArrayHandler":
        return SubArrayHandler(self, definition)

    @property
    def shape(self) -> tuple[int]:
        """The shape of the array, determined by the dimensions."""
        return self.dims.shape()

    @property
    def size(self) -> int:
        """The number of elements in the array."""
        return np.prod(self.shape)

    def set_values(self, values: np.ndarray):
        """Set the values of the FlodymArray and check if the shape is correct.

        For safety reasons, using scalars or broadcasting smaller arrays is not allowed,
        i.e. the shape of the values must match the shape of the FlodymArray.

        As a less safe but more flexible alternative, you can use e.g. flodym_array.values[...] = 3.5 to set the values directly.
        """
        self.values = values
        self._check_value_format()

    def sum_values(self):
        """Return the sum of all values in the FlodymArray."""
        return np.sum(self.values)

    def sum_values_over(self, sum_over_dims: tuple = ()):
        """Return the sum of the FlodymArray over a given tuple of dimensions.

        Args:
            sum_over_dims (tuple, optional): Tuple of dimension letters to sum over. If not given, no summation is performed and the values array is returned.

        Returns:
            np.ndarray: The partially summed values of the FlodymArray.
        """
        sum_over_dims = self._tuple_to_letters(sum_over_dims)
        result_dims = (o for o in self.dims.letters if o not in sum_over_dims)
        return np.einsum(f"{self.dims.string}->{''.join(result_dims)}", self.values)

    def cast_values_to(self, target_dims: DimensionSet):
        """Cast the values of the FlodymArray to a new set of dimensions.

        Args:
            target_dims (DimensionSet): New dimensions to cast the values to. Must be given as a DimensionSet object, as the new dimensions are otherwise not known to the FlodymArray object.

        Returns:
            np.ndarray: The values of the FlodymArray cast to the new dimensions.
        """
        assert all([d in target_dims.letters for d in self.dims.letters]), (
            "Target of cast must contain all "
            f"dimensions of the object! Source dims '{self.dims.string}' are not all contained in target dims "
            f"'{target_dims.string}'. Maybe use sum_values_to() before casting"
        )
        # safety procedure: order dimensions
        values = np.einsum(
            f"{self.dims.string}->{''.join([d for d in target_dims.letters if d in self.dims.letters])}",
            self.values,
        )
        index = tuple(
            [slice(None) if d in self.dims.letters else np.newaxis for d in target_dims.letters]
        )
        multiple = tuple([1 if d.letter in self.dims.letters else d.len for d in target_dims])
        values = values[index]
        values = np.tile(values, multiple)
        return values

    def cast_to(self, target_dims: DimensionSet):
        """Cast the FlodymArray to a new set of dimensions.

        Args:
            target_dims (DimensionSet): New dimensions to cast the FlodymArray to. Must be given as a DimensionSet object, as the new dimensions are otherwise not known to the FlodymArray object.

        Returns:
            FlodymArray: The FlodymArray cast to the new dimensions.
        """
        return FlodymArray(
            dims=target_dims, values=self.cast_values_to(target_dims), name=self.name
        )

    def sum_values_to(self, result_dims: tuple[str] = ()):
        """Return the values of the FlodymArray partially summed, such that only the dimensions given in the result_dims tuple are left.

        Args:
            result_dims (tuple, optional): Tuple of dimension letters to sum over. If not given, the sum over all dimensions is returned.
        """
        result_dims = self._tuple_to_letters(result_dims)
        return np.einsum(f"{self.dims.string}->{''.join(result_dims)}", self.values)

    def sum_to(self, result_dims: tuple = ()) -> "FlodymArray":
        """Return the FlodymArray summed, such that only the dimensions given in the result_dims tuple are left.

        Args:
            result_dims (tuple, optional): Tuple of the dimensions to sum to. If not given, the sum over all dimensions is returned.

        Returns:
            FlodymArray: FlodymArray object with the summed values and the reduced dimensions.
        """
        result_dims = self._tuple_to_letters(result_dims)
        return FlodymArray(
            dims=self.dims.get_subset(result_dims),
            values=self.sum_values_to(result_dims),
            name=self.name,
        )

    def sum_over(self, sum_over_dims: tuple = ()) -> "FlodymArray":
        """Return the FlodymArray summed over a given tuple of dimensions.

        Args:
            sum_over_dims (tuple, optional): Tuple of dimension letters to sum over. If not given, no summation is performed and the FlodymArray object is returned.

        Returns:
            FlodymArray: FlodymArray object with the summed values and the reduced dimensions.
        """
        sum_over_dims = self._tuple_to_letters(sum_over_dims)
        result_dims = tuple([d for d in self.dims.letters if d not in sum_over_dims])
        return FlodymArray(
            dims=self.dims.get_subset(result_dims),
            values=self.sum_values_over(sum_over_dims),
            name=self.name,
        )

    def _tuple_to_letters(self, dim_tuple: tuple) -> tuple:
        """Ensure that an input dimension tuple is converted to a tuple of dimension letters,
        if e.g. names or Dimension objects are given instead.

        Args:
            dim_tuple (tuple): Tuple of dimensions, which can be given as letters, names or Dimension objects.

        Returns:
            tuple: Tuple of dimension letters.
        """
        return tuple(self._get_dim_letter(item) for item in dim_tuple)

    def _get_dim_letter(self, dim: Union[str, Dimension]) -> str:
        """Get the letter of a dimension, given either the letter or the name of the dimension, or the Dimension object.

        Args:
            dim (Union[str, Dimension]): Dimension letter, name or object.

        Returns:
            str: Dimension letter.
        """
        if isinstance(dim, Dimension):
            return dim.letter
        elif isinstance(dim, str) and dim in self.dims:
            return self.dims[dim].letter
        else:
            raise KeyError(f"Dimension {dim} not found in FlodymArray dims.")

    def _prepare_other(self, other: Union["FlodymArray", int, float]) -> "FlodymArray":
        """If a math operation between a FlodymArray and an int or float is performed, the int or float is converted to a FlodymArray object.
        The following operations are then performed between the two FlodymArray objects.

        Args:
            other (Union[FlodymArray, int, float]): The other object to perform the operation with.

        Returns:
            FlodymArray: The other object converted to a FlodymArray object.
        """
        assert isinstance(other, (FlodymArray, int, float)), (
            "Can only perform operations between two " "FlodymArrays or FlodymArray and scalar."
        )
        if isinstance(other, (int, float)):
            other = FlodymArray(dims=self.dims, values=other * np.ones(self.shape))
        return other

    def __add__(self, other):
        other = self._prepare_other(other)
        dims_out = self.dims.intersect_with(other.dims)
        return FlodymArray(
            dims=dims_out,
            values=self.sum_values_to(dims_out.letters) + other.sum_values_to(dims_out.letters),
        )

    def __sub__(self, other):
        other = self._prepare_other(other)
        dims_out = self.dims.intersect_with(other.dims)
        return FlodymArray(
            dims=dims_out,
            values=self.sum_values_to(dims_out.letters) - other.sum_values_to(dims_out.letters),
        )

    def __mul__(self, other):
        other = self._prepare_other(other)
        dims_out = self.dims.union_with(other.dims)
        values_out = np.einsum(
            f"{self.dims.string},{other.dims.string}->{dims_out.string}", self.values, other.values
        )
        return FlodymArray(dims=dims_out, values=values_out)

    def __truediv__(self, other):
        other = self._prepare_other(other)
        dims_out = self.dims.union_with(other.dims)
        values_out = np.einsum(
            f"{self.dims.string},{other.dims.string}->{dims_out.string}",
            self.values,
            1.0 / other.values,
        )
        return FlodymArray(dims=dims_out, values=values_out)

    def minimum(self, other):
        other = self._prepare_other(other)
        dims_out = self.dims.intersect_with(other.dims)
        values_out = np.minimum(
            self.sum_values_to(dims_out.letters), other.sum_values_to(dims_out.letters)
        )
        return FlodymArray(dims=dims_out, values=values_out)

    def maximum(self, other):
        other = self._prepare_other(other)
        dims_out = self.dims.intersect_with(other.dims)
        values_out = np.maximum(
            self.sum_values_to(dims_out.letters), other.sum_values_to(dims_out.letters)
        )
        return FlodymArray(dims=dims_out, values=values_out)

    def apply(
        self, func: callable, kwargs: dict = {}, inplace: bool = False
    ) -> Optional["FlodymArray"]:
        """Apply a function to the values of the FlodymArray.

        Args:
            func (callable): Function to apply to the values. Must be a function that can be applied to a numpy array, and return a numpy array of the same shape.
            kwargs (dict): Keyword argument dictionary to pass to the function.
            inplace (bool, optional): Whether to apply the function in place. Defaults to False.

        Returns:
            FlodymArray: FlodymArray object with the values transformed by the function.
        """
        if inplace:
            self.values = func(self.values, **kwargs)
            return
        return FlodymArray(dims=self.dims, values=func(self.values, **kwargs))

    def abs(self, inplace: bool = False):
        return self.apply(np.abs, inplace=inplace)

    def sign(self, inplace: bool = False):
        return self.apply(np.sign, inplace=inplace)

    def cumsum(self, dim_letter: str, inplace: bool = False):
        """Calculate the cumulative sum along a dimension.

        Args:
            dim_letter (str): Dimension letter to calculate the cumulative sum along.
            inplace (bool, optional): Whether to apply the cumulative sum in place. Defaults to False.

        Returns:
            FlodymArray: FlodymArray object with the cumulative sum along the given dimension.
        """
        i_axis = self.dims.letters.index(dim_letter)
        return self.apply(np.cumsum, kwargs={"axis": i_axis}, inplace=inplace)

    def __neg__(self):
        return FlodymArray(dims=self.dims, values=-self.values)

    def __abs__(self):
        return FlodymArray(dims=self.dims, values=abs(self.values))

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return -self + other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        inv_self = FlodymArray(dims=self.dims, values=1 / self.values)
        return inv_self * other

    def __getitem__(self, keys):
        """Defines what is returned when the object with square brackets stands on the right-hand side of an assignment,
        e.g. foo = foo = bar[{'e': 'C'}] Here, it is solely used for slicing, the the input tot the square brackets must
        be a dictionary defining the slice."""
        return self._sub_array_handler(keys).to_flodym_array()

    def __setitem__(self, keys, item):
        """Defines what is returned when the object with square brackets stands on the left-hand side of an assignment,
        i.e. 'foo[bar] = baz' For allowed values in the square brackets (bar), see the docstring of the SubArrayHandler
        class.

        The RHS (baz) is required here to be a FlodymArray.
        If you want to set the values of a FlodymArray object directly to a numpy array, use the syntax
        'foo.values[...] = bar'."""
        assert isinstance(item, FlodymArray), "Item on RHS of assignment must be a FlodymArray"
        slice_obj = self._sub_array_handler(keys)
        self.values[slice_obj.ids] = item.sum_values_to(slice_obj.dim_letters)
        return

    def to_df(self, index: bool = True, dim_to_columns: str = None) -> pd.DataFrame:
        """Export the FlodymArray to a pandas DataFrame.

        Args:
            index (bool, optional): Whether to include the dimension items as a Multi-Index (True) or as columns of the DataFrame (False). Defaults to True.
            dim_to_columns (str, optional): Name of the dimension the items of which are to form the columns of the DataFrame. If not given, the DataFrame is returned in long format with a single 'value' column.

        Returns:
            pd.DataFrame: DataFrame representation of the FlodymArray.
        """
        multiindex = pd.MultiIndex.from_product([d.items for d in self.dims], names=self.dims.names)
        df = pd.DataFrame({"value": self.values.flatten()})
        df = df.set_index(multiindex)
        if dim_to_columns is not None:
            if dim_to_columns not in self.dims:
                raise ValueError(f"Dimension name {dim_to_columns} not found in flodym_array.dims")
            # transform to name, if given as letter
            dim_to_columns = self.dims[dim_to_columns].name
            df.reset_index(inplace=True)
            index_names = [n for n in self.dims.names if n != dim_to_columns]
            df = df.pivot(index=index_names, columns=dim_to_columns, values="value")
        if not index:
            df.reset_index(inplace=True)
        return df

    def set_values_from_df(
        self,
        df_in: pd.DataFrame,
        allow_missing_values: bool = False,
        allow_extra_values: bool = False,
    ):
        """Set the values of the FlodymArray from a pandas DataFrame.

        Parameters:
            df (DataFrame): pandas DataFrame containing the values of the FlodymArray.
                Dimensions of the named dim array can be given in DataFrame columns or the index.
                The DataFrame can be in long or wide format, that is there can either be one value column,
                or the value columns are named by items of one FlodymArray dimension.
                If dimension names are not given in the respective index or column, they are inferred from the
                items of the dimensions of the FlodymArray.
                It is advisable to give the dimension names in the DataFrame, as this makes the error messages
                more informative if there are typos in the items or if items are missing.
                Ordering of rows and columns is arbitrary, but the items across each dimension must be given,
                must be complete and exactly match those of the FlodymArray.
                Dimensions with only one item do not need to be given in the DataFrame.
                Supersets of dimensions (i.e. additional values) will lead to an error.
            allow_missing_values (bool, optional): Whether to allow missing values in the DataFrame.
                This includes both missing rows, and NaN values in the value column.
                Defaults to False.
            allow_extra_values (bool, optional): Whether to allow extra rows in the DataFrame,
                i.e. rows with index items not present in the FlodymArray dimension items.
                Defaults to False.
        """
        converter = DataFrameToFlodymDataConverter(
            df_in,
            self,
            allow_missing_values=allow_missing_values,
            allow_extra_values=allow_extra_values,
        )
        self.set_values(converter.target_values)

    def split(self, dim_letter: str) -> dict:
        """Reverse the flodym_array_stack, returns a dictionary of FlodymArray objects
        associated with the item in the dimension that has been split.
        Method can be applied to classes FlodymArray, StockArray, Parameter and Flow.
        """
        return {item: self[{dim_letter: item}] for item in self.dims[dim_letter].items}

    def get_shares_over(self, dim_letters: tuple) -> "FlodymArray":
        """Get shares of the FlodymArray along a tuple of dimensions, indicated by letter."""
        assert all(
            [d in self.dims.letters for d in dim_letters]
        ), "Dimensions to get share of must be in the object"

        if all([d in dim_letters for d in self.dims.letters]):
            return self / self.sum_values()

        return self / self.sum_over(sum_over_dims=dim_letters)


class SubArrayHandler:
    """This class handles subsets of the 'values' numpy array of a FlodymArray object, created by slicing along one or
    several dimensions. It specifies the behavior of `foo[definition] = bar` and `foo = bar[definition]`, where `foo` and `bar`
    are FlodymArray objects. This is done via the `__getitem__` and `__setitem__` methods of the FlodymArray class.

    It returns either

    - a new FlodymArray object (via the `to_flodym_array()` function), or
    - a pointer to a subset of the values array of the parent FlodymArray object, via the `values_pointer` attribute.

    There are several possible syntaxes for the definition of the subset:

    - An ellipsis slice `...` can be used to address all the values of the original FlodymArray object

      *Example:* `foo[...]` addresses all values of the FlodymArray object `foo`.
    - A dictionary can be used to define a subset along one or several dimensions.
      The dictionary has the form `{'dim_letter': 'item_name'}`.

      *Example:* `foo[{'e': 'C'}]` addresses all values where the element is carbon,

      Instead of a single 'item_name', a list of 'item_names' can be passed.

      *Example:* `foo[{'e': 'C', 'r': ['EUR', 'USA']}]` addresses all values where the element is carbon and the region is
      Europe or the USA.
    - Instead of a dictionary, an item name can be passed directly. In this case, the dimension is inferred from the
      item name.
      Throws an error if the item name is not unique, i.e. occurs in more than one dimension.

      *Example:* `foo['C']` addresses all values where the element is carbon

      Several comma-separated item names can be passed, which appear in `__getitem__` and `__setitem__` methods as a tuple.
      Those can either be in the same dimension or in different dimensions.

      *Example:* `foo['C', 'EUR', 'USA']` addresses all values where the element is carbon and the region is Europe or the
      USA.

    Note that does not inherit from FlodymArray, so it is not a FlodymArray object itself.
    However, one can use it to create a FlodymArray object with the `to_flodym_array()` method.
    """

    def __init__(self, flodym_array: FlodymArray, definition):
        self.flodym_array = flodym_array
        self._get_def_dict(definition)
        self.invalid_flodym_array = any(_is_iterable(v) for v in self.def_dict.values())
        self._init_dims_out()
        self._init_ids()

    def _get_def_dict(self, definition):
        if isinstance(definition, type(Ellipsis)):
            self.def_dict = {}
        elif isinstance(definition, dict):
            self.def_dict = definition
        elif isinstance(definition, tuple):
            self.def_dict = self._to_dict_tuple(definition)
        else:
            self.def_dict = {self._get_key_single_item(definition): definition}

    def _get_key_single_item(self, item):
        if isinstance(item, slice):
            raise ValueError(
                "Numpy indexing of FlodymArrays is not supported. Details are given in the FlodymArray class "
                "docstring."
            )
        key = None
        for d in self.flodym_array.dims:
            if item in d.items:
                if key is not None:
                    raise ValueError(
                        f"Ambiguous slicing: Item '{item}' is found in multiple dimensions. Please specify the "
                        "dimension by using a slicing dict instead."
                    )
                key = d.letter
        if key is None:
            raise ValueError(f"Slicing item '{item}' not found in any dimension.")
        return key

    def _to_dict_tuple(self, slice_def) -> dict:
        dict_out = defaultdict(list)
        for item in slice_def:
            key = self._get_key_single_item(item)
            dict_out[key].append(item)
        # if there is only one item along a dimension, convert list to single item
        return {k: v if len(v) > 1 else v[0] for k, v in dict_out.items()}

    @property
    def ids(self):
        """Indices used for slicing the values array."""
        return tuple(self._id_list)

    @property
    def values_pointer(self):
        """Pointer to the subset of the values array of the parent FlodymArray object."""
        return self.flodym_array.values[self.ids]

    def _init_dims_out(self):
        self.dims_out = deepcopy(self.flodym_array.dims)
        for letter, value in self.def_dict.items():
            if isinstance(value, Dimension):
                self.dims_out.replace(letter, value, inplace=True)
            elif not _is_iterable(value):
                self.dims_out.drop(letter, inplace=True)

    @property
    def dim_letters(self):
        """Updated dimension letters, where sliced dimensions with only one item along that direction are removed."""
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

        return FlodymArray(
            dims=self.dims_out, values=self.values_pointer, name=self.flodym_array.name
        )

    def _init_ids(self):
        """
        - Init the internal list of index slices to slice(None) (i.e. no slicing, keep all items along that dimension)
        - For each dimension that is sliced, get the corresponding item IDs and set the index slice to these IDs.
        """
        self._id_list = [slice(None) for _ in self.flodym_array.dims.letters]
        for dim_letter, item_or_items in self.def_dict.items():
            self._set_ids_single_dim(dim_letter, item_or_items)

    def _set_ids_single_dim(self, dim_letter, item_or_items):
        """Given either a single item name or a list of item names, return the corresponding item IDs, along one
        dimension 'dim_letter'."""
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
        self._id_list[self.flodym_array.dims.index(dim_letter)] = items_ids

    def _get_single_item_id(self, dim_letter, item_name):
        return self.flodym_array.dims[dim_letter].items.index(item_name)


class Flow(FlodymArray):
    """The values of Flow objects are the main computed outcome of the MFA system.
    A Flow object connects two :py:class:`Process` objects.
    The name of the Flow object is set as a combination of the names of the two processes it connects.

    Flow is a subclass of :py:class:`FlodymArray`, so most of its methods are inherited.

    **Example**

        >>> from flodym import DimensionSet, Flow, Process
        >>> goods = Dimension(name='Good', letter='g', items=['Car', 'Bus', 'Bicycle'])
        >>> time = Dimension(name='Time', letter='t', items=[1990, 2000, 2010, 2020, 2030])
        >>> dimensions = DimensionSet([goods, time])
        >>> fabrication = Process(name='fabrication', id=2)
        >>> use = Process(name='use', id=3)
        >>> flow = Flow(from_process='fabrication', to_process='use', dims=dimensions)

    In the above example, we did not pass any values when initialising the Flow instance,
    and these would get filled with zeros.
    See the validation (filling) method in :py:class:`FlodymArray`.
    """

    model_config = ConfigDict(protected_namespaces=())

    from_process: Process
    """Process from which the flow originates."""
    to_process: Process
    """Process to which the flow goes."""

    @property
    def from_process_id(self):
        """ID of the process from which the flow originates."""
        return self.from_process.id

    @property
    def to_process_id(self):
        """ID of the process to which the flow goes."""
        return self.to_process.id


class StockArray(FlodymArray):
    """Stocks allow accumulation of material at a process, i.e. between two flows.

    StockArray inherits all its functionality from :py:class:`FlodymArray`.
    StockArray's are used in the :py:class:`flodym.Stock` for the inflow, outflow and stock.
    """

    pass


class Parameter(FlodymArray):
    """Parameter's can be used when defining the :py:meth:`flodym.MFASystem.compute` of a specific MFA system,
    to quantify the links between specific :py:class:`flodym.Stock` and :py:class:`Flow` objects,
    for example as the share of flows that go into one branch when the flow splits at a process.

    Parameter inherits all its functionality from :py:class:`FlodymArray`.
    """

    pass
