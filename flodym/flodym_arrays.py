"""The classes and methods defined here are building blocks for creating MFA systems.

This includes the base `FlodymArray` class and its helper the `_SubArrayHandler`,
as well as applications of the `FlodymArray` for specific model components.
"""

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable
from copy import copy, deepcopy
from numbers import Number, Real
from typing import Any, Literal, Optional, SupportsFloat, TypeVar, Union, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import (
    BaseModel as PydanticBaseModel,
)
from pydantic import (
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    ValidationInfo,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from ._config import config, handle_error
from ._df_to_flodym_array import DataFrameToFlodymDataConverter
from .dimensions import Dimension, DimensionSet
from .processes import Process

T = TypeVar("T", bound="FlodymArray")


def _is_iterable(arg: object) -> bool:
    """Whether ``arg`` is an iterable of items, i.e. not a string or a :py:class:`Dimension`."""
    return isinstance(arg, Iterable) and not isinstance(arg, (str, Dimension))


class FlodymArray(PydanticBaseModel):
    """Parent class for an array with pre-defined dimensions, which are addressed by name.

    Operations between different multi-dimensional arrays can then be performed conveniently, as
    the dimensions are automatically matched.

    In order to 'fix' the dimensions of the array, the array has to be 'declared' by calling the
    FlodymArray object constructor with a set of dimensions before working with it.
    Basic mathematical operations between FlodymArrays are defined, which return a FlodymArray
    object as a result.

    In order to set the values of a FlodymArray object to that of another one, the ellipsis slice
    ('[...]') can be used, e.g.
    foo[...] = bar.
    This ensures that the dimensionality of the array (foo) is not changed, and that the
    dimensionality of the right-hand side FlodymArray (bar) is consistent.
    While the syntaxes like of 'foo = bar + baz' are also possible (where 'bar' and 'baz' are
    FlodymArrays), it is not recommended, as it provides no control over the dimensionality of
    'foo'. Use foo[...] = bar + baz instead.

    The values of the FlodymArray object are stored in a numpy array, and can be accessed directly
    via the 'values' attribute.
    So if type(bar) is np.ndarray, the operation
    foo.values[...] = bar
    is also possible.
    It is not recommended to use 'foo.values = bar' without the slice, as this might change the
    dimensionality of foo.values.

    Subsets of arrays can be set or retrieved.
    Here, slicing information is passed instead of the ellipsis to the square brackets of the
    FlodymArray, i.e. `foo[keys] = bar` or `foo = bar[keys]`. Retrieving returns a new
    FlodymArray. The allowed values of `keys` are:

    - An ellipsis slice `...` addresses all the values of the FlodymArray.

      *Example:* `foo[...]` addresses all values of `foo`.
    - A dictionary of the form `{'dim_letter': 'item_name'}` defines a subset along one or
      several dimensions. Instead of a single 'item_name', a list of 'item_names' can be passed.

      *Example:* `foo[{'e': 'C'}]` addresses all values where the element is carbon.
      `foo[{'e': 'C', 'r': ['EUR', 'USA']}]` addresses all values where the element is carbon and
      the region is Europe or the USA.
    - An item name can be passed directly. In this case, the dimension is inferred from the item
      name. An error is raised if the item name occurs in more than one dimension.
      Several comma-separated item names can be passed, in the same or in different dimensions.

      *Example:* `foo['C']` addresses all values where the element is carbon.
      `foo['C', 'EUR', 'USA']` addresses all values where the element is carbon and the region is
      Europe or the USA.

    The dimensions of a FlodymArray stored as a :py:class:`flodym.DimensionSet` object in the
    'dims' attribute.

    **Extensive and intensive arrays**

    Reducing an array along a dimension by summation is only correct for extensive quantities,
    which scale with the size of the system (e.g. masses). Expanding an array along a new
    dimension by copying its values is only correct for intensive quantities, which do not (e.g.
    shares, yields, concentrations). The :py:attr:`extensive` flag states which of the two an
    array holds. It is None (unknown) by default, True for :py:class:`Flow` and
    :py:class:`StockArray`, and set for parameters by :py:meth:`flodym.MFASystem.link_parameters`.

    - Arrays with ``extensive=None`` follow the conventions described above: Dimensions are
      summed when reducing and copied when expanding.
    - In ``+``, ``-``, :py:meth:`minimum` and :py:meth:`maximum` of intensive arrays, dimensions
      which only one operand has are copied to the other operand instead of being summed over.
    - In ``foo[...] = bar`` with an intensive ``bar``, ``bar`` is copied along the dimensions of
      ``foo`` which it does not have. Dimensions of ``bar`` which ``foo`` does not have are
      reported, since no correct reduction exists without weights.
    - :py:meth:`sum_to` and :py:meth:`sum_over` of a not extensive array, and
      :py:meth:`cast_to` of a not intensive array are performed, but reported.

    Reports are handled according to ``flodym.config.error_behaviors.extensivity``.

    The flag is propagated through operations by an extensivity degree, which is the exponent
    with which the quantity scales with the size of the system: 1 for extensive, 0 for intensive
    arrays. Degrees add up in ``*`` and are subtracted in ``/``. Operands of ``+`` and ``-`` must
    have the same degree. For example, the quotient of two flows is intensive, and the product
    of two flows divided by a third flow is extensive. The flag of an array with a degree other
    than 0 or 1 is None.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    dims: DimensionSet
    """Dimensions of the FlodymArray."""
    values: np.ndarray = Field(default=None, validate_default=True)
    """Values of the FlodymArray. Must have the same shape as the dimensions of the FlodymArray.
    If not given or None, an array of zeros is created."""
    name: str | None = "unnamed"
    """Name of the FlodymArray."""
    extensive: bool | None = None
    """Whether the array holds an extensive quantity (True), such as a mass, or an intensive one
    (False), such as a share, yield or concentration. None if unknown.
    See "Extensive and intensive arrays" in the class docstring."""
    _is_set: bool = False
    """Flag indicating whether the array has been set or not."""
    _higher_degree: int | None = None
    """Extensivity degree of the array if it is neither 0 nor 1, see :py:attr:`_degree`.
    Reset when :py:attr:`extensive` is assigned."""

    @field_validator("values", mode="before")
    @classmethod
    def _values_to_numpy(cls, value: Any, info: ValidationInfo) -> Any:
        """Convert the ``values`` input to a numpy array.

        Fill a missing/None value with zeros of the correct shape, and wrap a scalar Number in a
        numpy array. This keeps the field type ``np.ndarray`` while preserving the flexible
        constructor input.
        """
        if isinstance(value, Number):
            return np.array(value)
        if value is None:
            dims = info.data.get("dims")
            if dims is not None:
                return np.zeros(dims.shape)
        return value

    @model_validator(mode="after")
    def _copy_dims(self) -> Self:
        """Ensure dims is always copied to avoid shared references."""
        self.dims = self.dims.copy()
        return self

    @model_validator(mode="after")
    def _validate_values(self) -> Self:
        """Check that the values are a numpy array with the shape of the dimensions."""
        self._check_value_format()
        return self

    def _check_value_format(self) -> None:
        if not isinstance(self.values, np.ndarray):
            raise TypeError("Values must be a numpy array.")
        if self.values.shape != self.dims.shape:
            raise ValueError(
                f"Values passed to {self.__class__.__name__} must have the same shape as the DimensionSet.\n"
                f"Array shape: {self.dims.shape}\n"
                f"Values shape: {self.values.shape}\n"
            )

    @model_validator(mode="wrap")
    @classmethod
    def _mark_set_or_unset(cls, data: Any, handler: ModelWrapValidatorHandler[Self]) -> Self:
        """Mark the array as set if values are given at initialization, else as unset."""
        obj = handler(data)
        if isinstance(data, dict):
            obj._is_set = data.get("values") is not None
        return obj

    def __str__(self) -> str:
        base = f"{self.__class__.__name__} '{self.name}'"
        dims = f" with dims ({','.join(self.dims.letters)}) and shape {self.shape};"
        values = f"\nValues:\n{self.values!s}"
        return base + dims + values

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "extensive":
            # an explicitly assigned flag replaces any degree tracked from operations
            self._higher_degree = None
        super().__setattr__(name, value)

    @classmethod
    def from_dims_superset(
        cls,
        dims_superset: DimensionSet,
        dim_letters: tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a FlodymArray object from a superset of dimensions.

        The dimensions to take are specified by ``dim_letters``.

        Args:
            dims_superset: DimensionSet from which the objects dimensions are derived
            dim_letters: specify which dimensions to take from dims_superset
            **kwargs: additional keyword arguments passed to the FlodymArray constructor

        Returns:
            cls instance
        """
        dims = dims_superset.get_subset(dim_letters)
        return cls(dims=dims, **kwargs)

    @classmethod
    def full(
        cls,
        dims: DimensionSet,
        fill_value: SupportsFloat | np.ndarray,
        **kwargs: Any,
    ) -> Self:
        """Create a FlodymArray filled with a constant value for the provided dimensions.

        Args:
            dims: DimensionSet defining the dimensions of the FlodymArray.
            fill_value: Value to fill the array with.
                Can be a scalar or an array that is broadcastable to the shape of dims.
            **kwargs: Additional keyword arguments passed to the FlodymArray constructor
                (e.g., name).

        Returns:
            A new FlodymArray filled with the specified value.
        """
        return cls(dims=dims, values=np.full(dims.shape, fill_value), **kwargs)

    @classmethod
    def full_like(
        cls,
        other: "FlodymArray",
        fill_value: SupportsFloat | np.ndarray,
        dtype: type | np.dtype | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a FlodymArray filled with a constant value, matching another array's dimensions.

        Args:
            other: FlodymArray whose dimensions will be used for the new array.
            fill_value: Value to fill the array with.
                Can be a scalar or an array that is broadcastable to the shape of other.
            dtype: Data type of the new array.
                If None, the data type of fill_value is used. Defaults to None.
            **kwargs: Additional keyword arguments passed to the FlodymArray constructor
                (e.g., name).

        Returns:
            A new FlodymArray with the same dimensions as other,
                filled with the specified value.
        """
        if dtype is None:
            dtype = getattr(fill_value, "dtype", type(fill_value))
        return cls(
            dims=other.dims.copy(),
            values=np.full_like(other.values, fill_value, dtype=dtype),
            **kwargs,
        )

    @classmethod
    def scalar(
        cls,
        value: SupportsFloat,
        **kwargs: Any,
    ) -> Self:
        """Create a scalar (zero-dimensional) FlodymArray.

        Args:
            value: The scalar value to store in the FlodymArray.
            **kwargs: Additional keyword arguments passed to the FlodymArray constructor
                (e.g., name).

        Returns:
            A zero-dimensional FlodymArray containing the scalar value.
        """
        return cls(dims=DimensionSet.empty(), values=np.array(value), **kwargs)

    @classmethod
    def from_df(
        cls,
        dims: DimensionSet,
        df: pd.DataFrame,
        allow_missing_values: bool = False,
        allow_extra_values: bool = False,
        strip_whitespace: bool = True,
        **kwargs: Any,
    ) -> Self:
        """Create a FlodymArray object from a DataFrame.

        In case of errors, turning on debug logging might help to understand the process.

        Args:
            dims: Dimensions of the FlodymArray
            df: pandas DataFrame containing the values of the FlodymArray.
                Dimensions of the named dim array can be given in DataFrame columns or the index.
                The DataFrame can be in long or wide format, that is there can either be one value
                column, or the value columns are named by items of one FlodymArray dimension.
                If dimension names or letters are not given in the respective index or column, they
                are inferred from the items of the dimensions of the FlodymArray.
                It is advisable to give the dimension names in the DataFrame, as this makes the
                error messages more informative if there are typos in the items or if items are
                missing.
                Ordering of rows and columns is arbitrary, but the items across each dimension must
                be given.
                Dimensions with only one item do not need to be given in the DataFrame.
                Supersets of dimensions (i.e. additional values) will lead to an error.
            allow_missing_values: Whether to allow missing values in the DataFrame.
                This includes both missing rows, and NaN values in the value column.
                Defaults to False.
            allow_extra_values: Whether to allow extra rows in the DataFrame,
                i.e. rows with index items not present in the FlodymArray dimension items.
                Defaults to False.
            strip_whitespace: Whether to strip whitespace from string values in the DataFrame.
                Defaults to True.
            **kwargs: Additional keyword arguments passed to the FlodymArray constructor
                (e.g., name).

        Returns:
            FlodymArray object with the values from the DataFrame
        """
        flodym_array = cls(dims=dims, **kwargs)
        flodym_array.set_values_from_df(
            df,
            allow_missing_values=allow_missing_values,
            allow_extra_values=allow_extra_values,
            strip_whitespace=strip_whitespace,
        )
        return flodym_array

    @classmethod
    def from_array(cls, array: "FlodymArray", name: str | None = None, **kwargs: Any) -> Self:
        """Create an instance of this class from an existing FlodymArray.

        The dimensions and values of ``array`` are copied.
        This is the class-side counterpart to :py:meth:`to_class`: ``Parameter.from_array(arr)``
        is equivalent to ``arr.to_class(Parameter)``.

        Args:
            array: Source array whose dimensions and values are copied.
            name: Name of the returned array. If not given, ``array``'s name is kept.
            **kwargs: Additional keyword arguments forwarded to this class's constructor.

        Returns:
            An instance of this class with copied values and dimensions.
        """
        return array.to_class(cls, name=name, **kwargs)

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the array, determined by the dimensions."""
        return self.dims.shape

    @property
    def size(self) -> int:
        """The number of elements in the array."""
        return int(np.prod(self.shape))

    def set_values(self, values: npt.ArrayLike) -> None:
        """Set the values of the FlodymArray and check if the shape is correct.

        For safety reasons, broadcasting smaller arrays is not allowed,
        i.e. the shape of the values must match the shape of the FlodymArray.

        As a less safe but more flexible alternative, you can use e.g.
        flodym_array.values[...] = foo to set the values directly.

        Args:
            values: A numpy array of the shape of the FlodymArray, or a scalar or nested sequence
                that is written into the existing values array.
        """
        if isinstance(values, (np.ndarray, FlodymArray)):
            # FlodymArray will actually throw an error in _check_value_format.
            # It is not explicitly checked in this routine to avoid duplicating the error message.
            self.values = values
            self._check_value_format()
        else:
            self.values[...] = values
        self._is_set = True

    def sum_values(self) -> np.ndarray:
        """Return the sum of all values in the FlodymArray."""
        return np.sum(self.values)

    def sum_values_over(self, sum_over_dims: tuple[str | Dimension, ...] = ()) -> np.ndarray:
        """Return the sum of the FlodymArray over a given tuple of dimensions.

        Args:
            sum_over_dims: Tuple of the dimensions to sum over. If not given, no summation is
                performed and the values array is returned.

        Returns:
            The partially summed values of the FlodymArray.
        """
        sum_over_dims = self._tuple_to_letters(sum_over_dims)
        result_dims = (o for o in self.dims.letters if o not in sum_over_dims)
        return np.einsum(f"{self.dims.string}->{''.join(result_dims)}", self.values)

    def cast_values_to(self, target_dims: DimensionSet) -> np.ndarray:
        """Cast the values of the FlodymArray to a new set of dimensions.

        Args:
            target_dims: New dimensions to cast the values to. Must be given as a DimensionSet
                object, as the new dimensions are otherwise not known to the FlodymArray object.

        Returns:
            The values of the FlodymArray cast to the new dimensions.
        """
        assert all(d in target_dims.letters for d in self.dims.letters), (
            "Target of cast must contain all "
            f"dimensions of the object! Source dims '{self.dims.string}' are not all contained in target dims "
            f"'{target_dims.string}'. Maybe use sum_values_to() before casting"
        )
        # safety procedure: order dimensions
        axes = [self.dims.letters.index(d) for d in target_dims.letters if d in self.dims.letters]
        values = np.transpose(self.values, axes)
        index = tuple(
            [slice(None) if d in self.dims.letters else np.newaxis for d in target_dims.letters]
        )
        multiple = tuple([1 if d.letter in self.dims.letters else d.len for d in target_dims])
        values = values[index]
        values = np.tile(values, multiple)
        return values

    @overload
    def cast_to(
        self, target_dims: DimensionSet, inplace: Literal[False] = ...
    ) -> "FlodymArray": ...
    @overload
    def cast_to(self: T, target_dims: DimensionSet, inplace: Literal[True]) -> T: ...
    def cast_to(self, target_dims: DimensionSet, inplace: bool = False) -> "FlodymArray":
        """Cast the FlodymArray to a new set of dimensions.

        Copying values along new dimensions is only correct for intensive quantities. Casting an
        array which is not intensive (see :py:attr:`extensive`) is reported.

        Args:
            target_dims: New dimensions to cast the FlodymArray to. Must be given as a
                DimensionSet object, as the new dimensions are otherwise not known to the
                FlodymArray object.
            inplace: If True, modify the current FlodymArray in place. If False, return a new
                FlodymArray object.

        Returns:
            The FlodymArray cast to the new dimensions.
        """
        self._check_is_set("cast_to()")
        # dimension consistency checks are performed in cast_values_to
        values = self.cast_values_to(target_dims)
        self._check_expansion(target_dims - self.dims, "cast_to()")
        if inplace:
            self.dims = target_dims
            self.values = values
            return self
        else:
            result = FlodymArray(dims=target_dims, values=values, name=self.name)
            result._is_set = self._is_set
            return result._with_degree(self._degree)

    def sum_values_to(self, result_dims: tuple[str | Dimension, ...] = ()) -> np.ndarray:
        """Return the values of the FlodymArray partially summed to the given dimensions.

        Only the dimensions given in the result_dims tuple are left.

        Args:
            result_dims: Tuple of the dimensions to sum to. If not given, the sum over all
                dimensions is returned.

        Returns:
            The partially summed values of the FlodymArray.
        """
        result_letters = self._tuple_to_letters(result_dims)
        return np.einsum(f"{self.dims.string}->{''.join(result_letters)}", self.values)

    def sum_to(self, result_dims: tuple[str | Dimension, ...] = ()) -> "FlodymArray":
        """Return the FlodymArray summed to the given dimensions.

        Only the dimensions given in the result_dims tuple are left.

        Summation is only correct for extensive quantities. Summing over dimensions of an array
        which is not extensive (see :py:attr:`extensive`) is reported.

        Args:
            result_dims: Tuple of the dimensions to sum to. If not given, the sum over all
                dimensions is returned.

        Returns:
            FlodymArray object with the summed values and the reduced dimensions.
        """
        self._check_is_set("sum_to()")
        result_letters = self._tuple_to_letters(result_dims)
        result_dims_set = self.dims.get_subset(result_letters)
        self._check_reduction(self.dims - result_dims_set, "sum_to()")
        return FlodymArray(
            dims=result_dims_set,
            values=self.sum_values_to(result_letters),
            name=self.name,
        )._with_degree(self._degree)

    def sum_over(self, sum_over_dims: tuple[str | Dimension, ...] = ()) -> "FlodymArray":
        """Return the FlodymArray summed over a given tuple of dimensions.

        Summation is only correct for extensive quantities. Summing over dimensions of an array
        which is not extensive (see :py:attr:`extensive`) is reported.

        Args:
            sum_over_dims: Tuple of the dimensions to sum over. If not given, no summation is
                performed and the FlodymArray object is returned.

        Returns:
            FlodymArray object with the summed values and the reduced dimensions.
        """
        self._check_is_set("sum_over()")
        sum_over_dims = self._tuple_to_letters(sum_over_dims)
        result_dims = tuple([d for d in self.dims.letters if d not in sum_over_dims])
        result_dims_set = self.dims.get_subset(result_dims)
        self._check_reduction(self.dims - result_dims_set, "sum_over()")
        return FlodymArray(
            dims=result_dims_set,
            values=self.sum_values_over(sum_over_dims),
            name=self.name,
        )._with_degree(self._degree)

    def _tuple_to_letters(self, dim_tuple: tuple[str | Dimension, ...]) -> tuple[str, ...]:
        """Convert an input dimension tuple to a tuple of dimension letters.

        This is needed if e.g. names or Dimension objects are given instead of letters.

        Args:
            dim_tuple: Tuple of dimensions, which can be given as letters, names or Dimension
                objects.

        Returns:
            Tuple of dimension letters.
        """
        return tuple(self._get_dim_letter(item) for item in dim_tuple)

    def _get_dim_letter(self, dim: str | Dimension) -> str:
        """Get the letter of a dimension, given its letter, its name, or the Dimension object.

        Args:
            dim: Dimension letter, name or object.

        Returns:
            Dimension letter.
        """
        if isinstance(dim, Dimension):
            return dim.letter
        elif isinstance(dim, str) and dim in self.dims:
            return self.dims[dim].letter
        else:
            raise KeyError(f"Dimension {dim} not found in FlodymArray dims.")

    def _prepare_other(self, other: Union["FlodymArray", SupportsFloat]) -> "FlodymArray":
        """Prepare the other operand of a math operation, converting a scalar to a FlodymArray.

        If a math operation between a FlodymArray and a scalar is performed, the scalar is
        converted to a FlodymArray object. The following operations are then performed between
        the two FlodymArray objects.

        Args:
            other: The other object to perform the operation with.

        Returns:
            The other object converted to a FlodymArray object.
        """
        assert isinstance(other, (FlodymArray, Number)), (
            "Can only perform operations between two FlodymArrays or FlodymArray and scalar."
        )
        if isinstance(other, Number):
            other = FlodymArray(dims=self.dims, values=np.full(self.shape, other))
        self._check_is_set("a math operation")
        other._check_is_set("a math operation")
        return other

    def _operand_degree(
        self, other: Union["FlodymArray", SupportsFloat], additive: bool
    ) -> int | None:
        """Extensivity degree of the other operand of a math operation.

        A scalar has the degree of this array in additive operations (``+``, ``-``,
        :py:meth:`minimum`, :py:meth:`maximum`), and degree 0 otherwise.
        """
        if isinstance(other, FlodymArray):
            return other._degree
        return self._degree if additive else 0

    def _additive_degree(
        self, other: "FlodymArray", other_degree: int | None, operation: str
    ) -> int | None:
        """Extensivity degree of the result of an additive operation.

        Both operands must have the same degree. An operand of unknown degree is assumed to have
        the degree of the other one. Different known degrees are reported, and the degree of the
        result is unknown.
        """
        own_degree = self._degree
        if own_degree is None:
            return other_degree
        if other_degree is None or other_degree == own_degree:
            return own_degree
        handle_error(
            config.error_behaviors.extensivity,
            message=(
                f"{_array_label(self, own_degree)} and {_array_label(other, other_degree)} are "
                f"combined in {operation}, which requires both to have the same extensivity."
            ),
        )
        return None

    def _additive_operation(
        self,
        other: Union["FlodymArray", SupportsFloat],
        func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        operation: str,
    ) -> "FlodymArray":
        """Perform ``+``, ``-``, :py:meth:`minimum` or :py:meth:`maximum`.

        If the result is intensive, it has the dimensions of both operands, and each operand is
        copied along the dimensions it does not have. Otherwise, the result has the dimensions
        shared by both operands, and each operand is summed over the other dimensions.
        """
        other_degree = self._operand_degree(other, additive=True)
        other = self._prepare_other(other)
        degree = self._additive_degree(other, other_degree, operation)
        if degree == 0:
            dims_out = self.dims.union_with(other.dims)
            values_out = func(self.cast_values_to(dims_out), other.cast_values_to(dims_out))
        else:
            dims_out = self.dims.intersect_with(other.dims)
            if degree is not None:
                self._check_reduction(self.dims - dims_out, operation, degree)
                other._check_reduction(other.dims - dims_out, operation, degree)
            values_out = func(
                self.sum_values_to(dims_out.letters), other.sum_values_to(dims_out.letters)
            )
        return FlodymArray(dims=dims_out, values=values_out)._with_degree(degree)

    def _power_degree(self, power: Union["FlodymArray", SupportsFloat]) -> int | None:
        """Extensivity degree of this array raised to the given power.

        A scalar power multiplies the degree; the result is unknown if this gives no integer.
        An array power must be intensive; the result is intensive if this array is, and unknown
        otherwise.
        """
        if isinstance(power, FlodymArray):
            if power._degree not in (0, None):
                handle_error(
                    config.error_behaviors.extensivity,
                    message=(
                        f"{_array_label(power, power._degree)} is used as exponent of "
                        f"{_array_label(self, self._degree)}. Exponents must be intensive."
                    ),
                )
            return 0 if self._degree == 0 else None
        if self._degree is None or not isinstance(power, Real):
            return None
        product = self._degree * float(power)
        return int(product) if product.is_integer() else None

    def __add__(self, other: Union["FlodymArray", SupportsFloat]) -> "FlodymArray":
        return self._additive_operation(other, np.add, "an addition")

    def __sub__(self, other: Union["FlodymArray", SupportsFloat]) -> "FlodymArray":
        return self._additive_operation(other, np.subtract, "a subtraction")

    def __mul__(self, other: Union["FlodymArray", SupportsFloat]) -> "FlodymArray":
        degree = _combine_degrees(self._degree, self._operand_degree(other, additive=False), 1)
        other = self._prepare_other(other)
        dims_out = self.dims.union_with(other.dims)
        values_out = np.einsum(
            f"{self.dims.string},{other.dims.string}->{dims_out.string}", self.values, other.values
        )
        return FlodymArray(dims=dims_out, values=values_out)._with_degree(degree)

    def __truediv__(self, other: Union["FlodymArray", SupportsFloat]) -> "FlodymArray":
        degree = _combine_degrees(self._degree, self._operand_degree(other, additive=False), -1)
        other = self._prepare_other(other)
        dims_out = self.dims.union_with(other.dims)
        values_out = np.einsum(
            f"{self.dims.string},{other.dims.string}->{dims_out.string}",
            self.values,
            1.0 / other.values,
        )
        return FlodymArray(dims=dims_out, values=values_out)._with_degree(degree)

    def __pow__(self, power: Union["FlodymArray", SupportsFloat]) -> "FlodymArray":
        degree = self._power_degree(power)
        power = self._prepare_other(power)
        if any(l not in self.dims.letters for l in power.dims.letters):
            raise ValueError("Power must only contain dimensions also present in the base array.")
        values_out = self.values ** power.cast_values_to(self.dims)
        return FlodymArray(dims=self.dims, values=values_out)._with_degree(degree)

    def __neg__(self) -> "FlodymArray":
        self._check_is_set("a math operation")
        return FlodymArray(dims=self.dims, values=-self.values)._with_degree(self._degree)

    def __abs__(self) -> "FlodymArray":
        self._check_is_set("a math operation")
        return FlodymArray(dims=self.dims, values=abs(self.values))._with_degree(self._degree)

    def __radd__(self, other: SupportsFloat) -> "FlodymArray":
        return self + other

    def __rsub__(self, other: SupportsFloat) -> "FlodymArray":
        return -self + other

    def __rmul__(self, other: SupportsFloat) -> "FlodymArray":
        return self * other

    def __rtruediv__(self, other: SupportsFloat) -> "FlodymArray":
        self._check_is_set("a math operation")
        inv_self = FlodymArray(dims=self.dims, values=1 / self.values)
        inv_self._set_degree(_combine_degrees(0, self._degree, -1))
        return inv_self * other

    def minimum(self, other: Union["FlodymArray", SupportsFloat]) -> "FlodymArray":
        """Element-wise minimum of this array and another array or scalar.

        As for addition, the result has the dimensions shared by both arrays; other dimensions
        are summed over before the comparison. If both arrays are intensive, the result has the
        dimensions of both arrays instead (see the class docstring).
        """
        return self._additive_operation(other, np.minimum, "minimum()")

    def maximum(self, other: Union["FlodymArray", SupportsFloat]) -> "FlodymArray":
        """Element-wise maximum of this array and another array or scalar.

        As for addition, the result has the dimensions shared by both arrays; other dimensions
        are summed over before the comparison. If both arrays are intensive, the result has the
        dimensions of both arrays instead (see the class docstring).
        """
        return self._additive_operation(other, np.maximum, "maximum()")

    def apply(
        self,
        func: Callable[..., np.ndarray],
        kwargs: dict[str, Any] | None = None,
        inplace: bool = False,
    ) -> Optional["FlodymArray"]:
        """Apply a function to the values of the FlodymArray.

        As the function is arbitrary, the returned array has an unknown extensivity
        (``extensive=None``); set :py:attr:`extensive` afterwards if it is known. If ``inplace``
        is True, the flag of this array is kept.

        Args:
            func: Function to apply to the values. Must be a function that can be applied to a
                numpy array, and return a numpy array of the same shape.
            kwargs: Keyword argument dictionary to pass to the function.
            inplace: Whether to apply the function in place. Defaults to False.

        Returns:
            FlodymArray object with the values transformed by the function,
                or None if ``inplace`` is True.
        """
        return self._apply(func, kwargs, inplace, degree=self._degree if inplace else None)

    def _apply(
        self,
        func: Callable[..., np.ndarray],
        kwargs: dict[str, Any] | None,
        inplace: bool,
        degree: int | None,
    ) -> Optional["FlodymArray"]:
        """Apply a function to the values, see :py:meth:`apply`, giving the result the degree."""
        self._check_is_set("apply()")
        if kwargs is None:
            kwargs = {}
        if inplace:
            self.values = func(self.values, **kwargs)
            self._set_degree(degree)
            return None
        return FlodymArray(dims=self.dims, values=func(self.values, **kwargs))._with_degree(degree)

    def copy(self) -> Self:
        """Return a copy of the FlodymArray.

        This method creates a new FlodymArray with deep copies of both the DimensionSet and the
        numpy values array, ensuring modifications to the copy do not affect the original.

        Returns:
            A new FlodymArray object with copied values and dimensions.
        """
        return self.model_copy(update={"dims": self.dims.copy(), "values": self.values.copy()})

    def to_class(self, array_class: type[T], name: str | None = None, **kwargs: Any) -> T:
        """Return an instance of a FlodymArray subclass with copied dimensions and values.

        Re-tags this array as a different FlodymArray (sub)class — e.g. converting a plain
        :py:class:`FlodymArray` into a :py:class:`Parameter`, :py:class:`StockArray`, or
        :py:class:`Flow`, or a subclass instance back to the base :py:class:`FlodymArray`.

        The extensivity (see :py:attr:`extensive`) of this array is kept, unless ``extensive``
        is given in ``kwargs`` or it is unknown; then, the default of ``array_class`` applies.
        Converting to a class whose default extensivity differs, e.g. an intensive array to a
        :py:class:`Flow`, is reported.

        Args:
            array_class: The FlodymArray subclass to convert to.
            name: Name of the returned array. If not given, the name of the
                current array is kept.
            **kwargs: Additional keyword arguments forwarded to ``array_class``'s constructor
                (e.g. ``from_process`` and ``to_process`` for a :py:class:`Flow`).

        Returns:
            An ``array_class`` instance with copied values and dimensions.

        Raises:
            TypeError: If ``array_class`` is not a FlodymArray subclass.
        """
        if not (isinstance(array_class, type) and issubclass(array_class, FlodymArray)):
            raise TypeError(f"array_class must be a FlodymArray subclass, got {array_class!r}.")
        result = array_class(
            dims=self.dims.copy(),
            values=self.values.copy(),
            name=self.name if name is None else name,
            **kwargs,
        )
        result._is_set = self._is_set
        if "extensive" not in kwargs and self._degree is not None:
            if result._degree is not None and result._degree != self._degree:
                handle_error(
                    config.error_behaviors.extensivity,
                    message=(
                        f"{_array_label(self, self._degree)} is converted to "
                        f"{array_class.__name__}, which is "
                        f"{_degree_description(result._degree)} by default."
                    ),
                )
            result._set_degree(self._degree)
        return result

    @overload
    def abs(self, inplace: Literal[False] = ...) -> "FlodymArray": ...
    @overload
    def abs(self, inplace: Literal[True]) -> None: ...
    def abs(self, inplace: bool = False) -> Optional["FlodymArray"]:
        """Element-wise absolute value.

        Args:
            inplace: Whether to modify the values of this array instead of returning a new one.

        Returns:
            A new FlodymArray with the absolute values, or None if ``inplace`` is True.
        """
        return self._apply(np.abs, None, inplace, degree=self._degree)

    @overload
    def sign(self, inplace: Literal[False] = ...) -> "FlodymArray": ...
    @overload
    def sign(self, inplace: Literal[True]) -> None: ...
    def sign(self, inplace: bool = False) -> Optional["FlodymArray"]:
        """Element-wise sign (-1, 0 or 1).

        The signs are intensive (see :py:attr:`extensive`), also if ``inplace`` is True.

        Args:
            inplace: Whether to modify the values of this array instead of returning a new one.

        Returns:
            A new FlodymArray with the signs of the values, or None if ``inplace`` is True.
        """
        return self._apply(np.sign, None, inplace, degree=0)

    @overload
    def cumsum(self, dim_letter: str, inplace: Literal[False] = ...) -> "FlodymArray": ...
    @overload
    def cumsum(self, dim_letter: str, inplace: Literal[True]) -> None: ...
    def cumsum(self, dim_letter: str, inplace: bool = False) -> Optional["FlodymArray"]:
        """Calculate the cumulative sum along a dimension.

        Args:
            dim_letter: Dimension letter to calculate the cumulative sum along.
            inplace: Whether to apply the cumulative sum in place. Defaults to False.

        Returns:
            FlodymArray object with the cumulative sum along the given dimension,
                or None if ``inplace`` is True.
        """
        i_axis = self.dims.letters.index(dim_letter)
        return self._apply(np.cumsum, {"axis": i_axis}, inplace, degree=self._degree)

    def __getitem__(self, keys: dict[str, Any] | Hashable) -> "FlodymArray":
        """Return a slice of the array, as in ``foo = bar[{'e': 'C'}]``.

        Defines what is returned when the object with square brackets stands on the right-hand
        side of an assignment, e.g. foo = bar[{'e': 'C'}]. Here, it is solely used for slicing.
        For allowed values in the square brackets, see the class docstring.
        """
        self._check_is_set("slicing")
        return self._sub_array_handler(keys).to_flodym_array()

    def __setitem__(
        self, keys: dict[str, Any] | Hashable, item: "FlodymArray | npt.ArrayLike"
    ) -> None:
        """Set the values of a slice of the array, as in ``foo[bar] = baz``.

        Defines what happens when the object with square brackets stands on the left-hand side of
        an assignment, i.e. 'foo[bar] = baz'. For allowed values in the square brackets (bar), see
        the class docstring.

        The RHS (baz) is either a FlodymArray, a numpy array of correct shape, or a scalar.
        If it is a numpy array, a copy is used to avoid modifying the original array.

        A FlodymArray RHS is summed over dimensions which the slice does not have. If it is
        intensive, it is instead copied along dimensions of the slice which it does not have,
        and dimensions which the slice does not have are reported (see :py:attr:`extensive`).
        If the extensivity of this array is unknown, it takes the one of the RHS; different
        known extensivities are reported.
        """
        slice_obj = self._sub_array_handler(keys)
        if isinstance(item, FlodymArray):
            item._check_is_set("an assignment to another array")
            self._check_assigned_degree(item)
            self.values[slice_obj.ids] = item._values_for_assignment(slice_obj.dim_letters)
        elif isinstance(keys, type(Ellipsis)):
            self.set_values(copy(item))
        else:
            self.values[slice_obj.ids] = copy(item)
        self._is_set = True

    def _sub_array_handler(self, definition: dict[str, Any] | Hashable) -> "_SubArrayHandler":
        return _SubArrayHandler(self, definition)

    def to_df(
        self, index: bool = True, dim_to_columns: str | None = None, sparse: bool = False
    ) -> pd.DataFrame:
        """Export the FlodymArray to a pandas DataFrame.

        Args:
            index: Whether to include the dimension items as a Multi-Index (True) or as columns of
                the DataFrame (False). Defaults to True.
            dim_to_columns: Name of the dimension the items of which are to form the columns of the
                DataFrame. If not given, the DataFrame is returned in long format with a single
                'value' column.
            sparse: Whether to return a sparse DataFrame with only non-zero values. Defaults to
                False.

        Returns:
            DataFrame representation of the FlodymArray.
        """
        if sparse:
            non_zero_ids: tuple[np.ndarray, ...] = np.nonzero(self.values)

            def to_index(i: int) -> np.ndarray:
                dim = self.dims[i]
                ids = non_zero_ids[i]
                return np.array(dim.items)[ids]

            multiindex = pd.MultiIndex.from_arrays(
                arrays=[to_index(i) for i in range(self.dims.ndim)], names=self.dims.names
            )
            df = pd.DataFrame({"value": self.values[non_zero_ids].flatten()})
            df = df.set_index(multiindex)
        else:
            multiindex = pd.MultiIndex.from_product(
                [d.items for d in self.dims], names=self.dims.names
            )
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
        strip_whitespace: bool = True,
    ) -> None:
        """Set the values of the FlodymArray from a pandas DataFrame.

        In case of errors, turning on debug logging might help to understand the process.

        Args:
            df_in: pandas DataFrame containing the values of the FlodymArray.
                Dimensions of the named dim array can be given in DataFrame columns or the index.
                The DataFrame can be in long or wide format, that is there can either be one value
                column, or the value columns are named by items of one FlodymArray dimension.
                If dimension names or letters are not given in the respective index or column, they
                are inferred from the items of the dimensions of the FlodymArray.
                It is advisable to give the dimension names in the DataFrame, as this makes the
                error messages more informative if there are typos in the items or if items are
                missing.
                Ordering of rows and columns is arbitrary, but the items across each dimension must
                be given.
                Dimensions with only one item do not need to be given in the DataFrame.
                Supersets of dimensions (i.e. additional values) will lead to an error.
            allow_missing_values: Whether to allow missing values in the DataFrame.
                This includes both missing rows, and NaN values in the value column.
                Defaults to False.
            allow_extra_values: Whether to allow extra rows in the DataFrame,
                i.e. rows with index items not present in the FlodymArray dimension items.
                Defaults to False.
            strip_whitespace: Whether to strip whitespace from string values in the DataFrame.
                Defaults to True.
        """
        converter = DataFrameToFlodymDataConverter(
            df_in,
            self,
            allow_missing_values=allow_missing_values,
            allow_extra_values=allow_extra_values,
            strip_whitespace=strip_whitespace,
        )
        self.set_values(converter.target_values)

    def split(self, dim_letter: str) -> "dict[Any, FlodymArray]":
        """Split the FlodymArray along a dimension into a dictionary of FlodymArray objects.

        This reverses the flodym_array_stack. Each returned FlodymArray is associated with an item
        in the dimension that has been split.
        Method can be applied to classes FlodymArray, StockArray, Parameter and Flow.

        Args:
            dim_letter: Letter of the dimension to split along.

        Returns:
            Dictionary with the items of the split dimension as keys and the corresponding
            slices (without the split dimension) as values.
        """
        return {item: self[{dim_letter: item}] for item in self.dims[dim_letter].items}

    def get_shares_over(self, dim_letters: str | tuple[str, ...]) -> "FlodymArray":
        """Get shares of the FlodymArray along one or more dimensions, indicated by letter.

        Args:
            dim_letters: The letter of a single dimension, or a tuple of
                dimension letters, to get the shares over.

        Returns:
            The FlodymArray divided by its sum over the given dimensions.
        """
        if isinstance(dim_letters, str):
            dim_letters = (dim_letters,)
        assert all(d in self.dims.letters for d in dim_letters), (
            "Dimensions to get share of must be in the object"
        )

        if all(d in dim_letters for d in self.dims.letters):
            return self / self.sum_to(())

        return self / self.sum_over(sum_over_dims=dim_letters)

    def items_where(self, condition: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Get the dimension item tuples of all entries where a condition is met.

        Args:
            condition: A function that takes the values of the FlodymArray and returns a boolean
                array.

        Returns:
            A 2d numpy array of strings, where each row corresponds to a dimension item tuple
        """
        indices = np.argwhere(condition(self.values))
        # map dim items to indices
        items = [
            np.array(self.dims[letter].items)[indices[:, i]]
            for i, letter in enumerate(self.dims.letters)
        ]
        return np.array(items).transpose()

    @property
    def is_set(self) -> bool:
        """Whether the values of the array are known.

        Arrays are set if values are given at initialization, or if values are assigned through
        one of the methods of the class (e.g. ``set_values()``, ``foo[...] = bar``).
        Writing to the ``values`` attribute directly (e.g. ``foo.values[0, ...] = bar``) does not
        mark the array as set; call :py:meth:`mark_set` afterwards in that case.

        What happens if an unset array is used in an operation is controlled by
        ``flodym.config.error_behaviors.unset_array_used``.
        """
        return self._is_set

    def mark_set(self) -> Self:
        """Mark the array as having known values.

        Returns the array itself, to allow e.g. ``FlodymArray(dims=dims).mark_set()`` for an array
        of intended zeros.
        """
        self._is_set = True
        return self

    def mark_unset(self) -> Self:
        """Mark the array as not having known values.

        The values themselves are kept. Returns the array itself.
        """
        self._is_set = False
        return self

    @property
    def _absolute_float_precision(self) -> float:
        """The machine epsilon of the data type, multiplied by the maximum absolute value.

        NaN values are ignored.
        """
        abs_values = np.abs(self.values)
        max_value = float(np.max(abs_values, where=~np.isnan(abs_values), initial=0.0))
        dtype = self.values.dtype if np.issubdtype(self.values.dtype, np.inexact) else np.float64
        return float(np.finfo(dtype).eps) * max_value

    @property
    def _degree(self) -> int | None:
        """Extensivity degree: the exponent with which the quantity scales with the system size.

        1 for extensive, 0 for intensive arrays (see :py:attr:`extensive`). Other degrees result
        from operations, e.g. 2 for the product of two extensive arrays. None if unknown.
        """
        if self.extensive is None:
            return self._higher_degree
        return 1 if self.extensive else 0

    def _set_degree(self, degree: int | None) -> None:
        """Set :py:attr:`extensive` and the hidden degree from an extensivity degree."""
        self.extensive = {1: True, 0: False}.get(degree) if degree is not None else None
        if degree not in (0, 1):
            self._higher_degree = degree

    def _with_degree(self, degree: int | None) -> Self:
        """Set the extensivity degree (see :py:meth:`_set_degree`) and return the array."""
        self._set_degree(degree)
        return self

    def _check_is_set(self, operation: str) -> None:
        if not self._is_set:
            handle_error(
                config.error_behaviors.unset_array_used,
                message=(
                    f"{self.__class__.__name__} '{self.name}' is used in {operation}, but is not set "
                    "(is_set=False). If its values are intended, call mark_set() on it."
                ),
            )

    def _check_reduction(
        self, removed_dims: DimensionSet, operation: str, degree: int | None = None
    ) -> None:
        """Report summation over dimensions if the array is not extensive.

        Args:
            removed_dims: Dimensions summed over. Dimensions with a single item are ignored.
            operation: Description of the operation, for the message.
            degree: Extensivity degree to check. Defaults to the degree of the array.
        """
        if degree is None:
            degree = self._degree
        names = [dim.name for dim in removed_dims if dim.len > 1]
        if degree in (1, None) or not names:
            return
        handle_error(
            config.error_behaviors.extensivity,
            message=(
                f"{_array_label(self, degree)} is summed over the dimensions {names} in "
                f"{operation}. Summation is only correct for extensive quantities."
            ),
        )

    def _check_expansion(self, added_dims: DimensionSet, operation: str) -> None:
        """Report copying values along new dimensions if the array is not intensive.

        Args:
            added_dims: Dimensions added. Dimensions with a single item are ignored.
            operation: Description of the operation, for the message.
        """
        names = [dim.name for dim in added_dims if dim.len > 1]
        if self._degree in (0, None) or not names:
            return
        handle_error(
            config.error_behaviors.extensivity,
            message=(
                f"{_array_label(self, self._degree)} is copied along the dimensions {names} in "
                f"{operation}. Copying is only correct for intensive quantities."
            ),
        )

    def _check_assigned_degree(self, item: "FlodymArray") -> None:
        """Check the degree of an array assigned to this one, or take it if this one has none."""
        if self._degree is None:
            self._set_degree(item._degree)
        elif item._degree is not None and item._degree != self._degree:
            handle_error(
                config.error_behaviors.extensivity,
                message=(
                    f"{_array_label(item, item._degree)} is assigned to "
                    f"{_array_label(self, self._degree)}."
                ),
            )

    def _values_for_assignment(self, target_letters: tuple[str, ...]) -> np.ndarray:
        """Values of this array, to be assigned to a slice with the given dimension letters.

        Dimensions which the slice does not have are summed over; this is reported if the array
        is not extensive. An intensive array gets size-one axes for the dimensions of the slice
        which it does not have, along which it is broadcast when assigned. For other arrays,
        such dimensions raise an error.

        Raises:
            ValueError: If the array is not intensive and lacks dimensions of the slice.
        """
        extra_dims = self.dims - self.dims.get_subset(
            tuple(letter for letter in self.dims.letters if letter in target_letters)
        )
        self._check_reduction(extra_dims, "an assignment")
        missing = [letter for letter in target_letters if letter not in self.dims.letters]
        if missing and self._degree != 0:
            raise ValueError(
                f"{_array_label(self, self._degree)} is assigned to an array with the dimensions "
                f"{list(missing)}, which it does not have. Values are only copied along missing "
                "dimensions for intensive arrays (extensive=False). Use cast_to() to copy them "
                "explicitly."
            )
        kept = tuple(letter for letter in target_letters if letter in self.dims.letters)
        values = self.sum_values_to(kept)
        index = tuple(slice(None) if letter in kept else np.newaxis for letter in target_letters)
        return values[index]


def _combine_degrees(degree: int | None, other_degree: int | None, sign: int) -> int | None:
    """Degree of a product (``sign=1``) or quotient (``sign=-1``); None if one is unknown."""
    if degree is None or other_degree is None:
        return None
    return degree + sign * other_degree


def _degree_description(degree: int | None) -> str:
    """Description of an extensivity degree, for messages."""
    descriptions = {1: "extensive", 0: "intensive", None: "of unknown extensivity"}
    return descriptions.get(degree, f"of extensivity degree {degree}")


def _array_label(array: FlodymArray, degree: int | None) -> str:
    """Class, name and extensivity of an array, for messages."""
    return f"{array.__class__.__name__} '{array.name}' ({_degree_description(degree)})"


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
    """

    def __init__(self, flodym_array: FlodymArray, definition: dict[str, Any] | Hashable) -> None:
        self.flodym_array = flodym_array
        self._get_def_dict(definition)
        self.invalid_flodym_array = any(_is_iterable(v) for v in self.def_dict.values())
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


class _PrescribableArray(FlodymArray):
    """Base class of arrays whose values can be prescribed by another array.

    The prescribing array is usually a :py:class:`Parameter` (see
    :py:attr:`flodym.FlowDefinition.prescribed` and :py:class:`flodym.StockDefinition`).
    """

    _prescribed: str | FlodymArray | None = None
    """The array prescribing the values, or its parameter name before it is linked by
    :py:meth:`flodym.MFASystem.link_parameters`. Values are copied by
    :py:meth:`flodym.MFASystem.set_arrays_from_prescribed`."""

    def _set_prescribed(self, prescribed: str | FlodymArray | None) -> None:
        """Set the prescribing array or its parameter name, or remove it by passing None.

        An array must have all dimensions of this array; further dimensions are summed over
        when the values are copied.
        """
        if isinstance(prescribed, FlodymArray):
            missing_dims = self.dims - prescribed.dims
            if missing_dims:
                raise ValueError(
                    f"Array '{prescribed.name}' prescribed for '{self.name}' does not have its "
                    f"dimensions {missing_dims.names}."
                )
        elif prescribed is not None and not isinstance(prescribed, str):
            raise TypeError(
                f"Prescribed array of '{self.name}' must be a FlodymArray or a parameter name, "
                f"not {type(prescribed).__name__}."
            )
        self._prescribed = prescribed

    def _set_arrays_from_prescribed(self) -> None:
        """Copy the values of the prescribing array into this array, if one is given."""
        if self._prescribed is None:
            return
        if isinstance(self._prescribed, str):
            raise TypeError(
                f"Parameter '{self._prescribed}' prescribed for '{self.name}' is not linked. "
                "Call MFASystem.link_parameters() first."
            )
        self.__setitem__(..., self._prescribed)


class Flow(_PrescribableArray):
    """The values of Flow objects are the main computed outcome of the MFA system.

    A Flow object connects two :py:class:`Process` objects.
    The name of the Flow object is set as a combination of the names of the two processes it
    connects.

    Flow is a subclass of :py:class:`FlodymArray`, so most of its methods are inherited.

    **Example**

        >>> from flodym import DimensionSet, Flow, Process
        >>> goods = Dimension(name="Good", letter="g", items=["Car", "Bus", "Bicycle"])
        >>> time = Dimension(name="Time", letter="t", items=[1990, 2000, 2010, 2020, 2030])
        >>> dimensions = DimensionSet([goods, time])
        >>> fabrication = Process(name="fabrication", id=2)
        >>> use = Process(name="use", id=3)
        >>> flow = Flow(from_process=fabrication, to_process=use, dims=dimensions)

    In the above example, we did not pass any values when initialising the Flow instance,
    and these would get filled with zeros.
    See the validation (filling) method in :py:class:`FlodymArray`.
    """

    model_config = ConfigDict(protected_namespaces=())

    from_process: Process
    """Process from which the flow originates."""
    to_process: Process
    """Process to which the flow goes."""
    extensive: bool | None = True
    """Whether the flow is extensive, see :py:attr:`FlodymArray.extensive`. True by default."""
    _named_after_processes: bool = False
    """Whether the name was generated from the connected processes (see
    :py:attr:`flodym.Config.flow_naming`), and is thus updated by
    :py:meth:`flodym.MFASystem.relink`."""

    @classmethod
    def from_array(
        cls,
        array: "FlodymArray",
        from_process: Process | None = None,
        to_process: Process | None = None,
        name: str | None = None,
    ) -> "Flow":
        """Create a Flow from an existing FlodymArray, copying its dimensions and values.

        Args:
            array: Source array whose dimensions and values are copied.
            from_process: Process from which the flow originates. Inherited
                from ``array`` if it is a Flow and this is not given.
            to_process: Process to which the flow goes. Inherited from
                ``array`` if it is a Flow and this is not given.
            name: Name of the returned Flow. Defaults to ``array``'s name.

        Returns:
            A Flow instance with copied values and dimensions.
        """
        if from_process is None:
            from_process = getattr(array, "from_process", None)
        if to_process is None:
            to_process = getattr(array, "to_process", None)
        if from_process is None or to_process is None:
            raise ValueError(
                "from_process and to_process are required to create a Flow from a "
                "non-Flow array; they can only be inherited when the source array is a Flow."
            )
        return super().from_array(
            array, name=name, from_process=from_process, to_process=to_process
        )

    @property
    def from_process_id(self) -> int:
        """ID of the process from which the flow originates."""
        return self.from_process.id

    @property
    def to_process_id(self) -> int:
        """ID of the process to which the flow goes."""
        return self.to_process.id


class StockArray(_PrescribableArray):
    """Stocks allow accumulation of material at a process, i.e. between two flows.

    StockArray inherits all its functionality from :py:class:`FlodymArray`.
    StockArray's are used in the :py:class:`flodym.Stock` for the inflow, outflow and stock.
    """

    extensive: bool | None = True
    """Whether the array is extensive, see :py:attr:`FlodymArray.extensive`. True by default."""


class Parameter(FlodymArray):
    """Parameters quantify the links between stocks and flows in an MFA system.

    Parameter's can be used when defining the :py:meth:`flodym.MFASystem.compute` of a specific
    MFA system, to quantify the links between specific :py:class:`flodym.Stock` and
    :py:class:`Flow` objects, for example as the share of flows that go into one branch when the
    flow splits at a process.

    Parameter inherits all its functionality from :py:class:`FlodymArray`.
    Its :py:attr:`extensive` flag is None by default. It can be given in the
    :py:class:`flodym.ParameterDefinition`, and is set by
    :py:meth:`flodym.MFASystem.link_parameters` for parameters used as shares, dimension
    splitters or prescribed arrays.
    """
