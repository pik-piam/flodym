"""An uncertainty definition bound to the nominal values of parameters, and its random numbers.

Binding resolves the slice each entry of the definition applies to, casts the distribution
parameters to it and assigns each value the index of its draw. Sampled values are computed from
unit-interval numbers. By default, they come from one random stream per seed, parameter name,
entry and sample index (see :py:func:`unit_interval_numbers`); any other source with the
signature of :py:data:`UnitNumbers`, e.g. a design for a sensitivity analysis, can replace it.
"""

import difflib
import hashlib
import itertools
import logging
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import numpy as np

from ._bound_distribution import BoundDistribution, InvalidValuesError
from ._sub_array_handler import _SubArrayHandler
from .dimensions import Dimension, _free_letter
from .flodym_arrays import FlodymArray

if TYPE_CHECKING:
    from .uncertainty import ParameterUncertainty, UncertaintyDefinition

logger = logging.getLogger(__name__)

MAX_BLOCK_SIZE = 64
"""Largest number of samples whose values are computed together."""
MAX_BLOCK_VALUES = 2**22
"""Largest number of sampled values computed together, to limit the memory needed."""
MAX_LISTED_FAILURES = 3
"""Number of failing values listed in error messages. Fewer than the items of a dimension listed
by ``_sub_array_handler.MAX_LISTED_ITEMS``, since each is a tuple of items of all dimensions."""

UnitNumbers = Callable[["BoundEntry", int], np.ndarray]
"""Source of unit-interval numbers: given an entry and a sample index, it returns one number in
the open unit interval for each draw of the entry (see ``BoundEntry.n_draws``)."""


def unit_interval_numbers(
    seed: int, parameter_name: str, entry_index: int, sample_index: int, n_numbers: int
) -> np.ndarray:
    """Uniform numbers in the open unit interval for one entry and one sample.

    They are the first ``n_numbers`` numbers of a ``PCG64`` stream seeded by
    ``SeedSequence(seed, spawn_key=(*name_hash, entry_index, sample_index))``, with ``name_hash``
    the first 64 bits of the SHA-256 hash of the parameter name, as two 32-bit numbers. They are
    computed from the raw 64-bit output, which numpy keeps stable across versions and platforms,
    as ``((raw >> 11) + 0.5) * 2**-53``.
    """
    digest = hashlib.sha256(parameter_name.encode("utf-8")).digest()
    name_hash = (int.from_bytes(digest[:4], "big"), int.from_bytes(digest[4:8], "big"))
    seed_sequence = np.random.SeedSequence(seed, spawn_key=(*name_hash, entry_index, sample_index))
    raw = np.random.PCG64(seed_sequence).random_raw(n_numbers)
    return ((np.asarray(raw, dtype=np.uint64) >> np.uint64(11)).astype(np.float64) + 0.5) * (
        2.0**-53
    )


def random_unit_numbers(seed: int) -> UnitNumbers:
    """The default source of unit-interval numbers, see :py:func:`unit_interval_numbers`."""

    def numbers(entry: "BoundEntry", sample_index: int) -> np.ndarray:
        return unit_interval_numbers(
            seed, entry.parameter_name, entry.entry_index, sample_index, entry.n_draws
        )

    return numbers


def resolve_sample_indices(
    n_samples: int | None, sample_indices: Iterable[int] | None
) -> list[int]:
    """Sorted sample indices from either the number of samples or the indices.

    Raises:
        ValueError: If not exactly one is given, if there are no samples, or if indices are
            negative, not integers or repeated.
    """
    if (n_samples is None) == (sample_indices is None):
        raise ValueError("Give exactly one of n_samples and sample_indices.")
    if n_samples is not None:
        if not _is_integer(n_samples) or n_samples < 1:
            raise ValueError(f"n_samples must be a positive integer, got {n_samples!r}.")
        return list(range(int(n_samples)))
    assert sample_indices is not None
    indices = list(sample_indices)
    if not indices:
        raise ValueError("sample_indices is empty.")
    invalid = [index for index in indices if not _is_integer(index) or index < 0]
    if invalid:
        raise ValueError(f"Sample indices must be non-negative integers, got {invalid}.")
    if len(set(indices)) != len(indices):
        raise ValueError("sample_indices contains repeated indices.")
    return sorted(int(index) for index in indices)


def resolve_seed(seed: int | None) -> int:
    """The given seed, or one drawn from the operating system if it is None.

    Raises:
        ValueError: If the seed is not a non-negative integer.
    """
    if seed is None:
        seed = int(np.random.SeedSequence().entropy)
        logger.info(f"Drew the seed {seed} for the Monte Carlo samples.")
    if not _is_integer(seed) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}.")
    return int(seed)


def sample_dim(sample_indices: list[int], used_letters: Iterable[str]) -> Dimension:
    """The sample dimension, with a letter which is not used by the arrays."""
    return Dimension(
        name="Sample", letter=_free_letter(used_letters, "s"), items=sample_indices, dtype=int
    )


def check_parameters(parameters: Any) -> None:
    """Raise a TypeError if the parameters are not a mapping of names to FlodymArrays."""
    if not isinstance(parameters, Mapping) and isinstance(
        getattr(parameters, "parameters", None), Mapping
    ):
        raise TypeError(
            "Pass the parameters of the MFA system (mfa.parameters), not the MFA system itself."
        )
    if not isinstance(parameters, Mapping):
        raise TypeError(
            "parameters must be a mapping of parameter names to FlodymArrays, e.g. "
            f"mfa.parameters, but is a {type(parameters).__name__}."
        )
    not_arrays = [name for name, value in parameters.items() if not isinstance(value, FlodymArray)]
    if not_arrays:
        raise TypeError(f"Parameters {not_arrays} are not FlodymArrays.")


class ModelCheckError(ValueError):
    """A check of the model as a whole failed, independent of the sampled values.

    Raised e.g. by the model function of :py:func:`flodym.make_model_function`. Unlike other
    exceptions of the model, it stops :py:func:`flodym.run_monte_carlo` also with
    ``on_error="skip"``, since all samples would fail it.
    """


class BoundEntry:
    """One entry of the uncertainty of a parameter, bound to the parameter's nominal values."""

    def __init__(
        self,
        parameter_name: str,
        entry_index: int,
        entry: "ParameterUncertainty",
        parameter: FlodymArray,
        normalize_letters: tuple[str, ...],
    ) -> None:
        self.parameter_name = parameter_name
        self.entry_index = entry_index
        self.label = f"parameter '{parameter_name}'"
        if entry.where is not None:
            self.label += f" (where={entry.where})"
        self.dims = parameter.dims
        self.flat_ids = self._flat_ids(entry.where, parameter)
        """Indices of the values of the entry in the flattened values of the parameter."""
        self.draw_ids, self.n_draws = self._draw_ids(entry.independent_over, normalize_letters)
        """For each value of the entry, the index of its unit-interval number."""
        distribution = entry.distribution
        self.distribution_class = type(distribution)
        self.field_values = {
            name: self._field_values(name, getattr(distribution, name), parameter)
            for name in type(distribution).model_fields
        }
        """Fields of the distribution; numbers and arrays as arrays over the values."""
        nominal = parameter.values.ravel()[self.flat_ids].astype(float)
        try:
            self.distribution = BoundDistribution(distribution, nominal, self.field_values)
        except InvalidValuesError as error:
            raise ValueError(
                f"{type(distribution).__name__} of {self.label}{self.location(error.failed)}: "
                f"{error.message}."
            ) from None

    def values(self, unit_numbers: np.ndarray) -> np.ndarray:
        """Sampled values, by sample (axis 0) and value (axis 1) of the entry.

        Args:
            unit_numbers: Unit-interval numbers by sample (axis 0) and draw (axis 1).
        """
        return self.distribution.ppf(unit_numbers[:, self.draw_ids])

    def location(self, failed: np.ndarray) -> str:
        """Description of the items where a boolean array over the values is True."""
        if not self.dims.letters:
            return ""
        failed_ids = self.flat_ids[np.flatnonzero(failed)]
        items = []
        for flat_id in failed_ids[:MAX_LISTED_FAILURES]:
            ids = np.unravel_index(flat_id, self.dims.shape)
            items.append([str(dim.items[int(i)]) for dim, i in zip(self.dims, ids)])
        n_more = failed_ids.size - MAX_LISTED_FAILURES
        more = f" and {n_more} more" if n_more > 0 else ""
        if len(self.dims) == 1:
            return f" at {self.dims[0].name} = {', '.join(item[0] for item in items)}{more}"
        listed = ", ".join(f"({', '.join(item)})" for item in items)
        return f" at ({', '.join(self.dims.names)}) = {listed}{more}"

    def add_to_hash(self, hasher: Any) -> None:
        """Add everything which determines the sampled values, except the seed, to a hash."""
        _hash_value(hasher, self.parameter_name)
        _hash_value(hasher, self.entry_index)
        _hash_value(hasher, (self.distribution_class.__module__, self.distribution_class.__name__))
        _hash_value(hasher, self.flat_ids)
        _hash_value(hasher, self.draw_ids)
        for name in sorted(self.field_values):
            _hash_value(hasher, name)
            _hash_value(hasher, self.field_values[name])

    def _flat_ids(self, where: dict[str, Any] | None, parameter: FlodymArray) -> np.ndarray:
        all_ids = np.arange(parameter.size).reshape(parameter.shape)
        if where is None:
            return all_ids.ravel()
        unknown = [key for key in where if key not in parameter.dims]
        if unknown:
            raise ValueError(
                f"Uncertainty of {self.label}: unknown dimensions {unknown} in where. "
                f"Dimensions of the parameter: {list(parameter.dims.letters)}."
            )
        try:
            handler = _SubArrayHandler(parameter, where, allow_ancestor_items=True)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Uncertainty of {self.label}: {error}") from None
        flat_ids = np.asarray(all_ids[handler.ids]).ravel()
        if flat_ids.size == 0:
            raise ValueError(f"Uncertainty of {self.label}: where selects no values.")
        return flat_ids

    def _draw_ids(
        self, independent_over: tuple[str, ...] | str, normalize_letters: tuple[str, ...]
    ) -> tuple[np.ndarray, int]:
        """Index of the unit-interval number of each value, and the number of them."""
        n_values = self.flat_ids.size
        if independent_over == "all":
            return np.arange(n_values), n_values
        letters = tuple(independent_over) + tuple(
            letter for letter in normalize_letters if letter not in independent_over
        )
        if not letters:
            return np.zeros(n_values, dtype=int), 1
        ids_by_axis = np.unravel_index(self.flat_ids, self.dims.shape)
        coordinates = []
        for letter in letters:
            if letter in self.dims.letters:
                coordinates.append(ids_by_axis[self.dims.index(letter)])
                continue
            dim, ancestor = self._dim_below(letter)
            ancestor_ids = dim._indices_in_ancestor(ancestor)
            coordinates.append(ancestor_ids[ids_by_axis[self.dims.index(dim.letter)]])
        _, draw_ids = np.unique(np.stack(coordinates, axis=1), axis=0, return_inverse=True)
        draw_ids = np.asarray(draw_ids).reshape(-1)
        return draw_ids, int(draw_ids.max()) + 1

    def _dim_below(self, letter: str) -> tuple[Dimension, Dimension]:
        """The dimension of the parameter which descends from the dimension with the letter."""
        matches = [
            (dim, ancestor)
            for dim in self.dims
            for ancestor in dim.ancestors
            if ancestor.letter == letter
        ]
        if len(matches) != 1:
            reason = "unknown dimension" if not matches else "ancestor of several dimensions"
            raise ValueError(
                f"Uncertainty of {self.label}: {reason} '{letter}' in independent_over. Give "
                "letters of dimensions of the parameter or of their ancestors. Dimensions of "
                f"the parameter: {list(self.dims.letters)}."
            )
        return matches[0]

    def _field_values(self, field_name: str, value: Any, parameter: FlodymArray) -> Any:
        """Values of a field of the distribution for the values of the entry."""
        if isinstance(value, FlodymArray):
            for dim in value.dims:
                if dim.letter in parameter.dims.letters:
                    if dim.items != parameter.dims[dim.letter].items:
                        raise ValueError(
                            f"Uncertainty of {self.label}: {field_name} has other items of "
                            f"dimension '{dim.name}' than the parameter."
                        )
                elif not any(dim.letter in [a.letter for a in d.ancestors] for d in parameter.dims):
                    raise ValueError(
                        f"Uncertainty of {self.label}: {field_name} has the dimension "
                        f"'{dim.name}', which the parameter does not have. Dimensions of the "
                        f"parameter: {list(parameter.dims.letters)}."
                    )
            values = value.cast_values_to(parameter.dims).astype(float)
            return values.ravel()[self.flat_ids]
        if isinstance(value, Real) and not isinstance(value, bool):
            return np.full(self.flat_ids.size, float(value))
        return value


class BoundDefinition:
    """An uncertainty definition bound to the nominal values of parameters."""

    def __init__(
        self, definition: "UncertaintyDefinition", parameters: Mapping[str, FlodymArray]
    ) -> None:
        check_parameters(parameters)
        entries_by_name = definition._entries()
        unknown = [name for name in entries_by_name if name not in parameters]
        if unknown:
            hints = ""
            for name in unknown:
                matches = difflib.get_close_matches(name, list(parameters), n=1)
                if matches:
                    hints += f" Did you mean '{matches[0]}' instead of '{name}'?"
            raise ValueError(
                f"Uncertain parameters {unknown} are not among the given parameters.{hints} "
                f"Given parameters: {list(parameters)}."
            )
        not_set = [name for name in entries_by_name if not parameters[name].is_set]
        if not_set:
            raise ValueError(
                f"Uncertain parameters {not_set} are not set (is_set=False). Set their values "
                "before sampling."
            )
        self.nominal = {name: parameters[name].copy() for name in entries_by_name}
        """Copies of the uncertain parameters with their nominal values."""
        self.normalize_over = {
            name: self._normalize_letters(name, letters, parameters[name])
            for name, letters in definition._normalize_over_letters().items()
        }
        self.entries: dict[str, list[BoundEntry]] = {}
        for name, entries in entries_by_name.items():
            normalize_letters = self.normalize_over.get(name, ())
            self.entries[name] = [
                BoundEntry(name, index, entry, parameters[name], normalize_letters)
                for index, entry in enumerate(entries)
            ]
            self._check_no_overlap(name)
        self.nominal_sums = {
            name: self._nominal_sums(name, letters) for name, letters in self.normalize_over.items()
        }
        total_size = sum(array.size for array in self.nominal.values())
        self.block_size = max(1, min(MAX_BLOCK_SIZE, MAX_BLOCK_VALUES // max(total_size, 1)))

    def iter_samples(
        self, unit_numbers: UnitNumbers, sample_indices: Iterable[int]
    ) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
        """Yield the index and the sampled values of each sample, in the given order.

        The values of a sample are the same, whatever the other sample indices.
        """
        # Blocks are aligned to absolute sample indices and padded to their full size, so that a
        # sample is always computed at the same position of an array of the same shape: numpy
        # may compute element-wise functions differently, with other bits, for other shapes.
        for block, indices in itertools.groupby(sample_indices, key=self._block_of):
            indices = list(indices)
            start = block * self.block_size
            block_indices: list[int | None] = [None] * self.block_size
            for index in indices:
                block_indices[index - start] = index
            values = self._block_values(unit_numbers, block_indices)
            for index in indices:
                row = index - start
                yield index, {name: self._normalized(name, v[row]) for name, v in values.items()}

    def fingerprint(self) -> str:
        """SHA-256 hash of everything which determines the sampled values, except the seed."""
        hasher = hashlib.sha256()
        for name in sorted(self.entries):
            for entry in self.entries[name]:
                entry.add_to_hash(hasher)
            _hash_value(hasher, self.normalize_over.get(name))
        return hasher.hexdigest()

    def _block_of(self, sample_index: int) -> int:
        return sample_index // self.block_size

    def _block_values(
        self, unit_numbers: UnitNumbers, block_indices: Sequence[int | None]
    ) -> dict[str, np.ndarray]:
        """Sampled values of each parameter, by sample (axis 0) and flattened value (axis 1).

        Samples whose index is None get the unit-interval number 0.5; they only fill the block.
        """
        values = {}
        for name, array in self.nominal.items():
            parameter_values = np.tile(array.values.ravel().astype(float), (len(block_indices), 1))
            for entry in self.entries[name]:
                numbers = np.full((len(block_indices), entry.n_draws), 0.5)
                for row, sample_index in enumerate(block_indices):
                    if sample_index is not None:
                        numbers[row] = unit_numbers(entry, sample_index)
                parameter_values[:, entry.flat_ids] = entry.values(numbers)
            values[name] = parameter_values
        return values

    def _normalized(self, name: str, flat_values: np.ndarray) -> np.ndarray:
        """Values of one sample with the shape of the parameter, normalized if requested."""
        values = flat_values.reshape(self.nominal[name].shape)
        if name not in self.normalize_over:
            return values
        axes = tuple(self.nominal[name].dims.index(letter) for letter in self.normalize_over[name])
        sums = values.sum(axis=axes, keepdims=True)
        scale = np.divide(self.nominal_sums[name], sums, out=np.ones_like(sums), where=sums != 0)
        return values * scale

    def _normalize_letters(
        self, name: str, letters: tuple[str, ...], parameter: FlodymArray
    ) -> tuple[str, ...]:
        unknown = [letter for letter in letters if letter not in parameter.dims.letters]
        if unknown:
            raise ValueError(
                f"normalize_over of parameter '{name}': unknown dimensions {unknown}. "
                f"Dimensions of the parameter: {list(parameter.dims.letters)}."
            )
        return letters

    def _nominal_sums(self, name: str, letters: tuple[str, ...]) -> np.ndarray:
        """Sums of the nominal values over the dimensions to normalize over.

        Raises:
            ValueError: If a sum is zero, but has uncertain values.
        """
        array = self.nominal[name]
        axes = tuple(array.dims.index(letter) for letter in letters)
        sums = array.values.astype(float).sum(axis=axes, keepdims=True)
        uncertain = np.zeros(array.size, dtype=bool)
        for entry in self.entries[name]:
            uncertain[entry.flat_ids] = True
        uncertain = uncertain.reshape(array.shape).any(axis=axes, keepdims=True)
        zero = (sums == 0) & uncertain
        if np.any(zero):
            kept = [dim for dim in array.dims if dim.letter not in letters]
            groups = []
            for ids in np.argwhere(zero.squeeze(axis=axes) if axes else zero)[:MAX_LISTED_FAILURES]:
                groups.append(", ".join(str(dim.items[int(i)]) for dim, i in zip(kept, ids)))
            raise ValueError(
                f"normalize_over of parameter '{name}': the nominal values sum to zero over "
                f"{list(letters)}, so they cannot be normalized, at "
                f"({', '.join(dim.name for dim in kept)}) = "
                f"{', '.join(f'({group})' for group in groups)}."
            )
        return sums

    def _check_no_overlap(self, name: str) -> None:
        entries = self.entries[name]
        for first, second in itertools.combinations(entries, 2):
            if np.intersect1d(first.flat_ids, second.flat_ids).size:
                raise ValueError(
                    f"Uncertainty of parameter '{name}': entries {first.entry_index} and "
                    f"{second.entry_index} apply to the same values. Entries of a parameter must "
                    "not overlap."
                )


def parameters_fingerprint(parameters: Mapping[str, FlodymArray]) -> str:
    """SHA-256 hash of the names, dimensions and values of all parameters."""
    hasher = hashlib.sha256()
    for name in sorted(parameters):
        _hash_value(hasher, name)
        _hash_value(hasher, parameters[name])
    return hasher.hexdigest()


def _hash_value(hasher: Any, value: Any) -> None:
    """Add a value to a hash; arrays by their bytes, so that they are never abbreviated."""
    if isinstance(value, FlodymArray):
        for dim in value.dims:
            _hash_value(hasher, (dim.letter, dim.name))
            _hash_value(hasher, np.asarray([str(item) for item in dim.items]))
        value = value.values
    if isinstance(value, np.ndarray):
        values = np.ascontiguousarray(value)
        data = str((values.dtype.str, values.shape)).encode() + values.tobytes()
    else:
        data = repr(value).encode()
    hasher.update(len(data).to_bytes(8, "big"))
    hasher.update(data)


def _is_integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)
