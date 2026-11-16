"""Declaration of uncertain parameters, and arrays of sampled values.

See :py:class:`flodym.UncertaintyDefinition` and :py:func:`flodym.run_monte_carlo`.
"""

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any, cast

import numpy as np
from pydantic import field_validator, model_validator
from typing_extensions import Self

from ._base_model import FlodymBaseModel
from ._sampling import (
    BoundDefinition,
    random_unit_numbers,
    resolve_sample_indices,
    resolve_seed,
    sample_dim,
)
from .dimensions import Dimension, DimensionSet, _free_letter
from .distributions import Distribution
from .flodym_arrays import FlodymArray


class ParameterUncertainty(FlodymBaseModel):
    """The uncertainty of a parameter, or of a slice of it.

    **Example**

        >>> import flodym as fd
        >>> copper_yield = fd.ParameterUncertainty(
        ...     distribution=fd.UniformDistribution(lower=0.01, upper=0.05), where={"e": "Cu"}
        ... )
        >>> yield_by_region = fd.ParameterUncertainty(
        ...     distribution=fd.NormalDistribution(relative_std=0.05, lower=0.0, upper=1.0),
        ...     independent_over="r",
        ... )
    """

    distribution: Distribution
    """Distribution of the values around their nominal values."""
    where: dict[str, Any] | None = None
    """The slice of the parameter the distribution applies to, as in array slicing (see
    :py:class:`flodym.FlodymArray`), e.g. ``{"e": "Cu"}`` or ``{"r": ["EUR", "USA"]}``. Items of
    ancestors of a dimension address the items belonging to them, e.g. ``{"c": "EUR"}`` for the
    European countries. If None, the whole parameter."""
    independent_over: str | Sequence[str] = ()
    """Letters of the dimensions along which values are drawn independently, as a tuple or a
    single letter; stored as a tuple. By default, all values share one draw, i.e. they are all at
    the same quantile of their distributions: a time series is higher or lower as a whole. With
    ``"r"``, each region gets its own draw. A letter can also be the one of an ancestor of a
    dimension of the parameter: With countries ``c`` below world regions ``w``, ``"w"`` draws
    once per world region for all its countries. ``"all"`` gives each value its own draw.

    Independent draws for many values average out in sums and cumulative stocks, so they give a
    smaller uncertainty of these than shared draws."""

    @field_validator("independent_over", mode="before")
    @classmethod
    def _letters_to_tuple(cls, value: Any) -> Any:
        """Convert a single dimension letter or a sequence of them to a tuple."""
        return _to_letters(value, allow_all=True)


class SampledArray(FlodymBaseModel):
    """The samples of an array, e.g. of an output of :py:func:`flodym.run_monte_carlo`.

    Statistics over the samples are computed separately for each value of the array. Standard
    deviations and quantiles of a sum or a ratio of arrays are not the sums or ratios of the
    standard deviations and quantiles. Compute such quantities per sample instead: with
    :py:meth:`sum_over`, by combining the samples of outputs with the same sample dimension, e.g.
    ``fd.SampledArray(samples=copper.samples / total.samples)``, or as a derived quantity of the
    model (see :py:func:`flodym.make_model_function`).

    The arrays returned by :py:attr:`mean`, :py:attr:`std` and :py:meth:`quantile` keep the
    extensivity (see :py:attr:`flodym.FlodymArray.extensive`) of the samples, so flodym does not
    prevent summing them over dimensions.

    **Example**

        >>> steel = result.outputs["remelting => sysenv"]
        >>> steel.mean, steel.std, steel.quantile([0.05, 0.5, 0.95])
        >>> steel[{"e": "Fe"}].quantile(0.95)
        >>> steel.sum_over(("e",)).quantile(0.95)
    """

    samples: FlodymArray
    """The samples. Its last dimension is the sample dimension, the others are the dimensions of
    the array.

    Do not combine it with arrays which do not have the sample dimension: flodym sums an
    operand over dimensions which the other operand and the result do not have, so the samples
    would be summed."""

    @model_validator(mode="after")
    def _check_sample_dim(self) -> Self:
        """Check that the samples have a sample dimension."""
        if not self.samples.dims.letters:
            raise ValueError("samples must have a sample dimension as last dimension.")
        return self

    def __getitem__(self, keys: dict[str, Any] | Hashable) -> "SampledArray":
        """Samples of a slice of the array, as in slicing a FlodymArray, e.g. ``[{"e": "Fe"}]``.

        As for a FlodymArray, several items of a dimension are selected by a
        :py:class:`flodym.Dimension` with these items, e.g. ``[{"r": europe}]``. The sample
        dimension cannot be sliced; select samples from :py:attr:`samples` instead.
        """
        sample_keys = (self.sample_dim.letter, self.sample_dim.name)
        if isinstance(keys, dict):
            if any(key in sample_keys for key in keys):
                raise ValueError(
                    "The sample dimension cannot be sliced. Select samples from the FlodymArray "
                    "in SampledArray.samples instead."
                )
            several = [
                key
                for key, value in keys.items()
                if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 1
            ]
            if several:
                raise ValueError(
                    f"Several items are given for the dimensions {several}. To select several "
                    "items, give a new Dimension with a new letter and these items instead of a "
                    "list, as for a FlodymArray."
                )
        sliced = self.samples[keys]
        if sliced.dims.letters[-1:] != (self.sample_dim.letter,):
            raise ValueError(
                f"{keys!r} slices the sample dimension, which is not possible. Give the "
                "dimension of the items as a dictionary key."
            )
        return SampledArray(samples=sliced)

    def __str__(self) -> str:
        return (
            f"SampledArray '{self.samples.name}' with dims ({','.join(self.dims.letters)}) and "
            f"shape {self.dims.shape}; {self.n_samples} samples"
        )

    @classmethod
    def _from_values(
        cls,
        name: str | None,
        dims: DimensionSet,
        values: np.ndarray,
        degree: int | None,
        sample_dim: Dimension,
    ) -> Self:
        """Samples from values with the sample axis last, keeping an extensivity degree."""
        samples = FlodymArray(dims=dims.append(sample_dim), values=values, name=name)
        return cls(samples=samples._with_degree(degree))

    @property
    def sample_dim(self) -> Dimension:
        """The sample dimension, the last dimension of :py:attr:`samples`."""
        return self.samples.dims[-1]

    @property
    def dims(self) -> DimensionSet:
        """The dimensions of the array, without the sample dimension."""
        return self.samples.dims.drop(self.sample_dim.letter)

    @property
    def n_samples(self) -> int:
        """The number of samples."""
        return self.sample_dim.len

    @property
    def mean(self) -> FlodymArray:
        """Mean over the samples."""
        return self._reduced(self.samples.values.mean(axis=-1))

    @property
    def std(self) -> FlodymArray:
        """Sample standard deviation over the samples (ddof=1); NaN for a single sample."""
        if self.n_samples < 2:
            return self._reduced(np.full(self.dims.shape, np.nan))
        return self._reduced(self.samples.values.std(axis=-1, ddof=1))

    def quantile(self, q: float | Sequence[float]) -> FlodymArray:
        """Quantiles over the samples, e.g. 0.95 for the 95th percentile.

        Uses the default (linear) method of ``numpy.quantile``.

        Args:
            q: A quantile, or several. Each must be between 0 and 1.

        Returns:
            For a single quantile, an array with the dimensions of the array. For several, an
            array with the additional last dimension ``Quantile``, whose items are the quantiles,
            e.g. for ``quantile([0.05, 0.5, 0.95]).to_df(dim_to_columns="Quantile")``.

        Raises:
            ValueError: If a quantile is not between 0 and 1.
        """
        quantiles = np.asarray(q, dtype=float)
        if not np.all((quantiles >= 0) & (quantiles <= 1)):
            raise ValueError(f"Quantiles must be between 0 and 1, got {q}.")
        values = np.quantile(self.samples.values, quantiles, axis=-1)
        if quantiles.ndim == 0:
            return self._reduced(values)
        quantile_dim = Dimension(
            name="Quantile",
            letter=_free_letter(self.dims.letters, "q"),
            items=[float(value) for value in quantiles],
            dtype=float,
        )
        result = FlodymArray(
            dims=self.dims.append(quantile_dim),
            values=np.moveaxis(values, 0, -1),
            name=self.samples.name,
        )
        return result._with_degree(self.samples._degree)

    def sum_over(
        self, sum_over_dims: tuple[str, ...], *, check_extensivity: bool = True
    ) -> "SampledArray":
        """Samples of the array summed over dimensions, for each sample.

        See :py:meth:`flodym.FlodymArray.sum_over`.

        Args:
            sum_over_dims: Letters or names of the dimensions to sum over. The sample dimension
                cannot be summed over.
            check_extensivity: Whether to report summing an array which is not extensive.

        Raises:
            ValueError: If the sample dimension is given.
        """
        if any(dim in (self.sample_dim.letter, self.sample_dim.name) for dim in sum_over_dims):
            raise ValueError("The sample dimension cannot be summed over.")
        summed = self.samples.sum_over(sum_over_dims, check_extensivity=check_extensivity)
        return SampledArray(samples=summed)

    def _reduced(self, values: np.ndarray) -> FlodymArray:
        """An array with the dimensions of the array, e.g. a statistic over the samples."""
        result = FlodymArray(dims=self.dims, values=np.asarray(values), name=self.samples.name)
        return result._with_degree(self.samples._degree)


class UncertaintyDefinition(FlodymBaseModel):
    """Which parameters are uncertain, and how.

    Parameters are given by name. The values a parameter has in the model are its nominal
    values. A :py:class:`flodym.Distribution` adds the spread around them, e.g. a relative
    standard deviation, so that one definition covers all values of a parameter, and changes of
    the data move the distributions with it. Parameters which are not listed are certain.

    By default, all values of an entry share one draw (see
    :py:attr:`flodym.ParameterUncertainty.independent_over`).

    **Example**

        >>> import flodym as fd
        >>> uncertainty = fd.UncertaintyDefinition(
        ...     parameters={
        ...         "eol machines": fd.LogNormalDistribution(relative_std=0.2),
        ...         "demolition yield": fd.ParameterUncertainty(
        ...             distribution=fd.UniformDistribution(lower=0.01, upper=0.05),
        ...             where={"e": "Cu"},
        ...         ),
        ...         "composition eol machines": fd.TriangularDistribution(
        ...             relative_lower=0.8, relative_upper=1.2
        ...         ),
        ...     },
        ...     normalize_over={"composition eol machines": "e"},
        ... )

    The definition is checked against the parameters when it is used, by
    :py:func:`flodym.run_monte_carlo` or :py:meth:`sample`. A ``ValueError`` is raised for
    unknown parameter names, parameters which are not set, unknown dimensions or items in
    ``where``, ``independent_over``, ``normalize_over`` or in distribution parameters given as
    arrays, overlapping entries of a parameter, a nominal sum of zero under ``normalize_over``,
    and invalid distribution parameters (e.g. a negative standard deviation, a lower bound above
    the upper one, or NaN values).
    """

    parameters: Mapping[
        str, Distribution | ParameterUncertainty | Sequence[Distribution | ParameterUncertainty]
    ]
    """Uncertainty of each uncertain parameter, by parameter name. Each value can be given as a
    :py:class:`flodym.Distribution` for the whole parameter, as a
    :py:class:`flodym.ParameterUncertainty`, or as a list of them for slices of the parameter,
    which must not overlap. Stored as lists of :py:class:`flodym.ParameterUncertainty`."""
    normalize_over: Mapping[str, str | Sequence[str]] = {}
    """Letters of dimensions, by parameter name, over which the sampled values of a parameter
    are rescaled such that their sum keeps its nominal value, e.g. one for shares. Given as a
    tuple or a single letter; stored as a tuple. Values are always drawn independently along
    these dimensions, since the rescaling would cancel a common draw. Bounds apply before the
    rescaling, so rescaled values can lie outside them, e.g. a share above one if other shares
    are sampled below their nominal values."""

    @field_validator("parameters", mode="before")
    @classmethod
    def _to_lists_of_uncertainties(cls, value: Any) -> Any:
        """Convert distributions and single uncertainties to lists of uncertainties."""
        if not isinstance(value, Mapping):
            return value
        return {name: _to_uncertainty_list(entries) for name, entries in value.items()}

    @field_validator("normalize_over", mode="before")
    @classmethod
    def _letters_to_tuples(cls, value: Any) -> Any:
        """Convert single dimension letters or sequences of them to tuples."""
        if not isinstance(value, Mapping):
            return value
        return {name: _to_letters(letters, allow_all=False) for name, letters in value.items()}

    @model_validator(mode="after")
    def _check_normalized_are_uncertain(self) -> Self:
        """Check that each parameter in normalize_over is uncertain."""
        unknown = [name for name in self.normalize_over if name not in self.parameters]
        if unknown:
            raise ValueError(
                f"Parameters {unknown} in normalize_over have no uncertainty in parameters. Add "
                "them to parameters."
            )
        return self

    def sample(
        self,
        parameters: Mapping[str, FlodymArray],
        n_samples: int | None = None,
        *,
        seed: int,
        sample_indices: Iterable[int] | None = None,
    ) -> dict[str, SampledArray]:
        """Sampled values of the uncertain parameters.

        They are exactly the values :py:func:`flodym.run_monte_carlo` uses for the same
        parameters and seed, also if only some of the samples are computed. Use them to look at
        the inputs, or to rerun a single sample by hand.

        Args:
            parameters: The parameters with their nominal values, by name. Parameters without
                uncertainty are ignored.
            n_samples: Number of samples, for the sample indices 0 to ``n_samples - 1``.
            seed: Seed of the random numbers, a non-negative integer, e.g. the seed of a
                :py:class:`flodym.MonteCarloResult`.
            sample_indices: Indices of the samples, instead of ``n_samples``.

        Returns:
            The samples of each uncertain parameter, by name. The items of the sample
            dimension are the sample indices.

        Raises:
            ValueError: If not exactly one of ``n_samples`` and ``sample_indices`` is given, if
                ``seed`` is None, or if the definition does not match the parameters (see the
                class docstring).
            TypeError: If the parameters are not a mapping of names to FlodymArrays.
        """
        indices = resolve_sample_indices(n_samples, sample_indices)
        bound = BoundDefinition(self, parameters)
        if seed is None:
            raise ValueError("Give a seed, e.g. the seed of a MonteCarloResult.")
        unit_numbers = random_unit_numbers(resolve_seed(seed))
        values = {
            name: np.empty(array.shape + (len(indices),)) for name, array in bound.nominal.items()
        }
        for position, (_, sampled) in enumerate(bound.iter_samples(unit_numbers, indices)):
            for name, sample_values in sampled.items():
                values[name][..., position] = sample_values
        used_letters = {letter for array in bound.nominal.values() for letter in array.dims.letters}
        dim = sample_dim(indices, used_letters)
        return {
            name: SampledArray._from_values(name, array.dims, values[name], array._degree, dim)
            for name, array in bound.nominal.items()
        }

    def _entries(self) -> dict[str, list[ParameterUncertainty]]:
        """The entries of each uncertain parameter, as stored by the validator."""
        return cast("dict[str, list[ParameterUncertainty]]", self.parameters)

    def _normalize_over_letters(self) -> dict[str, tuple[str, ...]]:
        """The dimension letters of normalize_over, as stored by the validator."""
        return cast("dict[str, tuple[str, ...]]", self.normalize_over)


def _to_uncertainty_list(entries: Any) -> Any:
    """Convert a distribution or a single uncertainty (or a list of them) to a list."""
    if isinstance(entries, (Distribution, ParameterUncertainty)):
        entries = [entries]
    if not isinstance(entries, (list, tuple)):
        return entries
    return [
        ParameterUncertainty(distribution=entry) if isinstance(entry, Distribution) else entry
        for entry in entries
    ]


def _to_letters(value: Any, allow_all: bool) -> Any:
    """Convert a single dimension letter or a sequence of them to a tuple."""
    if isinstance(value, str):
        return value if allow_all and value == "all" else (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return value
