"""Monte Carlo propagation of parameter uncertainty through a model.

See :py:func:`flodym.run_monte_carlo`. :py:func:`flodym.make_model_function` turns an MFA system
into a model.
"""

import logging
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from numbers import Real
from typing import Any, Literal, NamedTuple

import numpy as np
from typing_extensions import Self

from ._attribute_dict import AttributeDict
from ._base_model import FlodymBaseModel
from ._checks import config_hint
from ._config import config, handle_error
from ._sampling import (
    BoundDefinition,
    ModelCheckError,
    parameters_fingerprint,
    random_unit_numbers,
    resolve_sample_indices,
    resolve_seed,
    sample_dim,
)
from .dimensions import Dimension, DimensionSet
from .flodym_arrays import FlodymArray
from .uncertainty import SampledArray, UncertaintyDefinition

logger = logging.getLogger(__name__)

_REJECTION_WARNING_RATE = 0.01
"""Rejection rate above which :py:func:`run_monte_carlo` reports the rejection."""
_MAX_LISTED_OUTPUTS = 5
"""Number of outputs listed in error messages."""
_GIB = 2**30


class MonteCarloResult(FlodymBaseModel):
    """Result of :py:func:`flodym.run_monte_carlo`: the samples of all outputs of the model.

    **Example**

        >>> result = fd.run_monte_carlo(model, parameters, uncertainty, n_samples=500, seed=1)
        >>> stock = result.outputs["stock"]
        >>> stock.mean, stock.std, stock.quantile([0.05, 0.5, 0.95])

    Results of runs with the same model, parameters, definition and seed, but different
    ``sample_indices``, can be joined with :py:meth:`concat`.
    """

    outputs: AttributeDict[SampledArray]
    """Samples of each output of the model, by name. Also accessible as attributes, see
    :py:class:`flodym.AttributeDict`."""
    seed: int
    """Seed of the random numbers. Pass it to :py:func:`flodym.run_monte_carlo` to reproduce the
    result."""
    sample_indices: list[int]
    """Indices of the requested samples, in ascending order."""
    failed_samples: dict[int, str] = {}
    """Error messages of the samples left out with ``on_error="skip"``, by sample index."""
    sample_dim: Dimension
    """The sample dimension, the last dimension of the samples of each output. Its items are the
    indices of the successful samples."""
    definition_fingerprint: str
    """Hash of everything in the uncertainty definition which determines the sampled values,
    compared by :py:meth:`concat`."""
    parameters_fingerprint: str
    """Hash of the names, dimensions and nominal values of all parameters, compared by
    :py:meth:`concat`."""

    def __str__(self) -> str:
        names = list(self.outputs)
        listed = ", ".join(f"'{name}'" for name in names[:3]) + (", ..." if len(names) > 3 else "")
        return (
            f"MonteCarloResult with {len(names)} outputs ({listed}); {self.sample_dim.len} of "
            f"{len(self.sample_indices)} samples successful (rejection rate "
            f"{self.rejection_rate:.1%}); seed {self.seed}"
        )

    @classmethod
    def concat(cls, results: Iterable["MonteCarloResult"]) -> Self:
        """Join results of runs with different sample indices into one result.

        This allows to split a run, e.g. over the jobs of a job array:

            >>> parts = [
            ...     fd.run_monte_carlo(
            ...         model,
            ...         parameters,
            ...         uncertainty,
            ...         sample_indices=range(start, start + 250),
            ...         seed=42,
            ...     )
            ...     for start in (0, 250, 500, 750)
            ... ]
            >>> result = fd.MonteCarloResult.concat(parts)

        The joined result equals the result of a single run with all sample indices. Samples
        are ordered by index.

        Args:
            results: The results to join. They must have the same seed, uncertainty definition,
                nominal parameter values and outputs, and different sample indices.

        Returns:
            The joined result.

        Raises:
            ValueError: If no results are given, if their seeds, definitions, nominal values or
                outputs differ, or if sample indices are requested in several results.
        """
        results = list(results)
        if not results:
            raise ValueError("No results to concatenate.")
        first = results[0]
        for position, result in enumerate(results[1:], start=1):
            _check_same_run(first, result, position)
        requested = [index for result in results for index in result.sample_indices]
        repeated = sorted(index for index, count in Counter(requested).items() if count > 1)
        if repeated:
            raise ValueError(
                f"Sample indices {repeated} are requested in several results. Give each part its "
                "own sample_indices."
            )
        successful = [index for result in results for index in result.sample_dim.items]
        order = np.argsort(successful, kind="stable")
        joined_dim = first.sample_dim.model_copy(update={"items": [successful[i] for i in order]})
        outputs = AttributeDict[SampledArray]()
        for name, output in first.outputs.items():
            values = np.concatenate(
                [result.outputs[name].samples.values for result in results], axis=-1
            )[..., order]
            outputs[name] = SampledArray._from_values(
                name, output.dims, values, output.samples._degree, joined_dim
            )
        failed = {
            index: message for result in results for index, message in result.failed_samples.items()
        }
        return cls(
            outputs=outputs,
            seed=first.seed,
            sample_indices=sorted(requested),
            failed_samples=dict(sorted(failed.items())),
            sample_dim=joined_dim,
            definition_fingerprint=first.definition_fingerprint,
            parameters_fingerprint=first.parameters_fingerprint,
        )

    @property
    def rejection_rate(self) -> float:
        """Share of the requested samples which failed and were left out."""
        return len(self.failed_samples) / len(self.sample_indices)


def run_monte_carlo(
    model: Callable[[dict[str, FlodymArray]], Mapping[str, FlodymArray | float]],
    parameters: Mapping[str, FlodymArray],
    uncertainty: UncertaintyDefinition,
    n_samples: int | None = None,
    *,
    seed: int | None = None,
    sample_indices: Iterable[int] | None = None,
    on_error: Literal["raise", "skip"] = "raise",
    max_memory_gib: float = 2.0,
) -> MonteCarloResult:
    """Propagate the uncertainty of parameters through a model by Monte Carlo sampling.

    ``model`` is any function of the parameters which returns the quantities of interest, as
    ``model(parameters) -> {name: FlodymArray or number}``. For an :py:class:`flodym.MFASystem`,
    create it with :py:func:`flodym.make_model_function`. For every sample, it is called with
    copies of all parameters, in which the uncertain ones have sampled values (see
    :py:meth:`flodym.UncertaintyDefinition.sample`). The model owns the copies: it may modify
    them, add derived parameters or rebind names without affecting the next sample.

    Samples are run one after another. To split a run, e.g. over the jobs of a job array, give
    each part its ``sample_indices`` and the same seed, and join the results with
    :py:meth:`flodym.MonteCarloResult.concat`.

    **Checks**

    - The first sample is run twice; the outputs must be identical. Otherwise, the model
      carries state from one run to the next, e.g. through a parameter it keeps and modifies in
      place, or it uses random numbers of its own, and the samples would not follow the declared
      distributions.
    - The samples of all outputs are stored. Their size is known after the first sample; above
      ``max_memory_gib``, an error lists the largest outputs. Return fewer or aggregated outputs
      in that case.
    - If no output differs between the samples, this is reported according to
      ``flodym.config.error_behaviors.monte_carlo_no_variation`` (default: a warning).
    - With ``on_error="skip"``, failing samples are left out, which is rejection sampling: The
      sampled distribution then differs from the declared one. The rejection rate is given in
      the result. A rate above 1 % is reported according to
      ``flodym.config.error_behaviors.monte_carlo_rejection`` (default: a warning).

    **Reproducibility**

    The random numbers depend only on the seed, the parameter name, the entry of its
    uncertainty and the sample index. Adding, removing or reordering other uncertain parameters
    does not change them, and a sample does not depend on which other samples are run. The
    unit-interval numbers are the same on all platforms and numpy versions; sampled values and
    results agree up to floating-point differences of scipy and numpy functions.

    **Example**

        >>> import flodym as fd
        >>> def stock_model(parameters):
        ...     lifetime = fd.NormalLifetime(
        ...         dims=dims, mean=parameters["lifetime"], std=0.3 * parameters["lifetime"]
        ...     )
        ...     stock = fd.InflowDrivenDSM(
        ...         dims=dims,
        ...         inflow=parameters["inflow"],
        ...         lifetime_model=lifetime,
        ...         cohort_dim=cohort_dim,
        ...     )
        ...     stock.compute()
        ...     return {"stock": stock.stock, "outflow": stock.outflow}
        >>> uncertainty = fd.UncertaintyDefinition(
        ...     parameters={"lifetime": fd.NormalDistribution(relative_std=0.15, lower=1.0)}
        ... )
        >>> result = fd.run_monte_carlo(
        ...     stock_model,
        ...     {"inflow": inflow, "lifetime": lifetime_mean},
        ...     uncertainty,
        ...     n_samples=500,
        ...     seed=1,
        ... )
        >>> result.outputs["stock"].quantile([0.05, 0.95])

    Args:
        model: Function of the parameters, by name, returning the outputs, by name. Outputs
            are FlodymArrays or numbers, and must have the same names and dimensions for all
            samples.
        parameters: All parameters of the model, by name, with their nominal values, e.g.
            ``mfa.parameters``.
        uncertainty: Which parameters are uncertain, and how.
        n_samples: Number of samples, for the sample indices 0 to ``n_samples - 1``.
        seed: Seed of the random numbers, a non-negative integer. If None, a seed is drawn from
            the operating system and stored in the result.
        sample_indices: Indices of the samples to run, instead of ``n_samples``.
        on_error: What to do if the model raises an exception for a sample: ``"raise"`` it, or
            ``"skip"`` the sample.
        max_memory_gib: Largest memory, in GiB, for the samples of all outputs.

    Returns:
        The samples of all outputs.

    Raises:
        ValueError: Before the loop: if not exactly one of ``n_samples`` and ``sample_indices``
            is given or there are no samples, if the definition does not match the parameters
            (see :py:class:`flodym.UncertaintyDefinition`), if the two runs of the first sample
            differ, or if the samples need more than ``max_memory_gib``. In the loop: if a
            sample returns other names or dimensions than the first one.
        TypeError: If ``model``, ``parameters`` or ``uncertainty`` have the wrong type, or if the
            model returns something other than a mapping of names to FlodymArrays or numbers.
        Exception: The exception of the model, if a sample fails with ``on_error="raise"``, or
            the one of the first sample if all samples fail with ``on_error="skip"``. The index
            of the sample and the seed are logged at level ERROR, and added to the exception as
            a note on Python 3.11 and later.
    """
    if not callable(model):
        raise TypeError(
            f"model must be a function of the parameters, but is a {type(model).__name__}. For "
            "an MFASystem, use flodym.make_model_function(mfa)."
        )
    if not isinstance(uncertainty, UncertaintyDefinition):
        raise TypeError(
            "uncertainty must be a flodym.UncertaintyDefinition, but is a "
            f"{type(uncertainty).__name__}. Use fd.UncertaintyDefinition(parameters=...)."
        )
    if on_error not in ("raise", "skip"):
        raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}.")
    if not max_memory_gib > 0:
        raise ValueError(f"max_memory_gib must be positive, got {max_memory_gib!r}.")
    indices = resolve_sample_indices(n_samples, sample_indices)
    seed = resolve_seed(seed)
    bound = BoundDefinition(uncertainty, parameters)
    nominal_fingerprint = parameters_fingerprint(parameters)
    run = _SampleRun(model, parameters, seed, on_error, max_memory_gib, len(indices))
    run.run(bound.iter_samples(random_unit_numbers(seed), indices))
    run.report(has_uncertainty=bool(uncertainty.parameters))
    dim = sample_dim(run.successful, run.used_letters)
    outputs = AttributeDict[SampledArray]()
    for name, layout in run.layout.items():
        values = run.storage[name][..., : len(run.successful)]
        outputs[name] = SampledArray._from_values(name, layout.dims, values, layout.degree, dim)
    return MonteCarloResult(
        outputs=outputs,
        seed=seed,
        sample_indices=indices,
        failed_samples=run.failed,
        sample_dim=dim,
        definition_fingerprint=bound.fingerprint(),
        parameters_fingerprint=nominal_fingerprint,
    )


class _Output(NamedTuple):
    """An output of the model for one sample, converted by :py:func:`_convert_outputs`."""

    dims: DimensionSet
    values: np.ndarray
    degree: int | None
    """Extensivity degree, see :py:attr:`flodym.FlodymArray.extensive`."""


class _Layout(NamedTuple):
    """Dimensions and extensivity degree of an output, from the first successful sample."""

    dims: DimensionSet
    degree: int | None


class _SampleRun:
    """Runs the model for each sample and stores the outputs."""

    def __init__(
        self,
        model: Callable[[dict[str, FlodymArray]], Mapping[str, FlodymArray | float]],
        parameters: Mapping[str, FlodymArray],
        seed: int,
        on_error: str,
        max_memory_gib: float,
        n_samples: int,
    ) -> None:
        self.model = model
        self.parameters = {name: parameter.copy() for name, parameter in parameters.items()}
        """Copies of the nominal parameters, which the model cannot modify."""
        self.seed = seed
        self.on_error = on_error
        self.max_memory_gib = max_memory_gib
        self.n_samples = n_samples
        self.layout: dict[str, _Layout] = {}
        """Dimensions and extensivity degree of each output, from the first successful sample."""
        self.storage: dict[str, np.ndarray] = {}
        """Values of each output, with the samples along the last axis."""
        self.successful: list[int] = []
        self.failed: dict[int, str] = {}
        self.first_failure: tuple[int, Exception] | None = None

    @property
    def used_letters(self) -> set[str]:
        """Dimension letters of all outputs."""
        return {letter for layout in self.layout.values() for letter in layout.dims.letters}

    def run(self, samples: Iterable[tuple[int, dict[str, np.ndarray]]]) -> None:
        """Run the model for each sample index and its sampled parameter values.

        Raises:
            Exception: The exception of the model, see :py:func:`run_monte_carlo`.
        """
        for sample_index, sampled in samples:
            outputs = self._call(sample_index, sampled)
            if outputs is None:
                continue
            if not self.successful:
                self.layout = {name: _Layout(o.dims, o.degree) for name, o in outputs.items()}
                self._check_memory()
                self._check_deterministic(sample_index, sampled, outputs)
                self.storage = {
                    name: np.empty(layout.dims.shape + (self.n_samples,))
                    for name, layout in self.layout.items()
                }
            else:
                self._check_layout(outputs, sample_index)
            for name, output in outputs.items():
                self.storage[name][..., len(self.successful)] = output.values
            self.successful.append(sample_index)
        if not self.successful:
            assert self.first_failure is not None
            first_index, error = self.first_failure
            _report_failure(
                error,
                f"All {self.n_samples} Monte Carlo samples failed and were skipped "
                f"(seed={self.seed}). This is the error of the first one, sample {first_index}.",
            )
            raise error

    def report(self, has_uncertainty: bool) -> None:
        """Report a high rejection rate, and outputs which do not vary."""
        rejection_rate = len(self.failed) / self.n_samples
        if rejection_rate > _REJECTION_WARNING_RATE:
            handle_error(
                config.error_behaviors.monte_carlo_rejection,
                message=(
                    f"{len(self.failed)} of {self.n_samples} Monte Carlo samples failed and were "
                    f"left out (rejection rate {rejection_rate:.1%}). The distribution of the "
                    "remaining samples differs from the declared one. See "
                    "MonteCarloResult.failed_samples for the errors."
                    + config_hint("monte_carlo_rejection")
                ),
            )
        if len(self.successful) < 2:
            return
        n_used = len(self.successful)
        if any(_varies(values[..., :n_used]) for values in self.storage.values()):
            return
        if has_uncertainty:
            message = (
                "No output differs between the Monte Carlo samples. Check that the model uses "
                "the uncertain parameters it is given."
            )
        else:
            message = (
                "The uncertainty definition has no uncertain parameters, so all Monte Carlo "
                "samples are the same."
            )
        handle_error(
            config.error_behaviors.monte_carlo_no_variation,
            message=message + config_hint("monte_carlo_no_variation"),
        )

    def _call(self, sample_index: int, sampled: dict[str, np.ndarray]) -> dict[str, _Output] | None:
        """Run the model for one sample. None if it failed and failures are skipped."""
        try:
            raw_outputs = self.model(self._copies(sampled))
        except Exception as error:
            if self.on_error == "raise" or isinstance(error, ModelCheckError):
                message = f"Monte Carlo sample {sample_index} failed (seed={self.seed})."
                _report_failure(error, message)
                raise
            self.failed[sample_index] = f"{type(error).__name__}: {error}"
            if self.first_failure is None:
                self.first_failure = (sample_index, error)
            return None
        return _convert_outputs(raw_outputs)

    def _copies(self, sampled: dict[str, np.ndarray]) -> AttributeDict[FlodymArray]:
        """Copies of all parameters, with the sampled values for the uncertain ones."""
        copies = AttributeDict[FlodymArray]()
        for name, parameter in self.parameters.items():
            if name not in sampled:
                copies[name] = parameter.copy()
                continue
            # copied, since the model may modify it, and the first sample is run twice
            values = sampled[name].copy()
            copies[name] = parameter.model_copy(
                update={"dims": parameter.dims.copy(), "values": values}
            )
        return copies

    def _check_deterministic(
        self, sample_index: int, sampled: dict[str, np.ndarray], outputs: dict[str, _Output]
    ) -> None:
        """Run the sample again and check that the outputs are identical.

        Raises:
            ValueError: If the outputs differ or the second run fails.
        """
        hint = (
            "The model carries state from one run to the next, e.g. through a parameter or "
            "array it keeps and modifies in place, or it uses random numbers of its own. Make "
            "sure that the outputs only depend on the given parameters."
        )
        try:
            raw_outputs = self.model(self._copies(sampled))
        except Exception as error:
            raise ValueError(
                f"The second run of Monte Carlo sample {sample_index}, with the same parameters "
                f"as the first, failed. {hint}"
            ) from error
        second = _convert_outputs(raw_outputs)
        if list(second) != list(outputs):
            raise ValueError(
                f"Two runs of Monte Carlo sample {sample_index} with the same parameters returned "
                f"different outputs: {list(outputs)} and {list(second)}. {hint}"
            )
        differences = []
        for name, output in outputs.items():
            if output.dims != second[name].dims:
                differences.append((np.inf, f"'{name}' (different dimensions)"))
            elif not np.array_equal(output.values, second[name].values, equal_nan=True):
                difference = _largest_difference(output.values, second[name].values)
                differences.append((difference, f"'{name}' (largest difference {difference:.6g})"))
        if differences:
            differences.sort(key=lambda item: -item[0])
            listed = ", ".join(label for _, label in differences[:_MAX_LISTED_OUTPUTS])
            more = len(differences) - _MAX_LISTED_OUTPUTS
            listed += f" and {more} more" if more > 0 else ""
            raise ValueError(
                f"Two runs of Monte Carlo sample {sample_index} with the same parameters returned "
                f"different values for the outputs {listed}. {hint}"
            )

    def _check_memory(self) -> None:
        """Raise a ValueError if the samples of all outputs need more than max_memory_gib."""
        sizes = {
            name: layout.dims.total_size * np.dtype(float).itemsize * self.n_samples
            for name, layout in self.layout.items()
        }
        total = sum(sizes.values())
        if total <= self.max_memory_gib * _GIB:
            return
        largest = sorted(sizes.items(), key=lambda item: -item[1])[:_MAX_LISTED_OUTPUTS]
        listed = ", ".join(f"'{name}' ({size / _GIB:.3g} GiB)" for name, size in largest)
        raise ValueError(
            f"The {self.n_samples} samples of all outputs need {total / _GIB:.3g} GiB, more than "
            f"max_memory_gib={self.max_memory_gib}. Largest outputs: {listed}. Return fewer or "
            "aggregated outputs from the model; for an MFASystem, restrict track or add derived "
            "outputs in make_model_function(). Alternatively, increase max_memory_gib, or split "
            "the run with sample_indices."
        )

    def _check_layout(self, outputs: dict[str, _Output], sample_index: int) -> None:
        """Raise a ValueError if the outputs have other names or dimensions than the first."""
        if list(outputs) != list(self.layout):
            raise ValueError(
                f"Monte Carlo sample {sample_index} returned the outputs {list(outputs)}, but the "
                f"first sample returned {list(self.layout)}. The model must return the same "
                "outputs for all samples."
            )
        for name, output in outputs.items():
            first_dims = self.layout[name].dims
            if output.dims != first_dims:
                raise ValueError(
                    f"Output '{name}' of Monte Carlo sample {sample_index} has the dimensions "
                    f"{output.dims.letters} with shape {output.dims.shape}, but the one of the "
                    f"first sample has {first_dims.letters} with shape {first_dims.shape}."
                )


def _convert_outputs(raw_outputs: Any) -> dict[str, _Output]:
    """Dimensions, copied values and extensivity degree of each output.

    Raises:
        TypeError: If the outputs are not a mapping of names to FlodymArrays or numbers.
    """
    if not isinstance(raw_outputs, Mapping):
        raise TypeError(
            "The model must return a mapping of output names to FlodymArrays or numbers, but "
            f"returned {type(raw_outputs).__name__}."
        )
    converted = {}
    for name, value in raw_outputs.items():
        if not isinstance(name, str):
            raise TypeError(f"Output names must be strings, got {name!r}.")
        if isinstance(value, FlodymArray):
            values = np.array(value.values, dtype=float)
            converted[name] = _Output(value.dims.copy(), values, value._degree)
        elif isinstance(value, Real) and not isinstance(value, bool):
            converted[name] = _Output(DimensionSet.empty(), np.array(float(value)), None)
        else:
            raise TypeError(
                f"Output '{name}' of the model is a {type(value).__name__}, but must be a "
                "FlodymArray or a number."
            )
    return converted


def _check_same_run(first: MonteCarloResult, result: MonteCarloResult, position: int) -> None:
    """Raise a ValueError if two results cannot be parts of one run."""
    if result.seed != first.seed:
        raise ValueError(
            f"Result {position} has the seed {result.seed}, but result 0 has the seed "
            f"{first.seed}. Run all parts with the same seed."
        )
    if result.definition_fingerprint != first.definition_fingerprint:
        raise ValueError(
            f"Result {position} was run with another uncertainty definition than result 0. Run "
            "all parts with the same UncertaintyDefinition."
        )
    if result.parameters_fingerprint != first.parameters_fingerprint:
        raise ValueError(
            f"Result {position} was run with other nominal parameter values than result 0. Run "
            "all parts with the same parameters."
        )
    if list(result.outputs) != list(first.outputs):
        raise ValueError(
            f"Result {position} has the outputs {list(result.outputs)}, but result 0 has "
            f"{list(first.outputs)}. Run all parts with the same model."
        )
    for name, output in result.outputs.items():
        if output.dims != first.outputs[name].dims:
            raise ValueError(
                f"Output '{name}' has other dimensions in result {position} than in result 0. "
                "Run all parts with the same model."
            )


def _report_failure(error: BaseException, message: str) -> None:
    """Log the failure of a sample as an error, and add it to the exception as a note.

    Notes exist since Python 3.11.
    """
    logger.error(message)
    add_note = getattr(error, "add_note", None)
    if add_note is not None:
        add_note(message)


def _largest_difference(values: np.ndarray, other: np.ndarray) -> float:
    """Largest absolute difference; infinite if NaN values or infinities differ."""
    with np.errstate(invalid="ignore"):
        difference = np.abs(values - other)
    nan_differs = np.isnan(values) != np.isnan(other)
    if np.any(nan_differs) or np.any(np.isnan(difference) & ~np.isnan(values)):
        return float("inf")
    return float(np.nanmax(difference, initial=0.0))


def _varies(samples: np.ndarray) -> bool:
    """Whether the samples (last axis) differ; NaN values are equal to each other."""
    first = samples[..., :1]
    return not np.all((samples == first) | (np.isnan(samples) & np.isnan(first)))
