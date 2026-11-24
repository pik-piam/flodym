"""Private helpers shared by the mass balance checks of processes, stocks and MFA systems."""

import logging
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Union

import numpy as np

from ._config import ErrorBehavior, handle_error

if TYPE_CHECKING:
    from .flodym_arrays import FlodymArray

logger = logging.getLogger(__name__)


def config_hint(error_behavior: str, tolerance: bool = False, check: str | None = None) -> str:
    """Brief note for error messages, naming the config options that control a check."""
    options = [f"flodym.config.error_behaviors.{error_behavior}"]
    if tolerance:
        options.append("flodym.config.relative_tolerance or absolute_tolerance")
    if check is not None:
        options.append(f"flodym.config.checks.{check}")
    return " Configure with " + ", ".join(options) + "."


def max_abs(array: Union["FlodymArray", np.ndarray, float]) -> float:
    """Largest absolute value of a FlodymArray, numpy array or scalar. NaN if any value is NaN."""
    values = np.asarray(getattr(array, "values", array))
    return float(np.max(np.abs(values), initial=0.0))


def summed_float_precision(arrays: Iterable["FlodymArray"]) -> float:
    """Float precision of the sum of several arrays, as it is computed for a mass balance.

    The arrays are summed to their common dimensions. The rounding error of each entry of the sum
    grows with the absolute values which are summed in it. The precision is therefore the machine
    epsilon times the sum of these absolute values, for the entry where it is largest. For a
    single array, this is the machine epsilon times its largest absolute value. NaN values are
    ignored. Zero if there are no arrays.
    """
    arrays = list(arrays)
    if not arrays:
        return 0.0
    common_letters = [
        letter
        for letter in arrays[0].dims.letters
        if all(letter in array.dims.letters for array in arrays)
    ]
    precision: np.ndarray | float = 0.0
    for array in arrays:
        letters = array.dims.letters
        summed_axes = tuple(i for i, letter in enumerate(letters) if letter not in common_letters)
        abs_values = np.abs(array.values)
        is_nan = np.isnan(abs_values)
        if is_nan.any():
            abs_values = np.where(is_nan, 0.0, abs_values)
        summed = abs_values.sum(axis=summed_axes)
        kept_letters = [letter for letter in letters if letter in common_letters]
        summed = np.transpose(summed, [kept_letters.index(letter) for letter in common_letters])
        dtype = array.values.dtype if np.issubdtype(array.values.dtype, np.inexact) else np.float64
        precision = precision + float(np.finfo(dtype).eps) * summed
    return float(np.max(precision, initial=0.0))


def balance_failure(
    label: str, balance: Union["FlodymArray", float], tolerance: float
) -> str | None:
    """Describe a failed mass balance check for use in an error message.

    Returns None if the check passed. NaN errors fail the check.
    """
    max_error = max_abs(balance)
    if max_error <= tolerance:
        return None
    return f"{label} (max error: {max_error}, tolerance: {tolerance})"


def report_balance_failures(
    failures: Sequence[str | None],
    error_behavior: ErrorBehavior | str,
    hint: str,
    success_message: str,
) -> bool:
    """Handle the results of one or several mass balance checks.

    Args:
        failures: Results of the checks, as returned by :py:func:`balance_failure`.
        error_behavior: What to do if any check failed.
        hint: Appended to the error message, e.g. from :py:func:`config_hint`.
        success_message: Logged at info level if all checks passed.

    Returns:
        Whether all checks passed.
    """
    failed = [f for f in failures if f is not None]
    if failed:
        message = "Mass balance check failed for " + ", ".join(failed) + "." + hint
        handle_error(behavior=error_behavior, message=message)
        return False
    logger.info(success_message)
    return True
