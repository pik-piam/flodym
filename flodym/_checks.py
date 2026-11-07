"""Private helpers shared by the mass balance checks of processes, stocks and MFA systems."""

import logging
from collections.abc import Sequence
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
