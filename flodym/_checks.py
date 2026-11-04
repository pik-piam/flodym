"""Private helpers shared by the mass balance checks of processes, stocks and MFA systems."""

import logging
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np

from .config import ErrorBehavior, handle_error

if TYPE_CHECKING:
    from .flodym_arrays import FlodymArray


def config_hint(error_behavior: str, tolerance: bool = False, check: Optional[str] = None) -> str:
    """Brief note for error messages, naming the config options that control a check."""
    options = [f"flodym.config.error_behaviors.{error_behavior}"]
    if tolerance:
        options.append("flodym.config.relative_tolerance or absolute_tolerance")
    if check is not None:
        options.append(f"flodym.config.checks.{check}")
    return " Configure with " + ", ".join(options) + "."


def max_abs(array: Union["FlodymArray", float]) -> float:
    """Largest absolute value of a FlodymArray or scalar. NaN if any value is NaN."""
    values = np.asarray(getattr(array, "values", array))
    return float(np.max(np.abs(values), initial=0.0))


def balance_failure(
    label: str, balance: Union["FlodymArray", float], tolerance: float
) -> Optional[str]:
    """Description of a failed mass balance check for use in an error message,
    or None if the check passed. NaN errors fail the check."""
    max_error = max_abs(balance)
    if max_error <= tolerance:
        return None
    return f"{label} (max error: {max_error}, tolerance: {tolerance})"


def report_balance_failures(
    failures: Sequence[Optional[str]],
    error_behavior: ErrorBehavior | str,
    hint: str,
    success_message: str,
) -> bool:
    """Handle the results of one or several mass balance checks, given as returned by
    :py:func:`balance_failure`. Returns whether all checks passed."""
    failed = [f for f in failures if f is not None]
    if failed:
        message = "Mass balance check failed for " + ", ".join(failed) + "." + hint
        handle_error(behavior=error_behavior, message=message)
        return False
    logging.info(success_message)
    return True
