"""Global configuration of flodym, accessible via ``flodym.config``."""

from contextlib import contextmanager
from enum import Enum
from typing import Annotated, Any, Generator, Optional
import logging

from pydantic import BaseModel, Field


class ErrorBehavior(str, Enum):
    ERROR = "error"
    WARN = "warn"
    INFO = "info"
    IGNORE = "ignore"


def handle_error(
    behavior: ErrorBehavior, message: Optional[str] = None, error: Optional[Exception] = None
):
    """Raise, log, or ignore an error, depending on the given behavior.
    Exactly one of message or error must be given.
    """

    # ensures valid ErrorBehavior
    behavior = ErrorBehavior(behavior)

    if (message is None) == (error is None):
        raise ValueError("Exactly one of message or error must be provided.")

    if behavior == ErrorBehavior.ERROR:
        raise error or ValueError(message)
    elif behavior == ErrorBehavior.WARN:
        logging.warning(message or str(error))
    elif behavior == ErrorBehavior.INFO:
        logging.info(message or str(error))
    elif behavior == ErrorBehavior.IGNORE:
        pass


def _config_hint(
    error_behavior: str, tolerance: bool = False, check: Optional[str] = None
) -> str:
    """Brief note for error messages, naming the config options that control a check."""
    options = [f"flodym.config.error_behaviors.{error_behavior}"]
    if tolerance:
        options.append("flodym.config.relative_tolerance or absolute_tolerance")
    if check is not None:
        options.append(f"flodym.config.checks.{check}")
    return " Configure with " + ", ".join(options) + "."


def _exceeds_tolerance(max_error: float, tolerance: float) -> bool:
    """Whether a check fails. NaN errors fail as well."""
    return not max_error <= tolerance


class Checks(BaseModel, validate_assignment=True):
    """
    Which checks to perform automatically during computation.
    """

    mass_balance_stocks: bool = True
    """Whether to check the mass balance of a stock after each of its computations
    (see :py:meth:`flodym.Stock.check_mass_balance`)."""


class ErrorBehaviors(BaseModel, validate_assignment=True):
    """
    What to do if a check fails.
    """

    unset_array_used: ErrorBehavior = ErrorBehavior.IGNORE
    """What to do if a FlodymArray which is not set (see :py:attr:`flodym.FlodymArray.is_set`)
    is used in an operation or a stock computation.
    """
    mass_balance: ErrorBehavior = ErrorBehavior.ERROR
    """What to do if an explicitly called mass balance check fails
    (see :py:meth:`flodym.MFASystem.check_mass_balance`, :py:meth:`flodym.Process.check_mass_balance`,
    :py:meth:`flodym.Stock.check_mass_balance`).
    """
    mass_balance_stocks: ErrorBehavior = ErrorBehavior.WARN
    """What to do if the automatic mass balance check after a stock computation fails
    (see :py:attr:`Checks.mass_balance_stocks`).
    """
    check_flows: ErrorBehavior = ErrorBehavior.WARN
    """What to do if a flow has negative or NaN values (see :py:meth:`flodym.MFASystem.check_flows`)."""


class Config(BaseModel, validate_assignment=True):
    """
    Configuration class for Flodym.
    """

    checks: Checks = Checks()
    """Which checks to perform automatically during computation."""

    error_behaviors: ErrorBehaviors = ErrorBehaviors()
    """What to do if a check fails."""

    relative_tolerance: Annotated[float, Field(ge=0)] = 100.0
    """Tolerance for mass balance and other checks, relative to the float precision of the checked
    values: The absolute tolerance is ``relative_tolerance`` times the machine epsilon of the
    data type, times the largest absolute value of the arrays involved in the check.
    Increase if checks fail due to rounding errors.
    Not used if :py:attr:`absolute_tolerance` is set.
    """

    absolute_tolerance: Optional[Annotated[float, Field(ge=0)]] = None
    """Absolute tolerance for mass balance and other checks.
    If set, it is used instead of :py:attr:`relative_tolerance`.
    """

    def get_tolerance(self, float_precision: float) -> float:
        """Absolute tolerance for checks, given the float precision of the checked values
        (machine epsilon times the largest absolute value).
        Returns :py:attr:`absolute_tolerance` if set, else ``relative_tolerance * float_precision``.
        """
        if self.absolute_tolerance is not None:
            return self.absolute_tolerance
        return self.relative_tolerance * float_precision

    @contextmanager
    def override(self, **changes: Any) -> Generator["Config", None, None]:
        """Temporarily change config options within a ``with`` block.
        Nested options are given as dictionaries.

        Example:
            >>> with config.override(error_behaviors={"unset_array_used": "error"}):
            ...     ...
        """
        old_values = {}
        try:
            for name, value in changes.items():
                old_values[name] = getattr(self, name)
                if isinstance(value, dict):
                    old_value = old_values[name]
                    value = type(old_value).model_validate({**old_value.model_dump(), **value})
                setattr(self, name, value)
            yield self
        finally:
            for name, value in old_values.items():
                setattr(self, name, value)


config = Config()
