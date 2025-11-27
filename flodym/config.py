from pydantic import BaseModel, confloat
from typing import Optional
from enum import Enum
import logging


class ErrorBehavior(str, Enum):
    ERROR = "error"
    WARN = "warn"
    INFO = "info"
    IGNORE = "ignore"


def handle_error(behavior: ErrorBehavior, message: str = None, error: Exception = None):

    # ensures valid ErrorBehavior
    behavior = ErrorBehavior(behavior)

    if message is None == error is None:
        raise ValueError("Exactly one of message or error must be provided.")

    if behavior == ErrorBehavior.ERROR:
        raise error or ValueError(message)
    elif behavior == ErrorBehavior.WARN:
        logging.warning(message or str(error))
    elif behavior == ErrorBehavior.INFO:
        logging.info(message or str(error))
    elif behavior == ErrorBehavior.IGNORE:
        pass


class Checks(BaseModel, validate_assignment=True):
    """
    Which checks to perform during computation.
    """

    mass_balance_stocks: bool = True
    """Whether to check mass balance after each `stock.compute()` call."""


class ErrorBehaviors(BaseModel, validate_assignment=True):
    """
    What to do if a check fails.
    """

    mass_balance: ErrorBehavior = ErrorBehavior.ERROR
    """what to do if mass balance is not satisfied."""
    check_flows: ErrorBehavior = ErrorBehavior.WARN
    """What to do if a flow has negative or NaN value."""


class Config(BaseModel, validate_assignment=True):
    """
    Configuration class for Flodym.
    """

    checks: Checks = Checks()
    """Which checks to perform during computation."""

    error_behaviors: ErrorBehaviors = ErrorBehaviors()
    """What to do if a check fails."""

    relative_tolerance: Optional[confloat(ge=0)] = 10.0
    """Tolerance relative to a float precision metric used for mass balance and other checks.
    Increase if your mass balance checks fail due to rounding errors.
    Overridden by `absolute_tolerance` if set.
    """
    absolute_tolerance: Optional[confloat(ge=0)] = None
    """Absolute tolerance used for mass balance and other checks.
    Overrides `relative_tolerance` if set.
    If None, `relative_tolerance` is used.
    """


config = Config()
