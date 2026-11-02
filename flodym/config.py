"""Global configuration of flodym, accessible via ``flodym.config``."""

from contextlib import contextmanager
from enum import Enum
from typing import Any, Generator, Optional
import logging

from pydantic import BaseModel


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


class ErrorBehaviors(BaseModel, validate_assignment=True):
    """
    What to do if a check fails.
    """

    unset_array_used: ErrorBehavior = ErrorBehavior.IGNORE
    """What to do if a FlodymArray which is not set (see :py:attr:`flodym.FlodymArray.is_set`)
    is used in an operation or a stock computation.
    """


class Config(BaseModel, validate_assignment=True):
    """
    Configuration class for Flodym.
    """

    error_behaviors: ErrorBehaviors = ErrorBehaviors()
    """What to do if a check fails."""

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
