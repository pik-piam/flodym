"""Global configuration of flodym, accessible via ``flodym.config``."""

import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field

from .flow_naming import process_names_with_arrow

logger = logging.getLogger(__name__)


class ErrorBehavior(str, Enum):
    """What to do if a check fails: raise an error, log a warning or info message, or ignore it.

    Wherever an ``ErrorBehavior`` is expected, its string value (e.g. ``"warn"``) can be given.
    """

    ERROR = "error"
    WARN = "warn"
    INFO = "info"
    IGNORE = "ignore"


def handle_error(
    behavior: ErrorBehavior | str, message: str | None = None, error: Exception | None = None
) -> None:
    """Raise, log, or ignore an error, depending on the given behavior.

    Exactly one of message or error must be given.

    Args:
        behavior: What to do with the error.
        message: Message of the error. If raised, a ``ValueError`` is used.
        error: Exception to raise, or whose message to log.

    Raises:
        ValueError: If not exactly one of message and error is given, or if behavior is not a valid
            :py:class:`ErrorBehavior`.
        Exception: The given error, or a ``ValueError`` with the given message, if behavior is
            ``"error"``.
    """
    # ensures valid ErrorBehavior
    behavior = ErrorBehavior(behavior)

    if (message is None) == (error is None):
        raise ValueError("Exactly one of message or error must be provided.")

    if behavior == ErrorBehavior.ERROR:
        raise error or ValueError(message)
    elif behavior == ErrorBehavior.WARN:
        logger.warning(message or str(error))
    elif behavior == ErrorBehavior.INFO:
        logger.info(message or str(error))
    elif behavior == ErrorBehavior.IGNORE:
        pass


class Checks(BaseModel, validate_assignment=True):
    """Which checks to perform automatically during computation."""

    mass_balance_after_compute: bool = False
    """Whether to check the mass balance of a stock after each of its computations
    (see :py:meth:`flodym.Stock.check_mass_balance`), and of a process after each
    :py:meth:`flodym.Process.compute` (see :py:meth:`flodym.Process.check_mass_balance`).
    Failures are handled according to :py:attr:`ErrorBehaviors.mass_balance`.
    Off by default, since :py:meth:`flodym.MFASystem.check_mass_balance` checks all processes and
    stocks as well. Enabling it helps to find where in a computation the mass balance is
    violated."""


class ErrorBehaviors(BaseModel, validate_assignment=True):
    """What to do if a check fails."""

    unset_array_used: ErrorBehavior = ErrorBehavior.IGNORE
    """What to do if a FlodymArray which is not set (see :py:attr:`flodym.FlodymArray.is_set`)
    is used in an operation or a stock computation.
    """
    mass_balance: ErrorBehavior = ErrorBehavior.ERROR
    """What to do if a mass balance check fails
    (see :py:meth:`flodym.MFASystem.check_mass_balance`,
    :py:meth:`flodym.Process.check_mass_balance`, :py:meth:`flodym.Stock.check_mass_balance`, and
    :py:attr:`Checks.mass_balance_after_compute`).
    """
    check_flows: ErrorBehavior = ErrorBehavior.WARN
    """What to do if a flow has negative or NaN values.

    See :py:meth:`flodym.MFASystem.check_flows`.
    """
    process_underdetermined: ErrorBehavior = ErrorBehavior.ERROR
    """What to do if :py:meth:`flodym.Process.compute` is called on a process which cannot be
    computed from the given values. Not used for the neighbors computed with
    ``compute_neighbors=True``, by :py:func:`flodym.compute_processes` and by
    :py:meth:`flodym.MFASystem.compute_all_possible`, where underdetermined processes are skipped.
    """
    process_shares: ErrorBehavior = ErrorBehavior.WARN
    """What to do if the shares or the dimension splitters of a process are invalid in
    :py:meth:`flodym.Process.compute`: If shares are not between zero and one, or sum to more
    than one (or not to exactly one, if all flows of a side have shares), or if a dimension
    splitter does not sum to one over the dimensions it adds.
    If ``"ignore"``, the checks are skipped."""
    process_overdetermined: ErrorBehavior = ErrorBehavior.ERROR
    """What to do if more values are given for a process than needed by
    :py:meth:`flodym.Process.compute` (see :py:attr:`flodym.Process.is_overdetermined`), and they
    are inconsistent: If shares are not fulfilled after the computation, or if stock arrays which
    were set are changed by the computation. Also applies to processes whose flows were all set
    by the computation of neighboring processes. Inconsistent mass balances are handled by
    :py:attr:`mass_balance`. If ``"ignore"``, the checks are skipped."""
    extensivity: ErrorBehavior = ErrorBehavior.WARN
    """What to do if an operation is inconsistent with the extensivity of arrays (see
    :py:attr:`flodym.FlodymArray.extensive`): If an array which is not extensive is summed over
    dimensions, an array which is not intensive is copied along new dimensions, arrays of
    different extensivity are added, subtracted, compared or assigned to each other, an array is
    converted to a class of different default extensivity, or a parameter's extensivity
    contradicts its use in :py:meth:`flodym.MFASystem.link_parameters`."""
    unused_dimension_splitter: ErrorBehavior = ErrorBehavior.WARN
    """What to do if a process has a dimension splitter, but it is not used when computing the
    flows of its side in :py:meth:`flodym.Process.compute`, since no dimensions have to be
    added."""


class Config(BaseModel, validate_assignment=True):
    """Configuration class for Flodym."""

    checks: Checks = Checks()
    """Which checks to perform automatically during computation."""

    error_behaviors: ErrorBehaviors = ErrorBehaviors()
    """What to do if a check fails."""

    flow_naming: Callable[..., str] = process_names_with_arrow
    """Function which generates the name of a flow from the process it starts from and the process
    it ends in, as ``flow_naming(from_process, to_process)``. Used by
    :py:func:`flodym.make_empty_flows` and :py:meth:`flodym.MFASystem.relink`, but not for flows
    with a ``name_override``. Predefined functions are found in :py:mod:`flodym.flow_naming`.
    Default: :py:func:`flodym.flow_naming.process_names_with_arrow`.
    """

    relative_tolerance: Annotated[float, Field(ge=0)] = 100.0
    """Tolerance for mass balance and other checks, relative to the float precision of the checked
    values: The absolute tolerance is ``relative_tolerance`` times the machine epsilon of the
    data type, times the largest absolute value of the arrays involved in the check.
    Increase if checks fail due to rounding errors.
    Not used if :py:attr:`absolute_tolerance` is set.
    """

    absolute_tolerance: Annotated[float, Field(ge=0)] | None = None
    """Absolute tolerance for mass balance and other checks.
    If set, it is used instead of :py:attr:`relative_tolerance`.
    """

    def get_tolerance(self, float_precision: float) -> float:
        """Return the absolute tolerance for checks, given the float precision of the values.

        The float precision is the machine epsilon times the largest absolute value of the
        checked values.
        Returns :py:attr:`absolute_tolerance` if set, else ``relative_tolerance * float_precision``.

        Args:
            float_precision: Machine epsilon of the data type of the checked values, times their
                largest absolute value.
        """
        if self.absolute_tolerance is not None:
            return self.absolute_tolerance
        return self.relative_tolerance * float_precision

    @contextmanager
    def override(self, **changes: Any) -> Generator["Config", None, None]:
        """Temporarily change config options within a ``with`` block.

        Nested options are given as dictionaries.

        Args:
            **changes: New values of config options, with option names as keys. For nested
                options such as ``error_behaviors``, a dictionary of the options to change.

        Yields:
            This config object, with the changed options. The previous values are restored when
            the block is left.

        Example:
            >>> with config.override(error_behaviors={"unset_array_used": "error"}):
            ...     ...
        """
        old_values: dict[str, Any] = {}
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
"""Global configuration object, accessible as ``flodym.config``."""
