"""A distribution bound to the nominal values of the slice of a parameter it applies to.

It checks the distribution parameters against the nominal values and computes sampled values
from unit-interval numbers, including truncation and values without spread.
"""

from typing import Any

import numpy as np

from .distributions import Distribution


class InvalidValuesError(Exception):
    """Values fail a check. Formatted with the items by :py:class:`flodym._sampling.BoundEntry`."""

    def __init__(self, failed: np.ndarray, message: str) -> None:
        super().__init__(message)
        self.failed = failed
        """Boolean array, True for the failing values."""
        self.message = message


class BoundDistribution:
    """A distribution bound to nominal values.

    Args:
        distribution: The distribution.
        nominal: The nominal values, as a one-dimensional float array.
        field_values: The values of all fields of the distribution; numbers and arrays as arrays
            of the shape of ``nominal``.

    Raises:
        InvalidValuesError: If values are NaN, or a check of the distribution fails.
        TypeError: If the distribution does not return a frozen scipy.stats distribution.
    """

    def __init__(
        self, distribution: Distribution, nominal: np.ndarray, field_values: dict[str, Any]
    ) -> None:
        checks = {"nominal value is NaN": np.isnan(nominal)}
        checks |= distribution._failed_checks(nominal, field_values)
        for message, failed in checks.items():
            _raise_where(np.arange(nominal.size), nominal.shape, failed, message)
        arguments, lower, upper = distribution._arguments(nominal, field_values)
        self.lower = lower
        self.upper = upper

        self.fixed_values = nominal.copy()
        """Values where the distribution has no spread; NaN elsewhere."""
        at_nominal = np.asarray(distribution.no_spread(nominal, **arguments), dtype=bool)
        at_bound = np.zeros(nominal.shape, dtype=bool)
        if lower is not None and upper is not None:
            at_bound = lower == upper
            self.fixed_values[at_bound] = lower[at_bound]
        self.random_ids = np.flatnonzero(~(at_nominal | at_bound))
        """Indices of the values with a spread."""
        self.fixed_values[self.random_ids] = np.nan
        self.frozen: Any = None
        self.cdf_lower: np.ndarray | float = 0.0
        self.cdf_upper: np.ndarray | float = 1.0
        if self.random_ids.size == 0:
            return
        ids = self.random_ids
        frozen_arguments = {
            key: value[ids] if isinstance(value, np.ndarray) else value
            for key, value in arguments.items()
        }
        self.frozen = distribution.frozen(nominal[ids], **frozen_arguments)
        if not (hasattr(self.frozen, "ppf") and hasattr(self.frozen, "cdf")):
            raise TypeError(
                f"{type(distribution).__name__}.frozen() must return a frozen scipy.stats "
                f"distribution, but returned {type(self.frozen).__name__}."
            )
        if lower is not None:
            self.cdf_lower = np.asarray(self.frozen.cdf(lower[ids]), dtype=float)
        if upper is not None:
            self.cdf_upper = np.asarray(self.frozen.cdf(upper[ids]), dtype=float)
        no_mass = self.cdf_upper <= self.cdf_lower
        _raise_where(ids, nominal.shape, no_mass, "no probability between the bounds")
        # scipy returns NaN instead of raising for invalid distribution parameters
        median = self.frozen.ppf(self.cdf_lower + 0.5 * (self.cdf_upper - self.cdf_lower))
        message = "the distribution gives NaN values; check its parameters"
        _raise_where(ids, nominal.shape, np.isnan(median), message)

    def ppf(self, unit_numbers: np.ndarray) -> np.ndarray:
        """Values for unit-interval numbers, given by sample (axis 0) and value (axis 1)."""
        values = np.tile(self.fixed_values, (unit_numbers.shape[0], 1))
        if self.frozen is not None:
            ids = self.random_ids
            probabilities = self.cdf_lower + unit_numbers[:, ids] * (
                self.cdf_upper - self.cdf_lower
            )
            values[:, ids] = self.frozen.ppf(probabilities)
        if self.lower is not None:
            np.maximum(values, self.lower, out=values)
        if self.upper is not None:
            np.minimum(values, self.upper, out=values)
        return values


def _raise_where(ids: np.ndarray, shape: tuple[int, ...], failed: Any, message: str) -> None:
    """Raise an InvalidValuesError if a check of the values with the given indices failed."""
    failed_values = np.zeros(shape, dtype=bool)
    failed_values[ids] = np.broadcast_to(np.asarray(failed, dtype=bool), ids.shape)
    if np.any(failed_values):
        raise InvalidValuesError(failed_values, message)
