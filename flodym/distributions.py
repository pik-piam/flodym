"""Distributions of uncertain parameter values, see :py:class:`flodym.UncertaintyDefinition`."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
import scipy.stats
from pydantic import model_validator
from typing_extensions import Self

from ._base_model import FlodymBaseModel
from .flodym_arrays import FlodymArray


class Distribution(FlodymBaseModel, ABC):
    """Base class of the distributions of uncertain parameter values.

    A distribution describes the spread of the values of a parameter around their nominal values,
    i.e. the values the parameter has in the deterministic model. One distribution covers all
    values of a parameter, or of a slice of it (see :py:class:`flodym.ParameterUncertainty`).

    Distribution parameters, such as a standard deviation or a bound, are numbers or
    :py:class:`flodym.FlodymArray` objects over a subset of the dimensions of the uncertain
    parameter, or over ancestors of them (see :py:class:`flodym.Dimension`). Parameters with the
    prefix ``relative_`` are given relative to the nominal values.

    **Bounds**

    ``lower`` and ``upper`` define the uniform and the triangular distribution and truncate the
    others. Truncated distributions are sampled exactly, as
    ``ppf(cdf(lower) + u * (cdf(upper) - cdf(lower)))`` with ``u`` uniform on the unit interval,
    which is what redrawing samples outside the bounds converges to. Sampled values lie within
    the bounds, unless ``normalize_over`` of :py:class:`flodym.UncertaintyDefinition` rescales
    them afterwards. Bounds which are not symmetric around the nominal value shift the mean of a
    truncated distribution away from the nominal value, e.g. ``upper=1.0`` for a yield close to
    one lowers it.

    Where a distribution has no spread, e.g. a standard deviation of zero, the values stay at
    their nominal values.

    **Own distributions**

    Subclass ``Distribution``, declare the parameters of the distribution as fields and implement
    :py:meth:`frozen`. Truncation, broadcasting and sampling are done by the base class.
    Optionally, override :py:meth:`invalid_values` to check the parameters, and
    :py:meth:`no_spread` for values which stay at their nominal values. For example, a beta
    distribution with the nominal value as mean, e.g. for a share:

        >>> import scipy.stats
        >>> import flodym as fd
        >>> class BetaDistribution(fd.Distribution):
        ...     concentration: fd.FlodymArray | float
        ...
        ...     def frozen(self, nominal, concentration):
        ...         return scipy.stats.beta(nominal * concentration, (1 - nominal) * concentration)
        ...
        ...     def invalid_values(self, nominal, concentration):
        ...         return {"nominal value is not between 0 and 1": (nominal < 0) | (nominal > 1)}
        ...
        ...     def no_spread(self, nominal, concentration):
        ...         return (nominal == 0) | (nominal == 1)

    Distributions which are defined by their bounds, like :py:class:`flodym.UniformDistribution`,
    set the class variable :py:attr:`requires_bounds` to True. Both bounds are then required, and
    they are passed to the methods as ``lower`` and ``upper``.
    """

    lower: FlodymArray | float | None = None
    """Lower bound of the values. At most one of ``lower`` and ``relative_lower`` can be given."""
    relative_lower: FlodymArray | float | None = None
    """Lower bound relative to the nominal values: ``lower = relative_lower * nominal value``."""
    upper: FlodymArray | float | None = None
    """Upper bound of the values. At most one of ``upper`` and ``relative_upper`` can be given."""
    relative_upper: FlodymArray | float | None = None
    """Upper bound relative to the nominal values: ``upper = relative_upper * nominal value``."""

    requires_bounds: ClassVar[bool] = False
    """Whether the distribution is defined by its bounds, like a uniform distribution. Both bounds
    are then required, and they are passed to :py:meth:`frozen`, :py:meth:`invalid_values` and
    :py:meth:`no_spread` as ``lower`` and ``upper``. Set it in subclasses."""

    @model_validator(mode="after")
    def _check_bounds_given(self) -> Self:
        """Check that each bound is given at most once, and both if they are required."""
        name = type(self).__name__
        for bound in ("lower", "upper"):
            if getattr(self, bound) is not None and getattr(self, f"relative_{bound}") is not None:
                raise ValueError(f"{name}: Give at most one of {bound} and relative_{bound}.")
        if self.requires_bounds and (not self._has_bound("lower") or not self._has_bound("upper")):
            raise ValueError(
                f"{name} requires a lower bound (lower or relative_lower) and an upper bound "
                "(upper or relative_upper)."
            )
        return self

    @model_validator(mode="after")
    def _check_numbers(self) -> Self:
        """Run the checks of :py:meth:`invalid_values` on the arguments given as numbers.

        Checks which depend on the nominal values, which are not known yet, pass. If arguments
        are given as arrays, all checks are run when the distribution is used.
        """
        field_values: dict[str, Any] = {}
        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            if isinstance(value, FlodymArray):
                return self
            is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
            field_values[field_name] = np.array([float(value)]) if is_number else value
        with np.errstate(invalid="ignore"):
            failed = self._failed_checks(np.array([np.nan]), field_values)
        for message, failed_values in failed.items():
            if np.any(failed_values):
                raise ValueError(f"{type(self).__name__}: {message}.")
        return self

    @abstractmethod
    def frozen(self, nominal: np.ndarray, **parameters: Any) -> Any:
        """Return the distribution for the given nominal values.

        Override this method in subclasses. Bounds are applied by the base class, unless the
        distribution is defined by them (see :py:attr:`requires_bounds`).

        Args:
            nominal: The nominal values the distribution applies to, as a one-dimensional array.
                Values without spread (see :py:meth:`no_spread`) are left out.
            **parameters: The fields declared by the subclass, as arrays of the same shape as
                ``nominal`` (None if not given). Distributions which require bounds also get
                the absolute bounds ``lower`` and ``upper``.

        Returns:
            A frozen ``scipy.stats`` distribution, e.g. ``scipy.stats.norm(loc=..., scale=...)``,
            whose parameters broadcast with ``nominal``. Sampling uses its methods ``ppf`` and
            ``cdf``; future features may use others, such as ``mean``, ``std`` or ``pdf``.
        """

    def invalid_values(self, nominal: np.ndarray, **parameters: Any) -> dict[str, np.ndarray]:
        """Check the distribution parameters against the nominal values.

        Override in subclasses to add checks; NaN values and a lower bound above the upper one are
        checked by the base class. Failed checks raise a ``ValueError``, which names the
        parameter and the failing items. Arguments given as numbers are checked already when the
        distribution is created, with NaN as nominal values. Checks must therefore not flag NaN
        nominal values, which are checked separately: write ``(nominal < 0) | (nominal > 1)``,
        not ``~((nominal >= 0) & (nominal <= 1))``.

        Args:
            nominal: The nominal values, as for :py:meth:`frozen`.
            **parameters: The distribution parameters, as for :py:meth:`frozen`.

        Returns:
            For each check, a description of the failure with a remedy (e.g.
            ``"std is negative; give a non-negative std"``) and a boolean array, which is True for
            the values failing it.
        """
        return {}

    def no_spread(self, nominal: np.ndarray, **parameters: Any) -> np.ndarray:
        """Where the distribution has no spread, so that the values stay at their nominal values.

        Override in subclasses, e.g. for a standard deviation of zero. Values where the bounds
        are equal are set to the bound by the base class.

        Args:
            nominal: The nominal values, as for :py:meth:`frozen`.
            **parameters: The distribution parameters, as for :py:meth:`frozen`.

        Returns:
            A boolean array, which is True where values stay at their nominal values.
        """
        return np.zeros(nominal.shape, dtype=bool)

    @classmethod
    def _parameter_names(cls) -> list[str]:
        """Names of the fields declared by the subclass, which are passed to the methods."""
        return [name for name in cls.model_fields if name not in Distribution.model_fields]

    def _arguments(
        self, nominal: np.ndarray, field_values: dict[str, Any]
    ) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None]:
        """Arguments of the methods, and the absolute bounds.

        Args:
            nominal: The nominal values.
            field_values: The values of all fields; numbers and arrays as arrays of the shape of
                ``nominal``.

        Returns:
            The keyword arguments of :py:meth:`frozen`, :py:meth:`invalid_values` and
            :py:meth:`no_spread`, and the lower and upper bound (None if not given).
        """
        lower = _resolve_bound(field_values["lower"], field_values["relative_lower"], nominal)
        upper = _resolve_bound(field_values["upper"], field_values["relative_upper"], nominal)
        arguments = {name: field_values[name] for name in self._parameter_names()}
        if self.requires_bounds:
            arguments |= {"lower": lower, "upper": upper}
        return arguments, lower, upper

    def _failed_checks(
        self, nominal: np.ndarray, field_values: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        """All checks of the distribution parameters, see :py:meth:`invalid_values`."""
        arguments, lower, upper = self._arguments(nominal, field_values)
        checks = {
            f"{name} is NaN; give a number": np.isnan(values)
            for name, values in field_values.items()
            if isinstance(values, np.ndarray) and np.issubdtype(values.dtype, np.number)
        }
        if lower is not None and upper is not None:
            checks["lower bound is greater than upper bound; give lower <= upper"] = lower > upper
        return checks | self.invalid_values(nominal, **arguments)

    def _has_bound(self, bound: str) -> bool:
        return getattr(self, bound) is not None or getattr(self, f"relative_{bound}") is not None


class _StdDistribution(Distribution):
    """Base class of distributions with the nominal value as mean and a standard deviation."""

    std: FlodymArray | float | None = None
    """Standard deviation."""
    relative_std: FlodymArray | float | None = None
    """Standard deviation relative to the absolute nominal value:
    ``std = relative_std * |nominal value|``. Unlike ``relative_std`` of the lifetime models
    (see :py:class:`flodym.NormalLifetime`), which is a flag, it is the relative standard
    deviation itself."""

    @model_validator(mode="after")
    def _check_std_given(self) -> Self:
        """Check that exactly one of std and relative_std is given."""
        if (self.std is None) == (self.relative_std is None):
            raise ValueError(f"{type(self).__name__}: Give exactly one of std and relative_std.")
        return self

    def invalid_values(
        self, nominal: np.ndarray, std: np.ndarray | None, relative_std: np.ndarray | None
    ) -> dict[str, np.ndarray]:
        """Check that the standard deviation is not negative.

        See :py:meth:`flodym.Distribution.invalid_values`.
        """
        if std is not None:
            return {"std is negative; give a non-negative std": std < 0}
        if relative_std is not None:
            return {"relative_std is negative; give a non-negative relative_std": relative_std < 0}
        return {}

    def no_spread(
        self, nominal: np.ndarray, std: np.ndarray | None, relative_std: np.ndarray | None
    ) -> np.ndarray:
        """Where the standard deviation is zero, see :py:meth:`flodym.Distribution.no_spread`."""
        return _absolute_std(nominal, std, relative_std) == 0


class NormalDistribution(_StdDistribution):
    """Normal distribution with the nominal value as mean.

    Exactly one of ``std`` and ``relative_std`` must be given. With bounds, the distribution is
    truncated (see :py:class:`flodym.Distribution`); its mean then differs from the nominal value.

        >>> import flodym as fd
        >>> yield_uncertainty = fd.NormalDistribution(relative_std=0.05, lower=0.0, upper=1.0)
    """

    def frozen(
        self, nominal: np.ndarray, std: np.ndarray | None, relative_std: np.ndarray | None
    ) -> Any:
        """Return a normal distribution with the nominal values as mean.

        See :py:meth:`flodym.Distribution.frozen`.
        """
        return scipy.stats.norm(loc=nominal, scale=_absolute_std(nominal, std, relative_std))


class LogNormalDistribution(_StdDistribution):
    """Lognormal distribution with the nominal value as mean.

    The distribution is parametrized by mean and standard deviation of the distribution itself,
    not of the underlying normal distribution, as :py:class:`flodym.LogNormalLifetime`. Exactly
    one of ``std`` and ``relative_std`` must be given. Nominal values must not be negative.
    Nominal values of zero stay zero, e.g. missing values read as zero (see
    ``allow_missing_values`` of :py:class:`flodym.CSVParameterReader`).

        >>> import flodym as fd
        >>> flow_uncertainty = fd.LogNormalDistribution(relative_std=0.2)
    """

    def frozen(
        self, nominal: np.ndarray, std: np.ndarray | None, relative_std: np.ndarray | None
    ) -> Any:
        """Return a lognormal distribution with the nominal values as mean.

        See :py:meth:`flodym.Distribution.frozen`.
        """
        relative = _absolute_std(nominal, std, relative_std) / nominal
        log_variance = np.log1p(relative * relative)
        return scipy.stats.lognorm(
            s=np.sqrt(log_variance), scale=nominal * np.exp(-0.5 * log_variance)
        )

    def invalid_values(
        self, nominal: np.ndarray, std: np.ndarray | None, relative_std: np.ndarray | None
    ) -> dict[str, np.ndarray]:
        """Check that std and the nominal values are not negative.

        See :py:meth:`flodym.Distribution.invalid_values`.
        """
        return super().invalid_values(nominal, std, relative_std) | {
            "nominal value is negative, which a lognormal distribution cannot have; use another "
            "distribution": nominal < 0
        }

    def no_spread(
        self, nominal: np.ndarray, std: np.ndarray | None, relative_std: np.ndarray | None
    ) -> np.ndarray:
        """Where std or the nominal value is zero, see :py:meth:`flodym.Distribution.no_spread`."""
        return super().no_spread(nominal, std, relative_std) | (nominal == 0)


class UniformDistribution(Distribution):
    """Uniform distribution between ``lower`` and ``upper``, which are required.

    The nominal value is only used for relative bounds.

        >>> import flodym as fd
        >>> uniform = fd.UniformDistribution(relative_lower=0.9, relative_upper=1.1)
    """

    requires_bounds: ClassVar[bool] = True

    def frozen(self, nominal: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> Any:
        """Return a uniform distribution between the bounds.

        See :py:meth:`flodym.Distribution.frozen`.
        """
        return scipy.stats.uniform(loc=lower, scale=upper - lower)


class TriangularDistribution(Distribution):
    """Triangular distribution between ``lower`` and ``upper``, which are required.

    Its mode, the most likely value, is the nominal value unless ``mode`` is given. The mode must
    lie between the bounds.

        >>> import flodym as fd
        >>> triangular = fd.TriangularDistribution(relative_lower=0.8, relative_upper=1.2)
    """

    mode: FlodymArray | float | None = None
    """Most likely value. If not given, the nominal value."""

    requires_bounds: ClassVar[bool] = True

    def frozen(
        self, nominal: np.ndarray, lower: np.ndarray, upper: np.ndarray, mode: np.ndarray | None
    ) -> Any:
        """Return a triangular distribution between the bounds.

        See :py:meth:`flodym.Distribution.frozen`.
        """
        mode = nominal if mode is None else mode
        width = upper - lower
        return scipy.stats.triang(c=(mode - lower) / width, loc=lower, scale=width)

    def invalid_values(
        self, nominal: np.ndarray, lower: np.ndarray, upper: np.ndarray, mode: np.ndarray | None
    ) -> dict[str, np.ndarray]:
        """Check that the mode lies between the bounds.

        See :py:meth:`flodym.Distribution.invalid_values`.
        """
        label = "mode (the nominal value)" if mode is None else "mode"
        mode = nominal if mode is None else mode
        message = f"{label} is not between the bounds; give a mode between lower and upper"
        return {message: (mode < lower) | (mode > upper)}


def _absolute_std(
    nominal: np.ndarray, std: np.ndarray | None, relative_std: np.ndarray | None
) -> np.ndarray:
    """Standard deviation from the absolute or the relative one; zero if neither is given."""
    if std is not None:
        return std
    if relative_std is not None:
        return relative_std * np.abs(nominal)
    return np.zeros(nominal.shape)


def _resolve_bound(
    absolute: np.ndarray | None, relative: np.ndarray | None, nominal: np.ndarray
) -> np.ndarray | None:
    """Absolute bound from the absolute or the relative one, or None if neither is given."""
    if absolute is not None:
        return absolute
    if relative is not None:
        return relative * nominal
    return None
