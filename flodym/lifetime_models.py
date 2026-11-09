"""Home to various lifetime models, for use in dynamic stock modelling."""

import warnings
from abc import abstractmethod
from collections.abc import Mapping
from numbers import Number
from types import MappingProxyType
from typing import Any, ClassVar, Literal, SupportsFloat, TypeAlias

import numpy as np
import scipy.special
import scipy.stats
from pydantic import ConfigDict, PrivateAttr, model_validator
from typing_extensions import Self

from ._base_model import FlodymBaseModel
from ._gauss_lobatto import GL_NODES, GL_WEIGHTS
from .dimensions import Dimension, DimensionSet
from .flodym_arrays import FlodymArray

LifetimeArrayType: TypeAlias = FlodymArray | np.ndarray | SupportsFloat
"""Types accepted for lifetime model parameters: a FlodymArray with a subset of the lifetime model
dimensions, a numpy array broadcastable to the lifetime model shape, or a number."""


class _UnevenTimeDim(FlodymBaseModel):
    """Time dimension with possibly uneven spacing, where each item is the center of a time step.

    The bounds of the time steps lie halfway between neighboring items. The outer bounds are
    extrapolated with the length of the neighboring time step.
    """

    dim: Dimension
    _bounds: np.ndarray | None = None

    @property
    def bounds(self) -> np.ndarray:
        """Bounds of the time steps, with one more entry than the time dimension."""
        if self._bounds is None:
            self._compute_t_bounds()
        return self._bounds

    @property
    def interval_lengths(self) -> np.ndarray:
        """Returns the length of the time intervals, i.e. the difference between the bounds."""
        return np.diff(self.bounds)

    def _compute_t_bounds(self) -> None:
        """Computes the bounds of the time steps from the items of the time dimension."""
        middle = (np.array(self.dim.items[:-1]) + np.array(self.dim.items[1:])) / 2.0
        self._bounds = np.concatenate(
            (
                [middle[0] - (middle[1] - middle[0])],
                middle,
                [middle[-1] + (middle[-1] - middle[-2])],
            )
        )


class LifetimeModel(FlodymBaseModel):
    """Contains shared functionality across the various lifetime models."""

    model_config = ConfigDict(validate_assignment=True)

    dims: DimensionSet
    time_letter: str = "t"
    inflow_at: Literal["start", "middle", "end"] = "middle"
    """If no quadrature is used, all inflow happens at one point in time, either at the beginning
    of the time period (start), in the middle (middle) or at the end (end).
    """
    n_pts_per_interval: int = 1
    """Inflow into the stock is in reality quite uniform over time, while the cumulative survival
    only takes into account a single point in time. This can be alleviated here by using numerical
    integration over each inflow time period with n points. A value of 1 means that the inflow
    is only evaluated once, depending on the inflow_at parameter. This is the default and should
    be used for long life times, e.g. >> 1 year.
    For n_pts > 1, the inflow is evaluated at n points in time within the time period.
    inflow_at is ignored in this case.
    A value of 2 means evaluation at the beginning and end of the time period, while higher
    values additionally evaluate at more points within the time period.
    Higher point numbers are only needed for short life times, e.g. < 1 year.
    Default is 1, meaning that the inflow is evaluated only once per time period.
    """
    lt_factor_by_year: LifetimeArrayType | None = None
    """Extend the lifetime (multiply mean and standard deviation by this factor)
    by year, not by age-cohort, so it also takes effect for surviving previous age-cohorts.
    This is the "nurture" approach in Krych et al. 2024 (https://doi.org/10.1111/jiec.13586)
    representing e.g. product care or maintenance.
    Dimensions must be a subset of the lifetime model dimensions.
    """
    lt_factor_by_cohort: LifetimeArrayType | None = None
    """Extend the lifetime (multiply mean and standard deviation by this factor)
    by age-cohort, not by year, so it only takes effect for new age-cohorts.
    This is the "nature" approach in Krych et al. 2024 (https://doi.org/10.1111/jiec.13586)
    representing e.g. product design or material choice.
    Dimensions must be a subset of the lifetime model dimensions.
    """
    hazard_factor_by_year: LifetimeArrayType | None = None
    """Period effect: multiply the hazard, i.e. the intensity of leaving the stock, in a year by
    this factor, for all age-cohorts present in that year. The conditional survival s in that
    year becomes s ** factor. A factor of 1 has no effect, 0 means that nothing leaves the stock,
    and values greater than 1 increase the outflow, e.g. through scrappage schemes.
    The effect does not depend on the length of the time step.
    It has no effect on lifetime models with conditional survival of only 0 and 1, like
    :py:class:`FixedLifetime`.
    For Weibull distributions, it is equivalent to ``lt_factor_by_year`` with
    hazard factor = lt_factor ** (-weibull_shape); for other distributions, it is not.
    Dimensions must be a subset of the lifetime model dimensions.
    """
    removal_share_by_year: LifetimeArrayType | None = None
    """Period effect: share of the stock removed in a time step in addition to the regular
    outflow, independent of age, e.g. by a disaster or a phase-out policy (share 1).
    The conditional survival s in that time step becomes s * (1 - share). The share refers to the
    whole time step, not to a unit of time. Of the age-cohort entering during the time step, only
    the part entering before the removal is affected, assuming the removal is evenly distributed
    over the time step and the inflow occurs as given by ``inflow_at`` and ``n_pts_per_interval``.
    Dimensions must be a subset of the lifetime model dimensions.
    """
    _cumulative_survival: np.ndarray | None = PrivateAttr(default=None)
    _outflow_pmf: np.ndarray | None = PrivateAttr(default=None)
    _conditional_survival: np.ndarray | None = PrivateAttr(default=None)
    _t: _UnevenTimeDim | None = PrivateAttr(default=None)
    _quad_points: list[float] | None = PrivateAttr(default=None)
    """Intra-time step quadrature points for numerical integration of the cumulative survival."""
    _quad_weights: list[float] | None = PrivateAttr(default=None)
    """Intra-time step quadrature weights for numerical integration of the cumulative survival."""
    _scaled_prms: dict[str, np.ndarray] = PrivateAttr(default={})
    _reduce_index_cache: tuple[slice, ...] | None = PrivateAttr(default=None)

    _prm_names: ClassVar[list[str]] = []
    _prm_names_to_scale: ClassVar[list[str]] = []
    _modifier_names: ClassVar[list[str]] = [
        "lt_factor_by_year",
        "lt_factor_by_cohort",
        "hazard_factor_by_year",
        "removal_share_by_year",
    ]
    """Names of the arrays of lifetime factors and period effects."""

    @model_validator(mode="after")
    def _init_t(self) -> Self:
        if self.dims.letters[0] != self.time_letter:
            raise ValueError(
                f"Lifetime model expects time dimension to be the first dimension. "
                f"Lifetime model uses time_letter '{self.time_letter}', but first dimension is {self.dims.letters[0]}. "
                f"If the time dimension uses a different letter, set time_letter explicitly."
            )
        self._t = _UnevenTimeDim(dim=self.dims[self.time_letter])
        return self

    @model_validator(mode="after")
    def _cast_all(self) -> Self:
        for name in self._array_names:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, self._any_to_np(value, name=name))
        self.reset_cached_arrays()
        return self

    def _any_to_np(self, prm_in: LifetimeArrayType, name: str) -> np.ndarray:
        if isinstance(prm_in, FlodymArray):
            if not all(dim in self.dims for dim in prm_in.dims):
                raise ValueError(
                    f"Dimensions of parameter {name} (now {prm_in.dims.letters}) must be a subset of lifetime model dims {self.dims.letters}."
                )
            prm_out = prm_in.cast_to(target_dims=self.dims).values
        elif isinstance(prm_in, np.ndarray):
            prm_out = np.ndarray(self.shape)
            try:
                prm_out[...] = prm_in
            except ValueError:
                raise ValueError(
                    f"Parameter {name} (shape {prm_in.shape}) has incompatible dimensions with lifetime model shape ({self.shape})"
                )
        elif isinstance(prm_in, Number):
            prm_out = np.ndarray(self.shape)
            prm_out[...] = prm_in
        else:
            raise TypeError(
                f"Parameter {name} must be a FlodymArray, np.ndarray, or number, but is {type(prm_in)}"
            )
        return prm_out

    def _check_prms_set(self) -> None:
        unset_prms = [prm_name for prm_name in self._prm_names if getattr(self, prm_name) is None]
        if unset_prms:
            raise ValueError(f"Lifetime parameters {unset_prms} must be set before use.")

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the lifetime model dimensions."""
        return self.dims.shape

    @property
    def _n_t(self) -> int:
        return self._t.dim.len

    @property
    def _shape_cohort(self) -> tuple[int, ...]:
        return (self._n_t,) + self.shape

    @property
    def _shape_no_t(self) -> tuple[int, ...]:
        return tuple(list(self.shape)[1:])

    @property
    def _reduce_index(self) -> tuple[slice, ...]:
        """Index reducing arrays to length 1 along non-time axes without parameter variation.

        The survival is computed with the reduced shape and broadcast to the full shape in the
        public properties, which saves time if e.g. the parameters only vary along some of the
        dimensions.
        """
        if self._reduce_index_cache is None:
            arrays = [getattr(self, name) for name in self._array_names]
            arrays = [a for a in arrays if a is not None]
            index = [slice(None)]
            for axis in range(1, len(self.shape)):
                varies = any(not np.all(a == a.take([0], axis=axis)) for a in arrays)
                index.append(slice(None) if varies else slice(0, 1))
            self._reduce_index_cache = tuple(index)
        return self._reduce_index_cache

    def _reduced(self, a: np.ndarray) -> np.ndarray:
        """View of an array of the lifetime model shape with the reduced shape."""
        return a[self._reduce_index]

    @property
    def _reduced_shape_no_t(self) -> tuple[int, ...]:
        return tuple(
            n if s == slice(None) else 1 for n, s in zip(self._shape_no_t, self._reduce_index[1:])
        )

    def _broadcast(self, a: np.ndarray) -> np.ndarray:
        """Read-only view of a reduced array by year and age-cohort, broadcast to the full shape."""
        return np.broadcast_to(a, self._shape_cohort)

    @property
    def _array_names(self) -> list[str]:
        """Names of all arrays of the lifetime model shape on which the survival depends."""
        return self._prm_names + self._modifier_names

    @property
    def prms(self) -> Mapping[str, np.ndarray | None]:
        """Read-only mapping of lifetime parameter names to their values.

        The values are numpy arrays (None if not set). Do NOT modify the values in this mapping
        (lifetime_model.prms["mean"][...] = 10) to update the lifetime model parameters. Use direct
        assignment instead, e.g. lifetime_model.mean = 10.
        """
        return MappingProxyType({prm_name: getattr(self, prm_name) for prm_name in self._prm_names})

    def _get_prm_names_to_scale(self) -> list[str]:
        """Names of the parameters which are changed by the lifetime factors."""
        return self._prm_names_to_scale

    def _scale_prm(self, value: np.ndarray, factor: np.ndarray) -> np.ndarray:
        """Returns the parameter value for a lifetime extended by factor.

        Must return a new array. The simple multiplication is overridden for parameters which are
        not proportional to the lifetime, such as in Leaching.
        """
        return value * factor

    def _scale_prms(self) -> None:
        names_to_scale = self._get_prm_names_to_scale()
        for prm_name in self._prm_names:
            value = self._reduced(getattr(self, prm_name))
            if self.lt_factor_by_cohort is not None and prm_name in names_to_scale:
                value = self._scale_prm(value, self._reduced(self.lt_factor_by_cohort))
            self._scaled_prms[prm_name] = value

    @property
    def _has_period_effects(self) -> bool:
        return self.hazard_factor_by_year is not None or self.removal_share_by_year is not None

    @property
    def cumulative_survival(self) -> np.ndarray:
        """Cumulative survival by year and age-cohort.

        cum_surv(t,c) is the share of the inflow of cohort c still present at the end of year t.
        Zero for c > t. Shape (time, cohort, other dims). Read-only.
        """
        return self._broadcast(self._get_reduced_cumulative_survival())

    def _get_reduced_cumulative_survival(self) -> np.ndarray:
        if self._cumulative_survival is None:
            if self._has_period_effects:
                self._cumulative_survival = self._cumulative_from_conditional_survival(
                    self._get_reduced_conditional_survival()
                )
            else:
                self._cumulative_survival = self._compute_base_cumulative_survival()
        return self._cumulative_survival

    @property
    def sf(self) -> np.ndarray:
        """Deprecated. Use :py:attr:`cumulative_survival` instead."""
        warnings.warn(
            "sf is deprecated since v2.0.0. It will be removed in v3.0.0. "
            "Use cumulative_survival instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.cumulative_survival

    def _compute_base_cumulative_survival(self) -> np.ndarray:
        """Cumulative survival of the lifetime distribution.

        Includes lifetime factors, but not period effects. Has the reduced shape.
        """
        self._check_prms_set()
        cum_surv = np.zeros((self._n_t, self._n_t) + self._reduced_shape_no_t)
        self._scale_prms()
        self._quad_points, self._quad_weights = self._get_quad_points_and_weights()
        if self.lt_factor_by_year is not None:
            self._compute_cumulative_survival_with_nurture(cum_surv)
        else:
            self._compute_cumulative_survival(cum_surv)
        return cum_surv

    @property
    def outflow_pmf(self) -> np.ndarray:
        """Probability mass function of the outflow time.

        outflow_pmf(t,c) is the share of the inflow of cohort c leaving the stock during year t.
        It is a share per time step, i.e. the lifetime density integrated over the time step, not
        a density. It equals cum_surv(t-1,c) - cum_surv(t,c) for c < t, 1 - cum_surv(t,t) for
        c = t, and zero for c > t. Shape (time, cohort, other dims). Read-only.
        """
        if self._outflow_pmf is None:
            self._outflow_pmf = self._compute_outflow_pmf(self._get_reduced_cumulative_survival())
        return self._broadcast(self._outflow_pmf)

    @property
    def pdf(self) -> np.ndarray:
        """Deprecated. Use :py:attr:`outflow_pmf` instead."""
        warnings.warn(
            "pdf is deprecated since v2.0.0. It will be removed in v3.0.0. "
            "Use outflow_pmf instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.outflow_pmf

    @property
    def conditional_survival(self) -> np.ndarray:
        """Conditional one-step survival by year and age-cohort.

        cond_surv(t,c) is the share of cohort c present at the end of year t-1 which is still
        present at the end of year t. For the newest cohort (c = t), it is the share of the inflow
        during year t which is still present at the end of year t.
        It equals cum_surv(t,c) / cum_surv(t-1,c) for c < t, cum_surv(t,t) for c = t, and zero
        for c > t. Period effects are applied to it, and the cumulative survival is then computed
        from it. Shape (time, cohort, other dims). Read-only.
        """
        return self._broadcast(self._get_reduced_conditional_survival())

    def _get_reduced_conditional_survival(self) -> np.ndarray:
        if self._conditional_survival is None:
            if self._has_period_effects:
                cond_surv = self._conditional_from_cumulative_survival(
                    self._compute_base_cumulative_survival()
                )
                self._apply_period_effects(cond_surv)
                self._conditional_survival = cond_surv
            else:
                self._conditional_survival = self._conditional_from_cumulative_survival(
                    self._get_reduced_cumulative_survival()
                )
        return self._conditional_survival

    def _conditional_from_cumulative_survival(self, cum_surv: np.ndarray) -> np.ndarray:
        cond_surv = np.zeros_like(cum_surv)
        # The share of cohort c present at the end of year t-1 which survives year t is
        # cum_surv(t,c) / cum_surv(t-1,c). If cum_surv(t-1,c) is zero, nothing is left to survive.
        cum_surv_prev = cum_surv[:-1, ...]
        np.divide(cum_surv[1:, ...], cum_surv_prev, out=cond_surv[1:, ...], where=cum_surv_prev > 0)
        # The newest cohort enters during year t, so its conditional survival is cum_surv(t,t)
        t_diag_indices = self._t_diag_indices
        cond_surv[t_diag_indices] = cum_surv[t_diag_indices]
        return cond_surv

    def _cumulative_from_conditional_survival(self, cond_surv: np.ndarray) -> np.ndarray:
        """cum_surv(t,c) is the product of cond_surv(t',c) over t' = c...t, and zero for c > t."""
        upper_triangle = np.triu(np.ones((self._n_t, self._n_t), dtype=bool), k=1)
        upper_triangle = upper_triangle[(...,) + (np.newaxis,) * len(self._shape_no_t)]
        cum_surv = np.cumprod(np.where(upper_triangle, 1.0, cond_surv), axis=0)
        cum_surv[np.broadcast_to(upper_triangle, cum_surv.shape)] = 0.0
        return cum_surv

    def _apply_period_effects(self, cond_surv: np.ndarray) -> None:
        """Applies hazard_factor_by_year and removal_share_by_year to cond_surv in place.

        cond_surv is the conditional survival. Row t of cond_surv contains the values of all
        age-cohorts in year t.
        """
        if self.hazard_factor_by_year is not None:
            hazard_factor = self._reduced(self.hazard_factor_by_year)[:, np.newaxis, ...]
            if np.min(hazard_factor) < 0:
                raise ValueError("hazard_factor_by_year must not be negative.")
            # zero values stay zero, also for a hazard factor of zero
            np.power(cond_surv, hazard_factor, out=cond_surv, where=cond_surv > 0)
        if self.removal_share_by_year is not None:
            share = self._reduced(self.removal_share_by_year)
            if np.min(share) < 0 or np.max(share) > 1:
                raise ValueError("removal_share_by_year must be between 0 and 1.")
            t_diag_indices = self._t_diag_indices
            newest_cohort_cond_surv = cond_surv[t_diag_indices].copy()
            cond_surv *= 1.0 - share[:, np.newaxis, ...]
            # The newest cohort is only affected if it enters before the removal, which is
            # assumed to be evenly distributed over the time step.
            quad_points, quad_weights = self._get_quad_points_and_weights()
            exposed_fraction = sum(w * (1.0 - p) for p, w in zip(quad_points, quad_weights))
            cond_surv[t_diag_indices] = newest_cohort_cond_surv * (1.0 - exposed_fraction * share)

    @property
    def _t_diag_indices(self) -> tuple[np.ndarray | slice, ...]:
        """Indices of the main diagonal of the time and cohort axes of a (t, c, ...) array."""
        return np.diag_indices(self._n_t) + (slice(None),) * len(self._shape_no_t)

    def _get_dims_cohort(self, cohort_dim: Dimension | None = None) -> DimensionSet:
        """Dimensions (time, cohort, other dimensions) of arrays by year and age-cohort.

        Args:
            cohort_dim: Cohort dimension with the same items as the time dimension. If not given,
                a dimension with name "cohort" and letter "c" is used.
        """
        time_dim = self.dims[self.time_letter]
        if cohort_dim is None:
            cohort_dim = Dimension(
                name="cohort", letter="c", items=time_dim.items, dtype=time_dim.dtype
            )
        if cohort_dim.letter in self.dims.letters or cohort_dim.name in self.dims.names:
            raise ValueError(
                f"Cohort dimension (letter '{cohort_dim.letter}', name '{cohort_dim.name}') must "
                "not be part of the lifetime model dimensions. Pass a different cohort_dim."
            )
        if cohort_dim.items != time_dim.items:
            raise ValueError("Cohort dimension must have the same items as the time dimension.")
        return time_dim + cohort_dim + self.dims.drop(self.time_letter)

    def get_cumulative_survival(self, cohort_dim: Dimension | None = None) -> FlodymArray:
        """:py:attr:`cumulative_survival` as FlodymArray with dims (time, cohort, other dims).

        Args:
            cohort_dim: Cohort dimension with the same items as the time dimension. If not given,
                a dimension with name "cohort" and letter "c" is used.
        """
        dims = self._get_dims_cohort(cohort_dim)
        return FlodymArray(
            dims=dims, values=self.cumulative_survival.copy(), name="cumulative_survival"
        )

    def get_conditional_survival(self, cohort_dim: Dimension | None = None) -> FlodymArray:
        """:py:attr:`conditional_survival` as FlodymArray with dims (time, cohort, other dims).

        Args:
            cohort_dim: Cohort dimension with the same items as the time dimension. If not given,
                a dimension with name "cohort" and letter "c" is used.
        """
        dims = self._get_dims_cohort(cohort_dim)
        return FlodymArray(
            dims=dims, values=self.conditional_survival.copy(), name="conditional_survival"
        )

    def get_outflow_pmf(self, cohort_dim: Dimension | None = None) -> FlodymArray:
        """:py:attr:`outflow_pmf` as FlodymArray with dimensions (time, cohort, other dimensions).

        Args:
            cohort_dim: Cohort dimension with the same items as the time dimension. If not given,
                a dimension with name "cohort" and letter "c" is used.
        """
        dims = self._get_dims_cohort(cohort_dim)
        return FlodymArray(dims=dims, values=self.outflow_pmf.copy(), name="outflow_pmf")

    def get_mean_lifetime_by_cohort(self, tail: Literal["constant_rate", "none"]) -> FlodymArray:
        """Mean time the items of each age-cohort remain in the stock.

        It includes the effects of lifetime factors, period effects and competing risks.
        The result has the dimensions of the lifetime model, where the time dimension denotes the
        age-cohort, i.e. the time step of inflow. It is computed by integrating the cumulative
        survival over time, interpolating linearly between the inflow time and the ends of the time
        steps. For smooth distributions, the resulting relative error is roughly
        (time step length / mean lifetime) ** 2 / 12, e.g. 0.1 % for yearly time steps and a mean
        lifetime of 10 years.

        Survival beyond the end of the time horizon is unknown, so the result for cohorts with
        a large share surviving at the end of the time horizon depends strongly on ``tail``.
        This share can be checked with :py:meth:`get_cumulative_survival`.

        Args:
            tail: Treatment of survival beyond the end of the time horizon.
                ``"constant_rate"``: extrapolate assuming that the conditional survival per unit
                of time of the last time step persists. This is exact for an exponential lifetime,
                but can be far off for other distributions, especially for young cohorts. If nothing
                leaves the stock in the last time step, the result is infinite.
                ``"none"``: ignore survival beyond the time horizon; the result is then a lower
                bound.
        """
        if tail not in ("constant_rate", "none"):
            raise ValueError(f"tail must be 'constant_rate' or 'none', not '{tail}'.")
        cum_surv = self.cumulative_survival
        expand = (slice(None),) + (np.newaxis,) * len(self._shape_no_t)
        interval_lengths = self._t.interval_lengths
        quad_points, quad_weights = self._get_quad_points_and_weights()
        mean_inflow_point = sum(w * p for p, w in zip(quad_points, quad_weights))
        # time from inflow to the end of the inflow time step, by cohort
        first_lengths = (1.0 - mean_inflow_point) * interval_lengths
        newest_cohort_cum_surv = cum_surv[self._t_diag_indices]
        mean_lifetime = (1.0 + newest_cohort_cum_surv) / 2 * first_lengths[expand]
        # later time steps t > c: trapezoid between cum_surv(t-1,c) and cum_surv(t,c)
        step_lengths = interval_lengths[1:][expand + (np.newaxis,)]
        trapezoids = (cum_surv[:-1] + cum_surv[1:]) / 2 * step_lengths
        later_steps = np.tril(np.ones((self._n_t - 1, self._n_t), dtype=bool))
        later_steps = later_steps[(...,) + (np.newaxis,) * len(self._shape_no_t)]
        mean_lifetime += np.where(later_steps, trapezoids, 0.0).sum(axis=0)
        if tail == "constant_rate":
            cum_surv_last = cum_surv[-1]
            exposure_last = np.full(self._n_t, interval_lengths[-1])
            exposure_last[-1] = first_lengths[-1]
            with np.errstate(divide="ignore", invalid="ignore"):
                hazard = -np.log(self.conditional_survival[-1]) / exposure_last[expand]
                tail_integral = np.where(cum_surv_last > 0, cum_surv_last / hazard, 0.0)
            mean_lifetime += tail_integral
        return FlodymArray(dims=self.dims, values=mean_lifetime, name="mean_lifetime_by_cohort")

    def set_prms(self, *args: LifetimeArrayType, **kwargs: LifetimeArrayType) -> None:
        """Set parameters of the lifetime distribution, like mean and standard deviation.

        Parameters can be set either by positional arguments or by keyword arguments, but not both.

        Deprecated. Use direct assignment instead, e.g. lifetime_model.mean = 10.

        Args:
            *args: Positional arguments for parameters, in the order of self.prms.
            **kwargs: Keyword arguments for parameters, with parameter names as keys.
        """
        warnings.warn(
            "set_prms is deprecated since v1.1.0. It will be removed in v3.0.0. "
            "Use direct assignment instead, e.g. lifetime_model.mean = 10.",
            DeprecationWarning,
            stacklevel=2,
        )

        if len(args) > 0 and len(kwargs) > 0:
            raise ValueError("Cannot set parameters with both positional and keyword arguments.")
        if len(args) > 0:
            if len(args) != len(self._prm_names):
                raise ValueError(
                    f"Expected {len(self._prm_names)} parameters, but got {len(args)}."
                )
            for prm_name, prm_value in zip(self._prm_names, args):
                setattr(self, prm_name, prm_value)
        elif len(kwargs) > 0:
            for prm_name, prm_value in kwargs.items():
                if prm_name not in self._prm_names:
                    raise ValueError(f"Unknown parameter: {prm_name}")
                setattr(self, prm_name, prm_value)
        self.reset_cached_arrays()

    def reset_cached_arrays(self) -> None:
        """Discards the cached cumulative survival, conditional survival and outflow pmf.

        They are then recomputed on the next access. This happens automatically when a field is
        assigned. Call it after modifying a parameter array in place, e.g.
        ``lifetime_model.mean[...] = 10``.
        """
        self._cumulative_survival = None
        self._outflow_pmf = None
        self._conditional_survival = None
        self._reduce_index_cache = None

    def _tile(self, a: np.ndarray) -> np.ndarray:
        """Tiles the input array a to the reduced shape, adding non-time dimensions.

        Args:
            a: either of shape (n_t,) or (n_t, n_t), where the second dimension
                corresponds to the age cohort

        Returns:
            The tiled array
        """
        index = (slice(None),) * a.ndim + (np.newaxis,) * len(self._shape_no_t)
        out = a[index]
        return np.tile(out, self._reduced_shape_no_t)

    def _quad_point_time(self, m: int | np.ndarray, quad_point: float) -> float | np.ndarray:
        """Returns the time point within inflow time period m, given the quadrature point eta."""
        return quad_point * self._t.bounds[m + 1] + (1 - quad_point) * self._t.bounds[m]

    def _remaining_ages(self, i_t: int, quad_point: float) -> np.ndarray:
        return self._tile(self._t.bounds[i_t + 1 :] - self._quad_point_time(i_t, quad_point))

    def _compute_cumulative_survival(self, cum_surv: np.ndarray) -> None:
        """Computes the cumulative survival without lifetime extension by year into cum_surv.

        cum_surv(t,c) is the share of the inflow of cohort c still present at the end of year t,
        i.e. the survival function of the lifetime distribution evaluated at the age at the end of
        year t, averaged over the inflow points within year c.
        For a lifetime of 0, it is also 0, meaning that the age-cohort leaves during the year of
        inflow.
        This is the only method for the inflow-driven model where the lifetime distribution
        directly enters the computation. All other stock variables are determined by mass balance.
        """
        for i_c in range(self._n_t):  # cohort index
            for quad_point, quad_weight in zip(self._quad_points, self._quad_weights):
                ages = self._remaining_ages(i_c, quad_point)
                cum_surv[i_c::, i_c, ...] += quad_weight * self._survival_by_cohort(ages, i_c)

    def _compute_cumulative_survival_with_nurture(self, cum_surv: np.ndarray) -> None:
        """Compute the cumulative survival with lifetime extension by year into cum_surv.

        Lifetime extension by year is the "nurture" approach in Krych et al. 2024.

        Math:
        sf_e = plain survival function with extended lifetime
        cum_surv = resulting combined cumulative survival based on previous survival
        cond_surv(t) = 1 - hazard_function = sf_e(t)/sf_e(t-1)
        cum_surv(t) = cum_surv(t-1) * cond_surv(t)
        cum_surv(t) = cum_surv(t-1) * sf_e(t)/sf_e(t-1)
        """
        factor = self._reduced(self.lt_factor_by_year)

        scaled_prms_orig = dict(self._scaled_prms)
        names_to_scale = self._get_prm_names_to_scale()
        for i_t in range(self._n_t):
            # scale such that mean and stddev are increased by lifetime extension factor
            for name in names_to_scale:
                # apply factor per time step - broadcast to all age cohorts
                # (The arrays in _scaled_prms can be the parameter arrays themselves, so
                # assign new arrays instead of writing into them)
                self._scaled_prms[name] = self._scale_prm(scaled_prms_orig[name], factor[i_t])
            # curr_survival calculates sf_e(t-1) ans sf_e(t) in one array
            # max(i_t, 1) omits prev time step for i_t = 0
            curr_time = self._t.bounds[max(i_t, 1) : i_t + 2]
            cohort_ids = np.arange(i_t + 1)
            curr_survival = np.zeros((len(curr_time), i_t + 1) + self._reduced_shape_no_t)
            for quad_point, quad_weight in zip(self._quad_points, self._quad_weights):
                cohort_time = self._quad_point_time(cohort_ids, quad_point)
                curr_ages = self._tile(curr_time[:, np.newaxis] - cohort_time[np.newaxis, :])
                curr_survival += quad_weight * self._survival_by_cohort(
                    curr_ages, slice(0, i_t + 1)
                )
            # main diagonal: cum_surv(t) = sf_e(t)
            cum_surv[i_t, i_t, ...] = curr_survival[-1, i_t, ...]
            if i_t > 0:
                sf_e_prev = curr_survival[0, :i_t, ...]
                sf_e = curr_survival[1, :i_t, ...]
                cum_surv_prev = cum_surv[i_t - 1, :i_t, ...]
                # for small lifetimes, sf_e_prev can be zero, which means that nothing has survived
                ratio = np.divide(sf_e, sf_e_prev, out=np.zeros_like(sf_e), where=sf_e_prev > 0)
                cum_surv[i_t, :i_t, ...] = cum_surv_prev * ratio

    def _get_quad_points_and_weights(self) -> tuple[list[float], list[float]]:
        """Returns the quadrature points and weights for the inflow time periods."""
        if not 1 <= self.n_pts_per_interval <= 10:
            raise ValueError(
                f"n_pts_per_interval must be between 1 and 10, got {self.n_pts_per_interval}."
            )
        if self.n_pts_per_interval > 1:
            points = [(x + 1) / 2 for x in GL_NODES[self.n_pts_per_interval]]
            weights = [w / 2 for w in GL_WEIGHTS[self.n_pts_per_interval]]
            return points, weights
        else:
            if self.inflow_at == "start":
                return [0], [1]
            elif self.inflow_at == "middle":
                return [0.5], [1]
            elif self.inflow_at == "end":
                return [1], [1]

    @abstractmethod
    def _survival_by_cohort(self, ages: np.ndarray, i_c: int | slice) -> np.ndarray:
        """Survival function.

        Args:
            ages: array of ages, i.e. difference of a time vector to the inflow time of cohort i_c.
                  If i_c is a slice, ages must have a cohort axis that
                  lines up with the parameters indexed by i_c, so implementations must broadcast
                  against it rather than assume a single cohort.
            i_c: cohort index or a slice of cohorts.
        """

    def _compute_outflow_pmf(self, cum_surv: np.ndarray) -> np.ndarray:
        """Computes the outflow pmf from the cumulative survival.

        outflow_pmf(t,c) is the probability that an item added to the stock in year c (cohort c)
        leaves during year t.
        """
        outflow_pmf = np.zeros_like(cum_surv)
        t_diag_indices = self._t_diag_indices
        outflow_pmf[t_diag_indices] = 1.0 - np.moveaxis(cum_surv.diagonal(0, 0, 1), -1, 0)
        for i_c in range(self._n_t):
            outflow_pmf[i_c + 1 :, i_c, ...] = -1 * np.diff(cum_surv[i_c:, i_c, ...], axis=0)
        return outflow_pmf


class FixedLifetime(LifetimeModel):
    """Fixed lifetime: the age-cohort leaves the stock when a certain age is reached.

    The age-cohort leaves in the model year when the age specified as 'Mean' is reached.
    """

    mean: LifetimeArrayType | None = None
    """The fixed lifetime, i.e. the age at which the age-cohort leaves the stock.
    """

    _prm_names: ClassVar[list[str]] = ["mean"]
    _prm_names_to_scale: ClassVar[list[str]] = ["mean"]

    def _survival_by_cohort(self, ages: np.ndarray, i_c: int | slice) -> np.ndarray:
        mean = self._scaled_prms["mean"]
        # Example: if lt is 3.5 years fixed, product will still be there after 0, 1, 2, and 3 years,
        # gone after 4 years.
        return (ages < mean[i_c, ...]).astype(int)


class ExponentialLifetime(LifetimeModel):
    """Exponentially distributed lifetime, i.e. a constant hazard rate 1 / mean.

    The share of the remaining stock leaving per time is the same for all ages.
    sf(age) = exp(-age / mean).
    """

    mean: LifetimeArrayType | None = None
    """Mean lifetime, equal to the inverse of the hazard rate.
    """

    _prm_names: ClassVar[list[str]] = ["mean"]
    _prm_names_to_scale: ClassVar[list[str]] = ["mean"]

    def _survival_by_cohort(self, ages: np.ndarray, i_c: int | slice) -> np.ndarray:
        mean = self._scaled_prms["mean"]
        if np.min(mean) <= 0:
            raise ValueError("mean must be greater than zero.")
        return np.exp(-ages / mean[i_c, ...])


class Leaching(LifetimeModel):
    """Loss of a fixed share of the remaining stock per unit of time (e.g. per year).

    The loss is independent of age, for example by leaching, corrosion or dissipation.
    sf(age) = (1 - loss_rate) ** age.
    This is an exponential lifetime with mean -1 / ln(1 - loss_rate).
    Often combined with another lifetime model in a :py:class:`CompetingRisksLifetime`.
    Lifetime factors f change the rate such that the mean lifetime is multiplied by f, i.e.
    1 - loss_rate is raised to the power 1 / f.
    """

    loss_rate: LifetimeArrayType | None = None
    """Share of the stock leaving per unit of time, between 0 and 1.
    """

    _prm_names: ClassVar[list[str]] = ["loss_rate"]
    _prm_names_to_scale: ClassVar[list[str]] = ["loss_rate"]

    def _scale_prm(self, value: np.ndarray, factor: np.ndarray) -> np.ndarray:
        return 1.0 - (1.0 - value) ** (1.0 / factor)

    def _survival_by_cohort(self, ages: np.ndarray, i_c: int | slice) -> np.ndarray:
        rate = self._scaled_prms["loss_rate"]
        if np.min(rate) < 0 or np.max(rate) > 1:
            raise ValueError("loss_rate must be between 0 and 1.")
        return (1.0 - rate[i_c, ...]) ** ages


class StandardDeviationLifetimeModel(LifetimeModel):
    """Base class for lifetime distributions parametrized by mean and standard deviation."""

    mean: LifetimeArrayType | None = None
    """Mean lifetime.
    """
    std: LifetimeArrayType | None = None
    """Standard deviation of lifetime. If relative_std is True, it is given relative to the
    mean, i.e. as coefficient of variation std / mean.
    """
    relative_std: bool = False
    """If True, std is interpreted relative to the mean. Lifetime factors then only multiply the
    mean, and the absolute standard deviation changes with it.
    """
    _prm_names: ClassVar[list[str]] = ["mean", "std"]
    _prm_names_to_scale: ClassVar[list[str]] = ["mean", "std"]

    def _get_prm_names_to_scale(self) -> list[str]:
        if self.relative_std:
            return ["mean"]
        return self._prm_names_to_scale

    def _mean_std(self, i_c: int | slice) -> tuple[np.ndarray, np.ndarray]:
        """Returns the scaled mean and absolute standard deviation for cohort(s) i_c."""
        mean = self._scaled_prms["mean"][i_c, ...]
        std = self._scaled_prms["std"][i_c, ...]
        if self.relative_std:
            std = std * mean
        return mean, std


class NormalLifetime(StandardDeviationLifetimeModel):
    """Normally distributed lifetime with mean and standard deviation.

    Watch out for nonzero values, for negative ages, no correction or truncation done here.
    NOTE: As normal distributions have nonzero pdf for negative ages,
    which are physically impossible, these outflow contributions can either be ignored (
    violates the mass balance) or allocated to the zeroth year of residence,
    the latter being implemented here, as the cumulative survival is evaluated at the end of
    the inflow year.
    As alternative, use lognormal or folded normal distribution options.
    """

    def _survival_by_cohort(self, ages: np.ndarray, i_c: int | slice) -> np.ndarray:
        if np.min(self._scaled_prms["mean"]) < 0:
            raise ValueError("mean must be greater than zero.")
        mean, std = self._mean_std(i_c)

        return scipy.stats.norm.sf(ages, loc=mean, scale=std)


class FoldedNormalLifetime(StandardDeviationLifetimeModel):
    """Folded normal distribution, cf. https://en.wikipedia.org/wiki/Folded_normal_distribution.

    NOTE: call this with the parameters of the normal distribution mu and sigma of curve
    BEFORE folding, curve after folding will have different mu and sigma.
    """

    def _survival_by_cohort(self, ages: np.ndarray, i_c: int | slice) -> np.ndarray:
        if np.min(self._scaled_prms["mean"]) < 0:
            raise ValueError("mean must be greater than zero.")
        mean, std = self._mean_std(i_c)

        return scipy.stats.foldnorm.sf(ages, mean / std, 0, scale=std)


class LogNormalLifetime(StandardDeviationLifetimeModel):
    """Lognormal distribution.

    Here, the mean and stddev of the lognormal curve, not those of the underlying normal
    distribution, need to be specified!
    Values chosen according to description on
    https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.stats.lognorm.html
    Same result as EXCEL function "=LOGNORM.VERT(x;LT_LN;SG_LN;TRUE)"
    """

    def _survival_by_cohort(self, ages: np.ndarray, i_c: int | slice) -> np.ndarray:
        mean, std = self._mean_std(i_c)
        mean_square = mean * mean
        std_square = std * std
        new_mean = np.log(mean_square / np.sqrt(mean_square + std_square))
        new_std = np.sqrt(np.log(1 + std_square / mean_square))
        # compute survival function
        sf_m = scipy.stats.lognorm.sf(ages, s=new_std, loc=0, scale=np.exp(new_mean))
        return sf_m


class WeibullLifetime(LifetimeModel):
    """Weibull distribution with standard definition of scale and shape parameters.

    For a parametrization with mean and standard deviation, see
    :py:class:`WeibullMeanStdLifetime`.
    """

    weibull_scale: LifetimeArrayType | None = None
    """Scale parameter of the Weibull distribution.
    """
    weibull_shape: LifetimeArrayType | None = None
    """Shape parameter of the Weibull distribution.
    """

    _prm_names: ClassVar[list[str]] = ["weibull_scale", "weibull_shape"]
    _prm_names_to_scale: ClassVar[list[str]] = ["weibull_scale"]

    def _survival_by_cohort(self, ages: np.ndarray, i_c: int | slice) -> np.ndarray:
        scale, shape = self._scaled_prms["weibull_scale"], self._scaled_prms["weibull_shape"]
        if np.min(shape) < 0:
            raise ValueError("Lifetime weibull_shape must be positive for Weibull distribution.")

        return scipy.stats.weibull_min.sf(
            ages,
            c=shape[i_c, ...],
            loc=0,
            scale=scale[i_c, ...],
        )


class WeibullMeanStdLifetime(StandardDeviationLifetimeModel):
    """Weibull distribution, parametrized by its mean and standard deviation.

    The Weibull shape parameter k is determined by the coefficient of variation std / mean via
    Gamma(1 + 2/k) / Gamma(1 + 1/k)^2 = 1 + (std / mean)^2,
    and the scale parameter by scale = mean / Gamma(1 + 1/k).
    Lifetime factors multiply mean and std alike, so they only change the scale parameter.
    """

    _weibull_shape: np.ndarray | None = PrivateAttr(default=None)

    _shape_bounds: ClassVar[tuple[float, float]] = (0.05, 1.0e4)
    """Range of Weibull shape parameters considered in the computation from mean and std."""

    def _scale_prms(self) -> None:
        super()._scale_prms()
        mean, std = self._reduced(self.mean), self._reduced(self.std)
        cv = std if self.relative_std else std / mean
        self._weibull_shape = self._weibull_shape_from_cv(cv)

    @classmethod
    def _weibull_shape_from_cv(cls, cv: np.ndarray) -> np.ndarray:
        """Solves Gamma(1 + 2/k) / Gamma(1 + 1/k)^2 = 1 + cv^2 for the Weibull shape k.

        Uses bisection in log(k). The left-hand side decreases monotonically with k.
        """

        def log_ratio(log_k: np.ndarray | float) -> np.ndarray | float:
            k_inv = np.exp(-log_k)
            return scipy.special.gammaln(1 + 2 * k_inv) - 2 * scipy.special.gammaln(1 + k_inv)

        target = np.log1p(cv**2)
        lower = np.full(cv.shape, np.log(cls._shape_bounds[0]))
        upper = np.full(cv.shape, np.log(cls._shape_bounds[1]))
        in_range = (target <= log_ratio(lower)) & (target >= log_ratio(upper))
        if not np.all(in_range):
            shape_max, shape_min = cls._shape_bounds[1], cls._shape_bounds[0]
            cv_min = np.sqrt(np.expm1(log_ratio(np.log(shape_max))))
            cv_max = np.sqrt(np.expm1(log_ratio(np.log(shape_min))))
            raise ValueError(
                f"Coefficient of variation std / mean must be between {cv_min:.2g} and "
                f"{cv_max:.2g} for WeibullMeanStdLifetime."
            )
        for _ in range(60):
            middle = (lower + upper) / 2
            shape_too_small = log_ratio(middle) > target
            lower = np.where(shape_too_small, middle, lower)
            upper = np.where(shape_too_small, upper, middle)
        return np.exp((lower + upper) / 2)

    def _survival_by_cohort(self, ages: np.ndarray, i_c: int | slice) -> np.ndarray:
        mean = self._scaled_prms["mean"][i_c, ...]
        shape = self._weibull_shape[i_c, ...]
        scale = mean / scipy.special.gamma(1 + 1 / shape)
        return scipy.stats.weibull_min.sf(ages, c=shape, loc=0, scale=scale)


class TabulatedLifetime(LifetimeModel):
    """Lifetime model with the cumulative survival given directly by year and age-cohort.

    The cumulative survival can come e.g. from statistics or from an external model.
    As no distribution is known, lifetime factors cannot be used. Period effects can be used.
    inflow_at and n_pts_per_interval are only used for removal_share_by_year and
    :py:meth:`get_mean_lifetime_by_cohort`, and should match the inflow time assumed in the
    given cumulative survival.
    """

    cumulative_survival_table: FlodymArray | np.ndarray | None = None
    """Share of the inflow of cohort c still present at the end of time step t.
    As FlodymArray, its dimensions must be the time dimension, a cohort dimension with the same
    items as the time dimension, and a subset of the other dimensions of the lifetime model, in
    any order. The cohort dimension is the one which is not part of the lifetime model.
    For example, the output of :py:meth:`get_cumulative_survival` of another lifetime model can be
    used.
    As np.ndarray, its shape must be broadcastable to (time, cohort, other dimensions).
    Values must be between 0 and 1, zero for cohorts after the time step (c > t), and must not
    increase over time for each cohort.
    """

    @model_validator(mode="after")
    def _cast_cumulative_survival_table(self) -> Self:
        if self.lt_factor_by_year is not None or self.lt_factor_by_cohort is not None:
            raise ValueError("Lifetime factors cannot be used with TabulatedLifetime.")
        value = self.cumulative_survival_table
        if value is None:
            return self
        if isinstance(value, FlodymArray):
            cohort_letters = [
                letter for letter in value.dims.letters if letter not in self.dims.letters
            ]
            if len(cohort_letters) != 1 or self.time_letter not in value.dims.letters:
                raise ValueError(
                    f"Dimensions of cumulative_survival_table ({value.dims.letters}) must be the "
                    "time dimension, one cohort dimension, and a subset of the lifetime model "
                    f"dimensions ({self.dims.letters})."
                )
            dims_cohort = self._get_dims_cohort(value.dims[cohort_letters[0]])
            values = value.cast_values_to(dims_cohort).astype(float)
        elif isinstance(value, np.ndarray):
            values = np.empty(self._shape_cohort)
            try:
                values[...] = value
            except ValueError:
                raise ValueError(
                    f"cumulative_survival_table (shape {value.shape}) cannot be broadcast to shape "
                    f"{self._shape_cohort} (time, cohort, other dimensions)."
                )
        else:
            raise TypeError(
                "cumulative_survival_table must be a FlodymArray or np.ndarray, but is "
                f"{type(value)}."
            )
        self._check_cumulative_survival_table(values)
        object.__setattr__(self, "cumulative_survival_table", values)
        self.reset_cached_arrays()
        return self

    def _check_cumulative_survival_table(self, values: np.ndarray) -> None:
        tolerance = 1e-12
        if np.min(values) < -tolerance or np.max(values) > 1 + tolerance:
            raise ValueError("cumulative_survival_table must be between 0 and 1.")
        no_t_axes = (np.newaxis,) * len(self._shape_no_t)
        future_cohorts = np.triu(np.ones((self._n_t, self._n_t), dtype=bool), k=1)
        if np.any(np.where(future_cohorts[(...,) + no_t_axes], values, 0.0) != 0):
            raise ValueError(
                "cumulative_survival_table must be zero for cohorts after the time step (c > t)."
            )
        # change from t-1 to t, for t > c
        later_steps = np.tril(np.ones((self._n_t - 1, self._n_t), dtype=bool))
        increase = np.where(later_steps[(...,) + no_t_axes], np.diff(values, axis=0), 0.0)
        if np.max(increase, initial=0.0) > tolerance:
            raise ValueError(
                "cumulative_survival_table must not increase over time for an age-cohort."
            )

    @property
    def _reduce_index(self) -> tuple[slice, ...]:
        """No reduction, as the cumulative survival is given with the full shape."""
        return (slice(None),) * len(self.shape)

    def _check_prms_set(self) -> None:
        if self.cumulative_survival_table is None:
            raise ValueError("Lifetime parameter cumulative_survival_table must be set before use.")

    def _compute_base_cumulative_survival(self) -> np.ndarray:
        self._check_prms_set()
        return np.clip(self.cumulative_survival_table, 0.0, 1.0)

    def _survival_by_cohort(self, ages: np.ndarray, i_c: int | slice) -> np.ndarray:
        raise NotImplementedError("TabulatedLifetime uses the given cumulative survival directly.")


class CompetingRisksLifetime(LifetimeModel):
    """Combination of several lifetime models describing independent causes of leaving the stock.

    Examples of such causes are a product lifetime and a constant material loss
    (see :py:class:`Leaching`). An item remains in the stock only if it survives all causes, so
    the cumulative survival, and thereby the conditional survival, of the models are multiplied.

    dims and time_letter are taken from the first model if not given.
    Lifetime factors, inflow_at and n_pts_per_interval cannot be set here; set them in the
    individual models instead. Period effects (hazard_factor_by_year, removal_share_by_year)
    can be set here, and then apply to all causes together, or in the individual models.
    Changes to the parameters of the individual models are taken into account in the next
    access of cumulative_survival, conditional_survival or outflow_pmf.
    """

    models: list[LifetimeModel]
    """Lifetime models to combine. All must have the same dimensions."""

    _model_cumulative_survivals: tuple[np.ndarray, ...] | None = PrivateAttr(default=None)
    """Survival factors of the individual models used in the cached arrays, to detect changes."""

    _fields_of_models: ClassVar[list[str]] = [
        "lt_factor_by_year",
        "lt_factor_by_cohort",
        "inflow_at",
        "n_pts_per_interval",
    ]

    @model_validator(mode="before")
    @classmethod
    def _dims_from_models(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("models"):
            first_model = data["models"][0]
            if isinstance(first_model, LifetimeModel):
                data = {"dims": first_model.dims, "time_letter": first_model.time_letter, **data}
        return data

    @model_validator(mode="after")
    def _check_models(self) -> Self:
        if len(self.models) < 2:
            raise ValueError("CompetingRisksLifetime needs at least two lifetime models.")
        fields_set_here = sorted(self.model_fields_set & set(self._fields_of_models))
        if fields_set_here:
            raise ValueError(
                f"{fields_set_here} cannot be set in CompetingRisksLifetime. "
                "Set them in the individual lifetime models instead."
            )
        time_items = self.dims[self.time_letter].items
        for model in self.models:
            if model.dims.letters != self.dims.letters or model.dims.shape != self.dims.shape:
                raise ValueError(
                    f"Dimensions of all lifetime models ({model.dims.letters}, shape "
                    f"{model.dims.shape}) must match ({self.dims.letters}, shape {self.dims.shape})."
                )
            if model.time_letter != self.time_letter or model.dims[model.time_letter].items != (
                time_items
            ):
                raise ValueError("Time dimension of all lifetime models must match.")
        return self

    @property
    def cumulative_survival(self) -> np.ndarray:
        """Cumulative survival, see :py:attr:`LifetimeModel.cumulative_survival`.

        Recomputed if the cumulative survival of any of the individual models has changed.
        """
        self._reset_if_models_changed()
        return super().cumulative_survival

    @property
    def outflow_pmf(self) -> np.ndarray:
        """Outflow pmf, see :py:attr:`LifetimeModel.outflow_pmf`.

        Recomputed if the cumulative survival of any of the individual models has changed.
        """
        self._reset_if_models_changed()
        return super().outflow_pmf

    @property
    def conditional_survival(self) -> np.ndarray:
        """Conditional survival, see :py:attr:`LifetimeModel.conditional_survival`.

        Recomputed if the cumulative survival of any of the individual models has changed.
        """
        self._reset_if_models_changed()
        return super().conditional_survival

    def reset_cached_arrays(self) -> None:
        """Discards the cached arrays, see :py:meth:`LifetimeModel.reset_cached_arrays`."""
        super().reset_cached_arrays()
        self._model_cumulative_survivals = None

    def _reset_if_models_changed(self) -> None:
        """Resets the cached arrays if the cumulative survival of any model has changed.

        Changes are detected relative to the time the cached arrays were computed, e.g. because
        the parameters of a model were changed.
        """
        model_cum_survs = tuple(model.cumulative_survival for model in self.models)
        previous = self._model_cumulative_survivals
        if previous is None or len(model_cum_survs) != len(previous):
            changed = True
        else:
            changed = any(new is not old for new, old in zip(model_cum_survs, previous))
        if changed:
            self.reset_cached_arrays()
            self._model_cumulative_survivals = model_cum_survs

    @property
    def _reduce_index(self) -> tuple[slice, ...]:
        """No reduction, as the cumulative survival is computed from the arrays of the models."""
        return (slice(None),) * len(self.shape)

    def _check_prms_set(self) -> None:
        for model in self.models:
            model._check_prms_set()

    def _compute_base_cumulative_survival(self) -> np.ndarray:
        cum_surv = self._model_cumulative_survivals[0].copy()
        for model_cum_surv in self._model_cumulative_survivals[1:]:
            cum_surv *= model_cum_surv
        return cum_surv

    def _get_quad_points_and_weights(self) -> tuple[list[float], list[float]]:
        """Inflow time within the time step, used for removal_share_by_year.

        Taken from the first model, as it cannot be set in CompetingRisksLifetime.
        """
        return self.models[0]._get_quad_points_and_weights()

    def _survival_by_cohort(self, ages: np.ndarray, i_c: int | slice) -> np.ndarray:
        raise NotImplementedError(
            "CompetingRisksLifetime computes its cumulative survival from the individual models."
        )
