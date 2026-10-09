"""Home to various lifetime models, for use in dynamic stock modelling."""

from abc import abstractmethod
from types import MappingProxyType
import numpy as np
import scipy.stats
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, PrivateAttr, model_validator
from typing import ClassVar, TypeAlias, Literal, SupportsFloat
from numbers import Number
import warnings

# from scipy.special import gammaln, logsumexp
# from scipy.optimize import root_scalar

from .dimensions import DimensionSet, Dimension
from .flodym_arrays import FlodymArray
from .gauss_lobatto import gl_nodes, gl_weights

LifetimeArrayType: TypeAlias = FlodymArray | np.ndarray | SupportsFloat


class UnevenTimeDim(PydanticBaseModel):
    dim: Dimension
    _bounds: np.ndarray = None

    @property
    def bounds(self):
        if self._bounds is None:
            self.compute_t_bounds()
        return self._bounds

    @property
    def interval_lengths(self):
        """Returns the length of the time intervals, i.e. the difference between the bounds."""
        return np.diff(self.bounds)

    def compute_t_bounds(self):
        middle = (np.array(self.dim.items[:-1]) + np.array(self.dim.items[1:])) / 2.0
        self._bounds = np.concatenate(
            (
                [middle[0] - (middle[1] - middle[0])],
                middle,
                [middle[-1] + (middle[-1] - middle[-2])],
            )
        )


class LifetimeModel(PydanticBaseModel):
    """Contains shared functionality across the various lifetime models."""

    model_config = ConfigDict(
        validate_assignment=True, arbitrary_types_allowed=True, extra="forbid"
    )

    dims: DimensionSet
    time_letter: str = "t"
    inflow_at: Literal["start", "middle", "end"] = "middle"
    """If no quadrature is used, all inflow happens at one point in time, either at the beginning
    of the time period (start), in the middle (middle) or at the end (end).
    """
    n_pts_per_interval: int = 1
    """Inflow into the stock is in reality quite uniform over time, while survival factors only
    take into account a single point in time. This can be alleviated here by using numerical integration
    over each inflow time period with n points. A value of 1 means that the inflow
    is only evaluated once, depending on the inflow_at parameter. This is the default and should be used
    for long life times, e.g. >> 1 year.
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
    _sf: np.ndarray | None = PrivateAttr(default=None)
    _pdf: np.ndarray | None = PrivateAttr(default=None)
    _t: UnevenTimeDim | None = PrivateAttr(default=None)
    _quad_points: list[float] | None = PrivateAttr(default=None)
    """Intra-time step quadrature points for numerical integration of survival factors."""
    _quad_weights: list[float] | None = PrivateAttr(default=None)
    """Intra-time step quadrature weights for numerical integration of survival factors."""
    _scaled_prms: dict[str, np.ndarray] = PrivateAttr(default={})

    _prm_names: ClassVar[list[str]] = []
    _prm_names_to_scale: ClassVar[list[str]] = []

    @model_validator(mode="after")
    def _init_t(self):
        if self.dims.letters[0] != self.time_letter:
            raise ValueError(
                f"Lifetime model expects time dimension to be the first dimension. "
                f"Lifetime model uses time_letter '{self.time_letter}', but first dimension is {self.dims.letters[0]}. "
                f"If the time dimension uses a different letter, set time_letter explicitly."
            )
        self._t = UnevenTimeDim(dim=self.dims[self.time_letter])
        return self

    @model_validator(mode="after")
    def _cast_all(self):
        lt_factors = ["lt_factor_by_year", "lt_factor_by_cohort"]
        for prm_name in self._prm_names + lt_factors:
            prm_value = getattr(self, prm_name)
            if prm_value is not None:
                casted = self._any_to_np(prm_value, name=prm_name)
                object.__setattr__(self, prm_name, casted)
        self.reset_cached_arrays()
        return self

    def _any_to_np(self, prm_in: LifetimeArrayType, name: str):
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
                    f"Parameter {name} (shape {prm_in.shape}) has incompatible dimensions with lifetime model shape ({self._shape})"
                )
        elif isinstance(prm_in, Number):
            prm_out = np.ndarray(self.shape)
            prm_out[...] = prm_in
        else:
            raise ValueError(
                f"Parameter {name} must be a FlodymArray, np.ndarray, or number, but is {type(prm_in)}"
            )
        return prm_out

    def _check_prms_set(self):
        unset_prms = [prm_name for prm_name in self._prm_names if getattr(self, prm_name) is None]
        if unset_prms:
            raise ValueError(f"Lifetime parameters {unset_prms} must be set before use.")

    @property
    def shape(self):
        return self.dims.shape

    @property
    def _n_t(self):
        return self._t.dim.len

    @property
    def _shape_cohort(self):
        return (self._n_t,) + self.shape

    @property
    def _shape_no_t(self):
        return tuple(list(self.shape)[1:])

    @property
    def prms(self) -> dict[str, np.ndarray]:
        """Dictionary of lifetime parameters, with parameter names as keys and parameter values as
        numpy arrays. Do NOT modify the values in this dictionary (lifetime_model.prms["mean"] = 10)
        to update the lifetime model parameters. Use direct assignment instead, e.g.
        lifetime_model.mean = 10.
        """
        return MappingProxyType({prm_name: getattr(self, prm_name) for prm_name in self._prm_names})

    def _scale_prms(self):
        if self.lt_factor_by_cohort is None:
            self._scaled_prms = dict(self.prms)
            return
        for prm_name in self._prm_names:
            if prm_name in self._prm_names_to_scale:
                self._scaled_prms[prm_name] = getattr(self, prm_name) * self.lt_factor_by_cohort
            else:
                self._scaled_prms[prm_name] = getattr(self, prm_name)

    @property
    def sf(self):
        if self._sf is None:
            self._sf = np.zeros(self._shape_cohort)
            self._check_prms_set()
            self._scale_prms()
            self._quad_points, self._quad_weights = self._get_quad_points_and_weights()
            if self.lt_factor_by_year is not None:
                self._compute_survival_factor_with_nurture()
            else:
                self.compute_survival_factor()
        return self._sf

    @property
    def pdf(self):
        if self._pdf is None:
            self._pdf = np.zeros(self._shape_cohort)
            self._compute_outflow_pdf()
        return self._pdf

    def _tile(self, a: np.ndarray) -> np.ndarray:
        """tiles the input array a to the shape of the lifetime model, by adding non-time dimensions

        Args:
            a (np.ndarray): either of shape (n_t,) or (n_t, n_t), where the second dimension
                corresponds to the age cohort

        Returns:
            np.ndarray: the tiled array
        """
        index = (slice(None),) * a.ndim + (np.newaxis,) * len(self._shape_no_t)
        out = a[index]
        return np.tile(out, self._shape_no_t)

    def _quad_point_time(self, m, quad_point):
        """Returns the time point within the inflow time period m, given the quadrature point eta."""
        return quad_point * self._t.bounds[m + 1] + (1 - quad_point) * self._t.bounds[m]

    def _remaining_ages(self, i_t, quad_point):
        return self._tile(self._t.bounds[i_t + 1 :] - self._quad_point_time(i_t, quad_point))

    def compute_survival_factor(self):
        """Survival table self.sf(m,n) denotes the share of an inflow in year n (age-cohort) still
        present at the end of year m (after m-n years).
        The computation is self.sf(m,n) = ProbDist.sf(m-n), where ProbDist is the appropriate
        scipy function for the lifetime model chosen.
        For lifetimes 0 the sf is also 0, meaning that the age-cohort leaves during the same year
        of the inflow.
        The method compute outflow_sf returns an array year-by-cohort of the surviving fraction of
        a flow added to stock in year m (aka cohort m) in in year n. This value equals sf(n,m).
        This is the only method for the inflow-driven model where the lifetime distribution directly
        enters the computation.
        All other stock variables are determined by mass balance.
        The shape of the output sf array is NoofYears * NoofYears,
        and the meaning is years by age-cohorts.
        The method does nothing if the sf alreay exists.
        For example, sf could be assigned to the dynamic stock model from an exogenous computation
        to save time.
        """
        for i_c in range(0, self._n_t):  # cohort index
            for quad_point, quad_weight in zip(self._quad_points, self._quad_weights):
                t = self._remaining_ages(i_c, quad_point)
                self._sf[i_c::, i_c, ...] += quad_weight * self._survival_by_cohort(t, i_c)

    def _get_quad_points_and_weights(self):
        """Returns the quadrature points and weights for the inflow time periods."""
        if self.n_pts_per_interval > 10:
            raise ValueError("quad_order must be between 0 and 9.")
        if self.n_pts_per_interval > 1:
            points = [(x + 1) / 2 for x in gl_nodes[self.n_pts_per_interval]]
            weights = [w / 2 for w in gl_weights[self.n_pts_per_interval]]
            return points, weights
        else:
            if self.inflow_at == "start":
                return [0], [1]
            elif self.inflow_at == "middle":
                return [0.5], [1]
            elif self.inflow_at == "end":
                return [1], [1]

    @abstractmethod
    def _survival_by_cohort(self, m, **kwargs):
        """Survival function at ages m.

        The second argument, i_c, is a cohort index or a slice of cohorts. With a slice, m has a
        cohort axis that lines up with the parameters indexed by i_c, so implementations must
        broadcast against it rather than assume a single cohort.
        """
        pass

    def set_prms(self, *args, **kwargs):
        """Set parameters of the lifetime distribution, like mean and standard deviation.
        Parameters can be set either by positional arguments or by keyword arguments, but not both.

        Deprecated. Use direct assignment instead, e.g. lifetime_model.mean = 10.

        Args:
            *args: Positional arguments for parameters, in the order of self.prms.
            **kwargs: Keyword arguments for parameters, with parameter names as keys.
        """
        warnings.warn(
            "set_prms is deprecated since v1.1.0. It will be removed in v2.0.0. "
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

    def reset_cached_arrays(self):
        self._sf = None
        self._pdf = None

    def _compute_outflow_pdf(self):
        """Returns an array year-by-cohort of the probability that an item
        added to stock in year m (aka cohort m) leaves in in year n. This value equals pdf(n,m).
        """
        t_diag_indices = np.diag_indices(self._n_t) + (slice(None),) * len(self._shape_no_t)
        self._pdf[t_diag_indices] = 1.0 - np.moveaxis(self.sf.diagonal(0, 0, 1), -1, 0)
        for i_c in range(0, self._n_t):
            self._pdf[i_c + 1 :, i_c, ...] = -1 * np.diff(self.sf[i_c:, i_c, ...], axis=0)

    def _compute_survival_factor_with_nurture(self):
        """
        Compute the survival factor in presence of lifetime extension by year, i.e. the "nurture"
        approach in Krych et al. 2024.
        Math:
        sf_e = plain survival function with extended lifetime
        sf = resulting combined survival function based on previous survival
        current_survival_rate = 1 - hazard_function = sf_e(t)/sf_e(t-1)
        sf(t) = sf(t-1) * current_survival_rate(t)
        sf(t) = sf(t-1) * sf_e(t)/sf_e(t-1)
        """
        factor = self.lt_factor_by_year

        # The arrays in _scaled_prms can be the parameter arrays themselves, so assign new arrays
        # instead of writing into them.
        scaled_prms_orig = dict(self._scaled_prms)
        for i_t in range(self._n_t):
            # scale such that mean and stddev are increased by lifetime extension factor
            for name in self._prm_names_to_scale:
                # apply factor per time step - broadcast to all age cohorts
                self._scaled_prms[name] = scaled_prms_orig[name] * factor[i_t]
            # curr_survival calculates sf_e(t-1) ans sf_e(t) in one array
            # for i_t = 0, the previous time step sf_e(t-1) is omitted
            # all cohorts up to i_t at once: axes are (time step, cohort, other dims)
            # max(i_t, 1) omits prev time step for i_t = 0
            curr_time = self._t.bounds[max(i_t, 1) : i_t + 2]
            cohorts = np.arange(i_t + 1)
            curr_survival = np.zeros((len(curr_time), i_t + 1) + self._shape_no_t)
            for quad_point, quad_weight in zip(self._quad_points, self._quad_weights):
                cohort_time = self._quad_point_time(cohorts, quad_point)
                curr_ages = self._tile(curr_time[:, np.newaxis] - cohort_time[np.newaxis, :])
                curr_survival += quad_weight * self._survival_by_cohort(
                    curr_ages, slice(0, i_t + 1)
                )
            # main diagonal: sf(t) = sf_e(t)
            self._sf[i_t, i_t, ...] = curr_survival[-1, i_t, ...]
            if i_t > 0:
                sf_e_prev = curr_survival[0, :i_t, ...]
                sf_e = curr_survival[1, :i_t, ...]
                sf_prev = self._sf[i_t - 1, :i_t, ...]
                # for small lifetimes, sf_e_prev can be zero, which means that nothing has survived
                ratio = np.divide(sf_e, sf_e_prev, out=np.zeros_like(sf_e), where=sf_e_prev > 0)
                self._sf[i_t, :i_t, ...] = sf_prev * ratio


class FixedLifetime(LifetimeModel):
    """Fixed lifetime, age-cohort leaves the stock in the model year when a certain age,
    specified as 'Mean', is reached."""

    mean: LifetimeArrayType | None = None
    """The fixed lifetime, i.e. the age at which the age-cohort leaves the stock.
    """

    _prm_names: ClassVar[list[str]] = ["mean"]
    _prm_names_to_scale: ClassVar[list[str]] = ["mean"]

    def _survival_by_cohort(self, t, i_c):
        mean = self._scaled_prms["mean"]
        # Example: if lt is 3.5 years fixed, product will still be there after 0, 1, 2, and 3 years,
        # gone after 4 years.
        return (t < mean[i_c, ...]).astype(int)


class StandardDeviationLifetimeModel(LifetimeModel):
    mean: LifetimeArrayType | None = None
    """Mean lifetime.
    """
    std: LifetimeArrayType | None = None
    """Standard deviation of lifetime.
    """
    _prm_names: ClassVar[list[str]] = ["mean", "std"]
    _prm_names_to_scale: ClassVar[list[str]] = ["mean", "std"]


class NormalLifetime(StandardDeviationLifetimeModel):
    """Normally distributed lifetime with mean and standard deviation.
    Watch out for nonzero values, for negative ages, no correction or truncation done here.
    NOTE: As normal distributions have nonzero pdf for negative ages,
    which are physically impossible, these outflow contributions can either be ignored (
    violates the mass balance) or allocated to the zeroth year of residence,
    the latter being implemented in the method compute compute_o_c_from_s_c.
    As alternative, use lognormal or folded normal distribution options.
    """

    def _survival_by_cohort(self, t, i_c):
        mean, std = self._scaled_prms["mean"], self._scaled_prms["std"]
        if np.min(mean) < 0:
            raise ValueError("mean must be greater than zero.")

        return scipy.stats.norm.sf(
            t,
            loc=mean[i_c, ...],
            scale=std[i_c, ...],
        )


class FoldedNormalLifetime(StandardDeviationLifetimeModel):
    """Folded normal distribution, cf. https://en.wikipedia.org/wiki/Folded_normal_distribution
    NOTE: call this with the parameters of the normal distribution mu and sigma of curve
    BEFORE folding, curve after folding will have different mu and sigma.
    """

    def _survival_by_cohort(self, t, i_c):
        mean, std = self._scaled_prms["mean"], self._scaled_prms["std"]
        if np.min(mean) < 0:
            raise ValueError("mean must be greater than zero.")

        return scipy.stats.foldnorm.sf(
            t,
            mean[i_c, ...] / std[i_c, ...],
            0,
            scale=std[i_c, ...],
        )


class LogNormalLifetime(StandardDeviationLifetimeModel):
    """Lognormal distribution
    Here, the mean and stddev of the lognormal curve, not those of the underlying normal
    distribution, need to be specified!
    Values chosen according to description on
    https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.stats.lognorm.html
    Same result as EXCEL function "=LOGNORM.VERT(x;LT_LN;SG_LN;TRUE)"
    """

    def _survival_by_cohort(self, t, i_c):
        mean, std = self._scaled_prms["mean"], self._scaled_prms["std"]
        mean_square = mean[i_c, ...] * mean[i_c, ...]
        std_square = std[i_c, ...] * std[i_c, ...]
        new_mean = np.log(mean_square / np.sqrt(mean_square + std_square))
        new_std = np.sqrt(np.log(1 + std_square / mean_square))
        # compute survival function
        sf_m = scipy.stats.lognorm.sf(t, s=new_std, loc=0, scale=np.exp(new_mean))
        return sf_m


class WeibullLifetime(LifetimeModel):
    """Weibull distribution with standard definition of scale and shape parameters."""

    weibull_scale: LifetimeArrayType | None = None
    """Scale parameter of the Weibull distribution.
    """
    weibull_shape: LifetimeArrayType | None = None
    """Shape parameter of the Weibull distribution.
    """

    _prm_names: ClassVar[list[str]] = ["weibull_scale", "weibull_shape"]
    _prm_names_to_scale: ClassVar[list[str]] = ["weibull_scale"]

    def _survival_by_cohort(self, t, i_c):
        scale, shape = self._scaled_prms["weibull_scale"], self._scaled_prms["weibull_shape"]
        if np.min(shape) < 0:
            raise ValueError("Lifetime weibull_shape must be positive for Weibull distribution.")

        return scipy.stats.weibull_min.sf(
            t,
            c=shape[i_c, ...],
            loc=0,
            scale=scale[i_c, ...],
        )

    # @staticmethod
    # def weibull_c_scale_from_mean_std(mean, std):
    #     """Compute Weibull parameters c and scale from mean and standard deviation.
    #     Works on scalars.
    #     Taken from https://github.com/scipy/scipy/issues/12134#issuecomment-1214031574.
    #     """
    #     def r(c, mean, std):
    #         log_mean, log_std = np.log(mean), np.log(std)
    #         # np.pi*1j is the log of -1
    #         logratio = (logsumexp([gammaln(1 + 2/c) - 2*gammaln(1+1/c), np.pi*1j])
    #                     - 2*log_std + 2*log_mean)
    #         return np.real(logratio)

    #     # Maybe a bit simpler; doesn't seem to be substantially different numerically
    #     # def r(c, mean, std):
    #     #     logratio = (gammaln(1 + 2/c) - 2*gammaln(1+1/c) -
    #     #                 logsumexp([2*log_std - 2*log_mean, 0]))
    #     #     return logratio

    #     # other methods are more efficient, but I've seen TOMS748 return garbage
    #     res = root_scalar(r, args=(mean, std), method='bisect',
    #                     bracket=[1e-300, 1e300], maxiter=2000, xtol=1e-16)
    #     assert res.converged
    #     c = res.root
    #     scale = np.exp(np.log(mean) - gammaln(1 + 1/c))
    #     return c, scale
