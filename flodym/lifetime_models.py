"""Home to various lifetime models, for use in dynamic stock modelling."""

from abc import abstractmethod
import numpy as np
import scipy.stats
from pydantic import BaseModel as PydanticBaseModel, model_validator
from typing import Any

# from scipy.special import gammaln, logsumexp
# from scipy.optimize import root_scalar

from .dimensions import DimensionSet, Dimension
from .flodym_arrays import FlodymArray
from .gauss_lobatto import gl_nodes, gl_weights


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

    dims: DimensionSet
    time_letter: str = "t"
    inflow_at: str = "middle"
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
    _sf: np.ndarray = None
    _pdf: np.ndarray = None
    _t: UnevenTimeDim = None

    @model_validator(mode="after")
    def check_inflow_at(self):
        if self.inflow_at not in ["start", "middle", "end"]:
            raise ValueError("inflow_at must be one of 'start', 'middle', or 'end'.")
        return self

    @model_validator(mode="after")
    def cast_prms(self):
        for prm_name, prm in self.prms.items():
            if prm is not None:
                setattr(self, prm_name, self.cast_any_to_np_array(prm))
        return self

    @model_validator(mode="after")
    def init_t(self):
        self._t = UnevenTimeDim(dim=self.dims[self.time_letter])
        return self

    @property
    @abstractmethod
    def prms(self) -> dict[str, np.ndarray | None]:
        raise NotImplementedError

    def _check_prms_set(self):
        for prm_name, prm in self.prms.items():
            if prm is None:
                raise ValueError(f"Lifetime {prm_name} must be set before use.")

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
    def sf(self):
        if self._sf is None:
            self._sf = np.zeros(self._shape_cohort)
            self.compute_survival_factor()
        return self._sf

    @property
    def pdf(self):
        if self._pdf is None:
            self._pdf = np.zeros(self._shape_cohort)
            self.compute_outflow_pdf()
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

    def _remaining_ages(self, m, eta):
        t = eta * self._t.bounds[m + 1] + (1 - eta) * self._t.bounds[m]
        return self._tile(self._t.bounds[m + 1 :] - t)

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
        self._check_prms_set()
        quad_eta, quad_weights = self.get_quad_points_and_weights()
        for m in range(0, self._n_t):  # cohort index
            for eta, weight in zip(list(quad_eta), list(quad_weights)):
                t = self._remaining_ages(m, eta)
                self._sf[m::, m, ...] += weight * self._survival_by_year_id(t, m)

    def get_quad_points_and_weights(self):
        """Returns the quadrature points and weights for the inflow time periods."""
        if self.n_pts_per_interval > 10:
            raise ValueError("quad_order must be between 0 and 9.")
        if self.n_pts_per_interval > 1:
            nodes = [(x + 1) / 2 for x in gl_nodes[self.n_pts_per_interval]]
            weights = [w / 2 for w in gl_weights[self.n_pts_per_interval]]
            return nodes, weights
        else:
            if self.inflow_at == "start":
                return [0], [1]
            elif self.inflow_at == "middle":
                return [0.5], [1]
            elif self.inflow_at == "end":
                return [1], [1]

    @abstractmethod
    def _survival_by_year_id(m, **kwargs):
        pass

    @abstractmethod
    def set_prms(self):
        pass

    def cast_any_to_np_array(self, prm_in):
        if isinstance(prm_in, FlodymArray):
            prm_out = prm_in.cast_to(target_dims=self.dims).values
        else:
            prm_out = np.ndarray(self.shape)
            prm_out[...] = prm_in
        return prm_out

    def compute_outflow_pdf(self):
        """Returns an array year-by-cohort of the probability that an item
        added to stock in year m (aka cohort m) leaves in in year n. This value equals pdf(n,m).
        """
        t_diag_indices = np.diag_indices(self._n_t) + (slice(None),) * len(self._shape_no_t)
        self._pdf[t_diag_indices] = 1.0 - np.moveaxis(self.sf.diagonal(0, 0, 1), -1, 0)
        for m in range(0, self._n_t):
            self._pdf[m + 1 :, m, ...] = -1 * np.diff(self.sf[m:, m, ...], axis=0)


class FixedLifetime(LifetimeModel):
    """Fixed lifetime, age-cohort leaves the stock in the model year when a certain age,
    specified as 'Mean', is reached."""

    mean: Any = None

    @property
    def prms(self):
        return {"mean": self.mean}

    def set_prms(self, mean: FlodymArray):
        self.mean = self.cast_any_to_np_array(mean)

    def _survival_by_year_id(self, t, m):
        # Example: if lt is 3.5 years fixed, product will still be there after 0, 1, 2, and 3 years,
        # gone after 4 years.
        return (t < self.mean[m, ...]).astype(int)


class StandardDeviationLifetimeModel(LifetimeModel):
    mean: Any = None
    std: Any = None

    @property
    def prms(self):
        return {"mean": self.mean, "std": self.std}

    def set_prms(self, mean: FlodymArray, std: FlodymArray):
        self.mean = self.cast_any_to_np_array(mean)
        self.std = self.cast_any_to_np_array(std)


class NormalLifetime(StandardDeviationLifetimeModel):
    """Normally distributed lifetime with mean and standard deviation.
    Watch out for nonzero values, for negative ages, no correction or truncation done here.
    NOTE: As normal distributions have nonzero pdf for negative ages,
    which are physically impossible, these outflow contributions can either be ignored (
    violates the mass balance) or allocated to the zeroth year of residence,
    the latter being implemented in the method compute compute_o_c_from_s_c.
    As alternative, use lognormal or folded normal distribution options.
    """

    def _survival_by_year_id(self, t, m):
        if np.min(self.mean) < 0:
            raise ValueError("mean must be greater than zero.")

        return scipy.stats.norm.sf(
            t,
            loc=self.mean[m, ...],
            scale=self.std[m, ...],
        )


class FoldedNormalLifetime(StandardDeviationLifetimeModel):
    """Folded normal distribution, cf. https://en.wikipedia.org/wiki/Folded_normal_distribution
    NOTE: call this with the parameters of the normal distribution mu and sigma of curve
    BEFORE folding, curve after folding will have different mu and sigma.
    """

    def _survival_by_year_id(self, t, m):
        if np.min(self.mean) < 0:
            raise ValueError("mean must be greater than zero.")

        return scipy.stats.foldnorm.sf(
            t,
            self.mean[m, ...] / self.std[m, ...],
            0,
            scale=self.std[m, ...],
        )


class LogNormalLifetime(StandardDeviationLifetimeModel):
    """Lognormal distribution
    Here, the mean and stddev of the lognormal curve, not those of the underlying normal
    distribution, need to be specified!
    Values chosen according to description on
    https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.stats.lognorm.html
    Same result as EXCEL function "=LOGNORM.VERT(x;LT_LN;SG_LN;TRUE)"
    """

    def _survival_by_year_id(self, t, m):
        mean_square = self.mean[m, ...] * self.mean[m, ...]
        std_square = self.std[m, ...] * self.std[m, ...]
        new_mean = np.log(mean_square / np.sqrt(mean_square + std_square))
        new_std = np.sqrt(np.log(1 + std_square / mean_square))
        # compute survival function
        sf_m = scipy.stats.lognorm.sf(t, s=new_std, loc=0, scale=np.exp(new_mean))
        return sf_m


class WeibullLifetime(LifetimeModel):
    """Weibull distribution with standard definition of scale and shape parameters."""

    weibull_shape: Any = None
    weibull_scale: Any = None

    @property
    def prms(self):
        return {"weibull_shape": self.weibull_shape, "weibull_scale": self.weibull_scale}

    def set_prms(self, weibull_shape: FlodymArray, weibull_scale: FlodymArray):
        self.weibull_shape = self.cast_any_to_np_array(weibull_shape)
        self.weibull_scale = self.cast_any_to_np_array(weibull_scale)

    def _survival_by_year_id(self, t, m):
        if np.min(self.weibull_shape) < 0:
            raise ValueError("Lifetime weibull_shape must be positive for Weibull distribution.")

        return scipy.stats.weibull_min.sf(
            t,
            c=self.weibull_shape[m, ...],
            loc=0,
            scale=self.weibull_scale[m, ...],
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
