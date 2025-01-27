"""Home to various lifetime models, for use in dynamic stock modelling."""

from abc import abstractmethod
import numpy as np
import scipy.stats
from pydantic import BaseModel as PydanticBaseModel, model_validator
from typing import Any

# from scipy.special import gammaln, logsumexp
# from scipy.optimize import root_scalar

from .dimensions import DimensionSet
from .flodym_arrays import FlodymArray


class LifetimeModel(PydanticBaseModel):
    """Contains shared functionality across the various lifetime models."""

    dims: DimensionSet
    time_letter: str = "t"
    _sf: np.ndarray = None

    @property
    def t(self):
        return np.array(self.dims[self.time_letter].items)

    @property
    def shape(self):
        return self.dims.shape()

    @property
    def _n_t(self):
        return len(self.t)

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
    def t_diag_indices(self):
        return np.diag_indices(self._n_t) + (slice(None),) * len(self._shape_no_t)

    def _tile(self, a: np.ndarray) -> np.ndarray:
        index = (slice(None),) * a.ndim + (np.newaxis,) * len(self._shape_no_t)
        out = a[index]
        return np.tile(out, self._shape_no_t)

    def _remaining_ages(self, m):
        return self._tile(self.t[m:] - self.t[m])

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
        for m in range(0, self._n_t):  # cohort index
            self._sf[m::, m, ...] = self._survival_by_year_id(m)

    @abstractmethod
    def _survival_by_year_id(m, **kwargs):
        pass

    @abstractmethod
    def _check_prms_set(self):
        pass

    @abstractmethod
    def set_prms(self):
        pass

    def cast_any_to_flodym_array(self, prm_in):
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
        self._sf = self.compute_survival_factor()
        pdf = np.zeros(self._shape_cohort)
        pdf[self.t_diag_indices] = 1.0 - np.moveaxis(self._sf.diagonal(0, 0, 1), -1, 0)
        for m in range(0, self._n_t):
            pdf[m + 1 :, m, ...] = -1 * np.diff(self._sf[m:, m, ...], axis=0)
        return pdf


class FixedLifetime(LifetimeModel):
    """Fixed lifetime, age-cohort leaves the stock in the model year when a certain age,
    specified as 'Mean', is reached."""

    mean: Any = None

    @model_validator(mode="after")
    def cast_mean(self):
        if self.mean is not None:
            self.mean = self.cast_any_to_flodym_array(self.mean)
        return self

    def set_prms(self, mean: FlodymArray):
        self.mean = self.cast_any_to_flodym_array(mean)

    def _check_prms_set(self):
        if self.mean is None:
            raise ValueError("Lifetime mean must be set before use.")

    def _survival_by_year_id(self, m):
        # Example: if lt is 3.5 years fixed, product will still be there after 0, 1, 2, and 3 years,
        # gone after 4 years.
        return (self._remaining_ages(m) < self.mean[m, ...]).astype(int)


class StandardDeviationLifetimeModel(LifetimeModel):
    mean: Any = None
    std: Any = None

    @model_validator(mode="after")
    def cast_mean_std(self):
        if self.mean is not None:
            self.mean = self.cast_any_to_flodym_array(self.mean)
        if self.std is not None:
            self.std = self.cast_any_to_flodym_array(self.std)
        return self

    def set_prms(self, mean: FlodymArray, std: FlodymArray):
        self.mean = self.cast_any_to_flodym_array(mean)
        self.std = self.cast_any_to_flodym_array(std)

    def _check_prms_set(self):
        if self.mean is None or self.std is None:
            raise ValueError("Lifetime mean and standard deviation must be set before use.")


class NormalLifetime(StandardDeviationLifetimeModel):
    """Normally distributed lifetime with mean and standard deviation.
    Watch out for nonzero values, for negative ages, no correction or truncation done here.
    NOTE: As normal distributions have nonzero pdf for negative ages,
    which are physically impossible, these outflow contributions can either be ignored (
    violates the mass balance) or allocated to the zeroth year of residence,
    the latter being implemented in the method compute compute_o_c_from_s_c.
    As alternative, use lognormal or folded normal distribution options.
    """

    def _survival_by_year_id(self, m):
        if np.min(self.mean) < 0:
            raise ValueError("mean must be greater than zero.")

        return scipy.stats.norm.sf(
            self._remaining_ages(m),
            loc=self.mean[m, ...],
            scale=self.std[m, ...],
        )


class FoldedNormalLifetime(StandardDeviationLifetimeModel):
    """Folded normal distribution, cf. https://en.wikipedia.org/wiki/Folded_normal_distribution
    NOTE: call this with the parameters of the normal distribution mu and sigma of curve
    BEFORE folding, curve after folding will have different mu and sigma.
    """

    def _survival_by_year_id(self, m):
        if np.min(self.mean) < 0:
            raise ValueError("mean must be greater than zero.")

        return scipy.stats.foldnorm.sf(
            self._remaining_ages(m),
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

    def _survival_by_year_id(self, m):
        mean_square = self.mean[m, ...] * self.mean[m, ...]
        std_square = self.std[m, ...] * self.std[m, ...]
        new_mean = np.log(mean_square / np.sqrt(mean_square + std_square))
        new_std = np.sqrt(np.log(1 + std_square / mean_square))
        lt_ln = new_mean
        sg_ln = new_std
        # compute survival function
        sf_m = scipy.stats.lognorm.sf(self._remaining_ages(m), s=sg_ln, loc=0, scale=np.exp(lt_ln))
        return sf_m


class WeibullLifetime(LifetimeModel):
    """Weibull distribution with standard definition of scale and shape parameters."""

    weibull_shape: Any = None
    weibull_scale: Any = None

    @model_validator(mode="after")
    def cast_shape_scale(self):
        if self.weibull_shape is not None:
            self.weibull_shape = self.cast_any_to_flodym_array(self.weibull_shape)
        if self.weibull_scale is not None:
            self.weibull_scale = self.cast_any_to_flodym_array(self.weibull_scale)
        return self

    def set_prms(self, weibull_shape: FlodymArray, weibull_scale: FlodymArray):
        self.weibull_shape = self.cast_any_to_flodym_array(weibull_shape)
        self.weibull_scale = self.cast_any_to_flodym_array(weibull_scale)

    def _check_prms_set(self):
        if self.weibull_shape is None or self.weibull_scale is None:
            raise ValueError("Lifetime mean and standard deviation must be set before use.")

    def _survival_by_year_id(self, m):
        if np.min(self.weibull_shape) < 0:
            raise ValueError("Lifetime shape must be positive for Weibull distribution.")

        return scipy.stats.weibull_min.sf(
            self._remaining_ages(m),
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
