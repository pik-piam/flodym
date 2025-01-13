"""Home to various survival models, for use in dynamic stock modelling."""

from abc import abstractmethod
import numpy as np
import scipy.stats
# from scipy.special import gammaln, logsumexp
# from scipy.optimize import root_scalar

from .dimensions import DimensionSet
from .flodym_arrays import FlodymArray


class SurvivalModel:
    """Contains shared functionality across the various survival models."""

    def __init__(self, dims: DimensionSet, time_letter: str = "t"):
        self.dims = dims
        self.t = np.array(dims[time_letter].items)
        if not dims.dim_list[0].letter == time_letter:
            raise ValueError(
                f"The time dimension {time_letter} must be the first dimension in the set."
            )
        self.shape = dims.shape()
        self._n_t = len(self.t)
        self._shape_cohort = (self._n_t,) + self.shape
        self._shape_no_t = tuple(list(self.shape)[1:])
        self.sf = np.zeros(self._shape_cohort)

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
        self._check_lifetime_set()
        for m in range(0, self._n_t):  # cohort index
            self.sf[m::, m, ...] = self._survival_by_year_id(m)

    @abstractmethod
    def _survival_by_year_id(m, **kwargs):
        pass

    @abstractmethod
    def _check_lifetime_set(self):
        pass

    @abstractmethod
    def set_lifetime(self):
        pass

    def compute_outflow_pdf(self):
        """Returns an array year-by-cohort of the probability that an item
        added to stock in year m (aka cohort m) leaves in in year n. This value equals pdf(n,m).
        """
        self.sf = self.compute_survival_factor()
        self.pdf = np.zeros(self._shape_cohort)
        self.pdf[self.t_diag_indices] = 1.0 - np.moveaxis(self.sf.diagonal(0, 0, 1), -1, 0)
        for m in range(0, self._n_t):
            self.pdf[m + 1 :, m, ...] = -1 * np.diff(self.sf[m:, m, ...], axis=0)
        return self.pdf


class FixedSurvival(SurvivalModel):
    """Fixed lifetime, age-cohort leaves the stock in the model year when a certain age,
    specified as 'Mean', is reached."""

    def __init__(
        self,
        dims: DimensionSet,
        time_letter: str = "t",
        lifetime_mean: FlodymArray = None,
    ):
        super().__init__(dims, time_letter)
        if lifetime_mean is None:
            self.lifetime_mean = None
        else:
            self.set_lifetime(lifetime_mean)

    def set_lifetime(self, lifetime_mean: FlodymArray):
        self.lifetime_mean = lifetime_mean.cast_to(target_dims=self.dims).values

    def _check_lifetime_set(self):
        if self.lifetime_mean is None:
            raise ValueError("Lifetime mean must be set before use.")

    def _survival_by_year_id(self, m):
        # Example: if lt is 3.5 years fixed, product will still be there after 0, 1, 2, and 3 years,
        # gone after 4 years.
        return (self._remaining_ages(m) < self.lifetime_mean[m, ...]).astype(int)


class StandardDeviationSurvivalModel(SurvivalModel):
    def __init__(
        self,
        dims: DimensionSet,
        time_letter: str = "t",
        lifetime_mean: FlodymArray = None,
        lifetime_std: FlodymArray = None,
    ):
        super().__init__(dims, time_letter)
        self.lifetime_mean = None
        self.lifetime_std = None
        if lifetime_mean is not None and lifetime_std is not None:
            self.set_lifetime(lifetime_mean, lifetime_std)
        elif lifetime_mean is None != lifetime_std is None:
            raise ValueError("Either both or none of lifetime_mean and lifetime_std must be set.")

    def set_lifetime(self, lifetime_mean: FlodymArray, lifetime_std: FlodymArray):
        self.lifetime_mean = lifetime_mean.cast_to(target_dims=self.dims).values
        self.lifetime_std = lifetime_std.cast_to(target_dims=self.dims).values

    def _check_lifetime_set(self):
        if self.lifetime_mean is None or self.lifetime_std is None:
            raise ValueError("Lifetime mean and standard deviation must be set before use.")


class NormalSurvival(StandardDeviationSurvivalModel):
    """Normally distributed lifetime with mean and standard deviation.
    Watch out for nonzero values, for negative ages, no correction or truncation done here.
    NOTE: As normal distributions have nonzero pdf for negative ages,
    which are physically impossible, these outflow contributions can either be ignored (
    violates the mass balance) or allocated to the zeroth year of residence,
    the latter being implemented in the method compute compute_o_c_from_s_c.
    As alternative, use lognormal or folded normal distribution options.
    """

    def _survival_by_year_id(self, m):
        if np.min(self.lifetime_mean) < 0:
            raise ValueError("lifetime_mean must be greater than zero.")

        return scipy.stats.norm.sf(
            self._remaining_ages(m),
            loc=self.lifetime_mean[m, ...],
            scale=self.lifetime_std[m, ...],
        )


class FoldedNormalSurvival(StandardDeviationSurvivalModel):
    """Folded normal distribution, cf. https://en.wikipedia.org/wiki/Folded_normal_distribution
    NOTE: call this with the parameters of the normal distribution mu and sigma of curve
    BEFORE folding, curve after folding will have different mu and sigma.
    """

    def _survival_by_year_id(self, m):
        if np.min(self.lifetime_mean) < 0:
            raise ValueError("lifetime_mean must be greater than zero.")

        return scipy.stats.foldnorm.sf(
            self._remaining_ages(m),
            self.lifetime_mean[m, ...] / self.lifetime_std[m, ...],
            0,
            scale=self.lifetime_std[m, ...],
        )


class LogNormalSurvival(StandardDeviationSurvivalModel):
    """Lognormal distribution
    Here, the mean and stddev of the lognormal curve, not those of the underlying normal
    distribution, need to be specified!
    Values chosen according to description on
    https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.stats.lognorm.html
    Same result as EXCEL function "=LOGNORM.VERT(x;LT_LN;SG_LN;TRUE)"
    """


    def _survival_by_year_id(self, m):
        # calculate parameter mu of underlying normal distribution:
        lt_ln = np.log(
            self.lifetime_mean[m, ...]
            / np.sqrt(
                1
                + (
                    self.lifetime_mean[m, ...]
                    * self.lifetime_mean[m, ...]
                    / (self.lifetime_std[m, ...] * self.lifetime_std[m, ...])
                )
            )
        )
        # calculate parameter sigma of underlying normal distribution
        sg_ln = np.sqrt(
            np.log(
                1
                + (
                    self.lifetime_mean[m, ...]
                    * self.lifetime_mean[m, ...]
                    / (self.lifetime_std[m, ...] * self.lifetime_std[m, ...])
                )
            )
        )
        # compute survial function
        return scipy.stats.lognorm.sf(self._remaining_ages(m), s=sg_ln, loc=0, scale=np.exp(lt_ln))


class WeibullSurvival(SurvivalModel):
    """Weibull distribution with standard definition of scale and shape parameters."""

    def __init__(
        self,
        dims: DimensionSet,
        time_letter: str = "t",
        lifetime_shape: FlodymArray = None,
        lifetime_scale: FlodymArray = None,
    ):
        super().__init__(dims, time_letter)
        self.lifetime_shape = None
        self.lifetime_scale = None
        if lifetime_shape is not None and lifetime_scale is not None:
            self.set_lifetime(lifetime_shape, lifetime_scale)
        elif lifetime_shape is None != lifetime_scale is None:
            raise ValueError("Either both or none of lifetime_shape and lifetime_scale must be set.")

    def set_lifetime(self, lifetime_shape: FlodymArray, lifetime_scale: FlodymArray):
        self.lifetime_shape = lifetime_shape.cast_to(target_dims=self.dims).values
        self.lifetime_scale = lifetime_scale.cast_to(target_dims=self.dims).values

    def _check_lifetime_set(self):
        if self.lifetime_shape is None or self.lifetime_scale is None:
            raise ValueError("Lifetime mean and standard deviation must be set before use.")

    def _survival_by_year_id(self, m):
        if np.min(self.lifetime_shape) < 0:
            raise ValueError("Lifetime shape must be positive for Weibull distribution.")

        return scipy.stats.weibull_min.sf(
            self._remaining_ages(m),
            c=self.lifetime_shape[m, ...],
            loc=0,
            scale=self.lifetime_scale[m, ...],
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


def get_survival_model_by_type(stock_type: str) -> SurvivalModel:
    """Return the stock class for a given stock type."""
    survival_model_by_type = {
        "fixed": FixedSurvival,
        "normal": NormalSurvival,
        "foldedNormal": FoldedNormalSurvival,
        "logNormal": LogNormalSurvival,
        "weibull": WeibullSurvival,
        None: None,
    }
    if stock_type not in survival_model_by_type:
        raise ValueError(f"Stock type {stock_type} must be one of {list(survival_model_by_type.keys())}.")
    return survival_model_by_type[stock_type]
