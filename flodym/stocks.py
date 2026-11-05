"""Home to various `Stock` classes,
including flow-driven stocks and dynamic (lifetime-based) stocks.
"""

from abc import abstractmethod
import functools
import numpy as np
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, model_validator
from typing import TYPE_CHECKING, Callable, ClassVar, Optional, Union, TypeVar, Type
import logging
import warnings

from .processes import Process, UnderdeterminedError
from .flodym_arrays import StockArray, FlodymArray
from .dimensions import Dimension, DimensionSet
from .lifetime_models import LifetimeModel, UnevenTimeDim
from ._checks import balance_failure, config_hint, report_balance_failures
from .config import ErrorBehavior, config, handle_error

if TYPE_CHECKING:
    from ._process_compute import ProcessComputation

StockSubtype = TypeVar("StockSubtype", bound="Stock")


def _wrap_compute(compute: Callable) -> Callable:
    """Adds checks before a stock compute routine, and marks the stock arrays as set and checks
    the mass balance of the stock (if ``flodym.config.checks.mass_balance_after_compute``) after it.
    Nested calls (e.g. via ``super().compute()``) only run the outermost checks."""

    @functools.wraps(compute)
    def wrapper(self: "Stock", *args, **kwargs):
        if self._in_compute:
            return compute(self, *args, **kwargs)
        self._check_needed_arrays()
        self._in_compute = True
        try:
            result = compute(self, *args, **kwargs)
        finally:
            self._in_compute = False
        self.mark_all_set()
        if config.checks.mass_balance_after_compute:
            self._check_mass_balance(
                tolerance=None,
                error_behavior=config.error_behaviors.mass_balance,
                hint=config_hint("mass_balance", tolerance=True, check="mass_balance_after_compute"),
            )
        return result

    wrapper._is_wrapped_compute = True
    return wrapper


class Stock(PydanticBaseModel):
    """Stock objects are components of an MFASystem, where materials can accumulate over time.
    They consist of three :py:class:`flodym.FlodymArray` objects:
    stock (the accumulation), inflow, outflow.

    The base class only allows to compute the stock from known inflow and outflow.
    The subclasses allows computations using a lifetime distribution function,
    which is necessary if not both inflow and outflow are known.
    """

    model_config = ConfigDict(protected_namespaces=(), arbitrary_types_allowed=True)

    dims: DimensionSet
    """Dimensions of the stock, inflow, and outflow arrays. Time must be the first dimension."""
    stock: Optional[StockArray] = None
    """Accumulation of the stock"""
    inflow: Optional[StockArray] = None
    """Inflow into the stock"""
    outflow: Optional[StockArray] = None
    """Outflow from the stock"""
    name: Optional[str] = "unnamed"
    """Name of the stock"""
    process: Optional[Process] = None
    """Process the stock is associated with, if any. Needed for example for the mass balance."""
    time_letter: str = "t"
    """Letter of the time dimension in the dimensions set, to make sure it's the first one."""
    _t: UnevenTimeDim = None
    _in_compute: bool = False
    _n_equations: ClassVar[int] = 1
    """Number of equations relating inflow, outflow and stock. Used to detect overdetermined
    processes in :py:attr:`flodym.Process.is_overdetermined`."""

    def __init_subclass__(cls, **kwargs):
        """Wrap the compute method of every subclass with checks before and
        marking the stock arrays as set afterwards."""
        super().__init_subclass__(**kwargs)
        compute = cls.__dict__.get("compute")
        if compute is not None and not getattr(compute, "_is_wrapped_compute", False):
            cls.compute = _wrap_compute(compute)

    @model_validator(mode="after")
    def validate_stock_arrays(self):
        if self.stock is None:
            self.stock = StockArray(dims=self.dims, name=f"{self.name}_stock")
        elif self.stock.dims.letters != self.dims.letters:
            raise ValueError(
                f"Stock dimensions {self.stock.dims.letters} do not match prescribed dims {self.dims.letters}."
            )
        if self.inflow is None:
            self.inflow = StockArray(dims=self.dims, name=f"{self.name}_inflow")
        elif self.inflow.dims.letters != self.dims.letters:
            raise ValueError(
                f"Inflow dimensions {self.inflow.dims.letters} do not match prescribed dims {self.dims.letters}."
            )
        if self.outflow is None:
            self.outflow = StockArray(dims=self.dims, name=f"{self.name}_outflow")
        elif self.outflow.dims.letters != self.dims.letters:
            raise ValueError(
                f"Outflow dimensions {self.outflow.dims.letters} do not match prescribed dims {self.dims.letters}."
            )
        return self

    @model_validator(mode="after")
    def validate_time_first_dim(self):
        if self.dims.letters[0] != self.time_letter:
            raise ValueError(
                f"Time dimension must be the first dimension, i.e. time_letter (now {self.time_letter}) must be the first letter in dims.letters (now {self.dims.letters[0]})."
            )
        return self

    @model_validator(mode="after")
    def init_t(self):
        self._t = UnevenTimeDim(dim=self.dims[self.time_letter])
        return self

    @abstractmethod
    def compute(self):
        """Compute the missing stock arrays from the known ones.

        In all subclasses, ``_check_needed_arrays()`` is automatically called before, and the
        stock arrays are marked as set after the computation.
        """
        pass

    @abstractmethod
    def _check_needed_arrays(self):
        pass

    def _compute_process(self, computation: "ProcessComputation"):
        """Compute the stock together with the unknown in- and outflows of its process, as part
        of :py:meth:`flodym.Process.compute`. The inflows of the process sum to the stock
        inflow, the outflows to the stock outflow.

        Implemented by the subclasses, using the methods of ``computation``. They raise a
        :py:class:`flodym.UnderdeterminedError` if the process cannot be computed. If they raise
        any error, :py:meth:`flodym.Process.compute` restores all arrays of the process.
        """
        raise UnderdeterminedError(
            self.process,
            f"Stock type {self.__class__.__name__} of stock '{self.name}' does not support the "
            "automatic computation of its process.",
        )

    def _set_from_total(self, array: StockArray, total: FlodymArray, label: str) -> None:
        """Set the stock inflow or outflow to the total of the in- or outflows of the process."""
        missing_dims = array.dims - total.dims
        if missing_dims:
            raise ValueError(
                f"In process '{self.process.name}': The stock {label} has the dimensions "
                f"{missing_dims.names}, which the total of the {label}s of the process does not "
                "have. Use fewer dimensions for the stock, or a neighboring process with a "
                "dimension splitter."
            )
        array[...] = total

    def _check_array_known(self, array: StockArray, label: str, consequence: str) -> None:
        """Warn if a stock array needed for the computation is zero, and handle the case that it
        is not set according to ``config.error_behaviors.unset_array_used``."""
        if np.allclose(array.values, 0.0):
            logging.warning(f"{label} is zero. {consequence}")
        if not array.is_set:
            handle_error(
                config.error_behaviors.unset_array_used,
                message=(
                    f"{label} of stock '{self.name}' is not set (is_set=False). "
                    "If its values are intended, call mark_set() on it."
                ),
            )

    def mark_all_set(self):
        """Mark all StockArrays (inflow, outflow, stock) as set."""
        self.inflow.mark_set()
        self.outflow.mark_set()
        self.stock.mark_set()

    def mark_all_unset(self):
        """Mark all StockArrays (inflow, outflow, stock) as unset."""
        self.inflow.mark_unset()
        self.outflow.mark_unset()
        self.stock.mark_unset()

    @property
    def shape(self) -> tuple:
        """Shape of the stock, inflow, outflow arrays, defined by the dimensions."""
        return self.dims.shape

    @property
    def process_id(self) -> int:
        """ID of the process the stock is associated with."""
        return self.process.id

    def to_stock_type(self, desired_stock_type: Type[StockSubtype], **kwargs) -> StockSubtype:
        """Return an object of a new stock type with values and dimensions the same as the original.
        `**kwargs` can be used to pass additional model attributes as required by the desired stock
        type, if these are not contained in the original stock type.
        """
        return desired_stock_type(**self.__dict__, **kwargs)

    def copy(self) -> "Stock":
        """Return a copy of the Stock.

        The copy has its own :py:class:`flodym.DimensionSet` and its own stock, inflow and
        outflow arrays (created via :py:meth:`flodym.FlodymArray.copy`), so that modifications
        to either the copy or the original do not affect the other.
        """
        return self.model_copy(
            update={
                "dims": self.dims.copy(),
                "stock": self.stock.copy(),
                "inflow": self.inflow.copy(),
                "outflow": self.outflow.copy(),
            }
        )

    @property
    def net_inflow(self) -> FlodymArray:
        """Inflow minus outflow of the stock, per year."""
        return self.inflow - self.outflow

    @property
    def stock_change(self) -> FlodymArray:
        """Change of the stock in each time step: stock(t) - stock(t-1), where the stock before the
        first time step is zero."""
        return FlodymArray(
            dims=self.dims,
            values=np.diff(self.stock.values, axis=0, prepend=0),
            name=f"{self.name}_stock_change",
        )

    def get_mass_balance(self) -> FlodymArray:
        """Mass balance error of the stock: The net inflow over each time step (see
        :py:attr:`net_inflow`, multiplied by the length of the time step) minus the stock
        change in the time step (see :py:attr:`stock_change`).
        Zero if inflow, outflow and stock are consistent.
        """
        net_inflow_whole_period = self._to_whole_period(self.net_inflow.values)
        return FlodymArray(
            dims=self.dims,
            values=net_inflow_whole_period - self.stock_change.values,
            name=f"{self.name}_mass_balance",
        )

    @property
    def _absolute_float_precision(self) -> float:
        """Machine epsilon times the largest absolute value of inflow, outflow and stock."""
        return max(
            self.inflow._absolute_float_precision,
            self.outflow._absolute_float_precision,
            self.stock._absolute_float_precision,
        )

    @property
    def tolerance(self) -> float:
        """Absolute tolerance for the mass balance check of the stock, see
        :py:attr:`flodym.config.Config.relative_tolerance`."""
        return config.get_tolerance(self._absolute_float_precision)

    def check_mass_balance(
        self,
        tolerance: Optional[float] = None,
        error_behavior: Optional[ErrorBehavior | str] = None,
    ) -> bool:
        """Check whether inflow, outflow and stock are consistent within a tolerance
        (see :py:meth:`get_mass_balance`).

        This check is also performed automatically after each computation of the stock if
        ``flodym.config.checks.mass_balance_after_compute`` is True (default: False).

        Args:
            tolerance: Absolute tolerance. If None, :py:attr:`tolerance` is used.
            error_behavior: What to do if the check fails. If None,
                ``flodym.config.error_behaviors.mass_balance`` is used (default: raise an error).

        Returns:
            Whether the check passed.
        """
        if error_behavior is None:
            error_behavior = config.error_behaviors.mass_balance
        return self._check_mass_balance(
            tolerance, error_behavior, hint=config_hint("mass_balance", tolerance=True)
        )

    def _check_mass_balance(
        self, tolerance: Optional[float], error_behavior: ErrorBehavior | str, hint: str
    ) -> bool:
        return report_balance_failures(
            [self._mass_balance_failure(tolerance)],
            error_behavior,
            hint=hint,
            success_message=f"Mass balance of stock '{self.name}' is consistent.",
        )

    def _mass_balance_failure(self, tolerance: Optional[float] = None) -> Optional[str]:
        """Description of the failed mass balance check for an error message, or None if it
        passed. If no tolerance is given, :py:attr:`tolerance` is used."""
        if tolerance is None:
            tolerance = self.tolerance
        return balance_failure(f"stock '{self.name}'", self.get_mass_balance(), tolerance)

    def check_stock_balance(self) -> bool:
        """Deprecated. Use :py:meth:`check_mass_balance` instead."""
        warnings.warn(
            "check_stock_balance is deprecated since v1.2.0. It will be removed in v2.0.0. "
            "Use check_mass_balance instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.check_mass_balance()

    def get_stock_balance(self) -> np.ndarray:
        """Deprecated. Use :py:meth:`get_mass_balance` instead, which returns a
        :py:class:`flodym.FlodymArray`. This method returns its values."""
        warnings.warn(
            "get_stock_balance is deprecated since v1.2.0. It will be removed in v2.0.0. "
            "Use get_mass_balance instead, which returns a FlodymArray.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_mass_balance().values

    @property
    def is_computed(self) -> bool:
        """Check whether the stock has been computed, i.e. whether the stock, inflow, and outflow arrays
        are all set.
        """
        return self.inflow.is_set and self.outflow.is_set and self.stock.is_set

    def _to_whole_period(self, annual_flow: np.ndarray) -> np.ndarray:
        """multiply annual flow by interval length to get flow over whole period."""
        return np.einsum("t...,t->t...", annual_flow, self._t.interval_lengths)

    def _to_annual(self, whole_period_flow: np.ndarray) -> np.ndarray:
        """divide flow over whole period by interval length to get annual flow"""
        return np.einsum("t...,t->t...", whole_period_flow, 1.0 / self._t.interval_lengths)

    def __str__(self):
        base = f"{self.__class__.__name__} '{self.name}'"
        dims = f" with dims ({','.join(self.dims.letters)}) and shape {self.shape};"
        return base + dims


class SimpleFlowDrivenStock(Stock):
    """Given inflows and outflows, the stock can be calculated without a lifetime model or cohorts."""

    def _check_needed_arrays(self):
        if (
            np.max(np.abs(self.inflow.values)) < 1e-10
            and np.max(np.abs(self.outflow.values)) < 1e-10
        ):
            logging.warning("Inflow and Outflow are zero. This will lead to a zero stock.")
        if not self.inflow.is_set and not self.outflow.is_set:
            handle_error(
                config.error_behaviors.unset_array_used,
                message=(
                    f"Neither inflow nor outflow of stock '{self.name}' is set (is_set=False). "
                    "If their values are intended, call mark_set() on one of them."
                ),
            )

    def compute(self):
        annual_net_inflow = self.inflow.values - self.outflow.values
        net_inflow_whole_period = self._to_whole_period(annual_net_inflow)
        self.stock.values[...] = np.cumsum(net_inflow_whole_period, axis=0)

    def _compute_process(self, computation: "ProcessComputation"):
        """Compute the stock and the unknown in- and outflows of its process.

        If the stock is already computed, its inflow and outflow are the totals of the in- and
        outflows of the process. Otherwise, each side of the process (inflows and outflows) is
        treated separately: Its total is computed from its flows as described in
        :py:meth:`flodym.Process.compute`, and assigned to the stock inflow (or outflow). If this
        is not possible, but the stock inflow (or outflow) is set, it is used as total instead.
        A side without flows has a total of zero. The stock is then computed from its inflow and
        outflow. Finally, the unknown flows of each side are computed from its total.
        """
        computation.require_flows()
        stock_arrays = {"in": self.inflow, "out": self.outflow}
        totals = dict(stock_arrays)
        if not self.is_computed:
            for side, array in stock_arrays.items():
                process_side = computation.sides[side]
                if not process_side.flows:
                    array[...] = 0.0
                elif process_side.total_method() is not None or not array.is_set:
                    # keep the total, which may have more dimensions than the stock array
                    totals[side] = computation.compute_total((side,))
                    self._set_from_total(array, totals[side], f"{side}flow")
            self.compute()
        for side, total in totals.items():
            computation.compute_side(total, side)

class DynamicStockModel(Stock):
    """Dynamic stock model, based on stocks having a specified lifetime (distribution).

    It can be computed in three modes, each given by a public method:

    - :py:meth:`compute_inflow_driven`: from the inflow,
    - :py:meth:`compute_stock_driven`: from the stock,
    - :py:meth:`compute_inflow_by_cohort_driven`: from ``inflow_by_cohort``.

    These methods can be called repeatedly and in any order on the same object, for example to
    compute a stock-driven model, correct the resulting inflow, and recompute inflow-driven.

    :py:meth:`compute` chooses the mode from which of stock, inflow and inflow_by_cohort is set.
    The subclasses :py:class:`flodym.InflowDrivenDSM` and :py:class:`flodym.StockDrivenDSM`
    instead fix the mode of :py:meth:`compute`.
    """

    lifetime_model: Union[LifetimeModel, type]
    """Lifetime model, which contains the lifetime distribution function.
    Can be input either as a LifetimeModel subclass, or as an instance of a
    LifetimeModel subclass. For available subclasses, see `flodym.lifetime_models`.
    """
    cohort_dim: Optional[Dimension] = None
    inflow_by_cohort: Optional[StockArray] = None
    """Annual inflow by year and age-cohort, which can also enter older cohorts.
    Requires ``cohort_dim``. Only used by :py:meth:`compute_inflow_by_cohort_driven`.
    """

    _outflow_by_cohort: np.ndarray = None
    _stock_by_cohort: np.ndarray = None
    _dims_cohort: DimensionSet = None
    _initial_stock_by_cohort: Optional[np.ndarray] = None
    _initial_stock_year: int = None
    _initial_stock_included_in_inflow: bool = False
    _n_equations: ClassVar[int] = 2

    def validate_cohort_dim(self):
        t = self.dims[self.time_letter]
        c = self.cohort_dim
        if c.letter == t.letter or c.name == t.name:
            raise ValueError(
                "Cohort dimension letter and name must be different from time dimension letter and name."
            )
        if c.items != t.items:
            raise ValueError("Cohort dimension size must be the same as time dimension size.")
        if c.letter in self.dims.letters or c.name in self.dims.names:
            raise ValueError("Cohort dimension must not be part of the stock dimensions.")

    @model_validator(mode="after")
    def init_cohort_arrays(self):
        self._stock_by_cohort = np.zeros(self._shape_cohort)
        self._outflow_by_cohort = np.zeros(self._shape_cohort)
        return self

    @model_validator(mode="after")
    def init_inflow_by_cohort(self):
        if self.inflow_by_cohort is None:
            if self.cohort_dim is not None:
                self.inflow_by_cohort = StockArray(
                    dims=self.dims_cohort, name=f"{self.name}_inflow_by_cohort"
                )
        elif self.inflow_by_cohort.dims.letters != self.dims_cohort.letters:
            raise ValueError(
                f"Inflow by cohort dimensions {self.inflow_by_cohort.dims.letters} do not match expected dims {self.dims_cohort.letters}."
            )
        return self

    @model_validator(mode="after")
    def init_lifetime_model(self):
        if isinstance(self.lifetime_model, type):
            if not issubclass(self.lifetime_model, LifetimeModel):
                raise ValueError("lifetime_model must be a subclass of LifetimeModel.")
            self.lifetime_model = self.lifetime_model(dims=self.dims, time_letter=self.time_letter)
        elif self.lifetime_model.dims.letters != self.dims.letters:
            raise ValueError("Lifetime model dimensions do not match stock dimensions.")
        return self

    def _check_needed_arrays(self):
        self.lifetime_model._check_prms_set()

    def _check_inflow_by_cohort_known(self):
        if self.inflow_by_cohort is None:
            raise ValueError(
                "Inflow by cohort driven computation requires cohort_dim to be given at DSM initialization."
            )
        self._check_array_known(
            self.inflow_by_cohort, "Inflow by cohort", "This will lead to a zero stock and outflow."
        )

    def mark_all_unset(self):
        """Mark all StockArrays (inflow, outflow, stock, and inflow_by_cohort, if present) as
        unset."""
        super().mark_all_unset()
        if self.inflow_by_cohort is not None:
            self.inflow_by_cohort.mark_unset()

    @property
    def dims_cohort(self):
        if self._dims_cohort is None:
            if self.cohort_dim is None:
                raise ValueError("Cohort dimension must be provided at DSM initialization")
            self.validate_cohort_dim()
            t = self.dims[self.time_letter]
            c = self.cohort_dim
            self._dims_cohort = t + c + self.dims.drop(self.time_letter)
        return self._dims_cohort

    @property
    def _n_t(self) -> int:
        return list(self.shape)[0]

    @property
    def _shape_cohort(self) -> tuple:
        return (self._n_t,) + self.shape

    @property
    def _shape_no_t(self) -> tuple:
        return tuple(list(self.shape)[1:])

    @property
    def _t_diag_indices(self) -> tuple:
        return np.diag_indices(self._n_t) + (slice(None),) * len(self._shape_no_t)

    def get_outflow_by_cohort(self) -> StockArray:
        """Outflow by cohort, i.e. the outflow of each production year at each time step."""
        return self._get_by_cohort_array(self._outflow_by_cohort, "outflow_by_cohort")

    def get_stock_by_cohort(self) -> StockArray:
        """Stock by cohort, i.e. the stock of each production year at each time step."""
        return self._get_by_cohort_array(self._stock_by_cohort, "stock_by_cohort")

    def _get_by_cohort_array(self, values: np.ndarray, name: str) -> StockArray:
        if self.cohort_dim is None:  # leave in for backwards compatibility
            logging.warning(
                f"Cohort dimension is not defined; {name} cannot be retrieved as FlodymArray. Returning raw ndarray instead."
            )
            return values
        else:
            return StockArray(dims=self.dims_cohort, values=values, name=f"{self.name}_{name}")

    def _compute_cohorts(
        self,
        get_newest_cohort_inflow: Callable[[int, np.ndarray], np.ndarray],
        use_inflow_by_cohort: bool = False,
    ):
        """Step through time with the stock by cohort as state, and set stock, inflow and outflow.

        The inflow consists of two parts:

        - The older inflow by cohort is known before the time stepping. It contains the initial
          stock and, if ``use_inflow_by_cohort`` is True, the given ``inflow_by_cohort``.
          It can enter any cohort, also older ones.
        - The newest cohort inflow is the inflow of cohort t in year t. It is determined during
          the time stepping by ``get_newest_cohort_inflow``, from the prescribed inflow or from
          the prescribed stock.

        In every year t, the amount present by cohort is the stock by cohort at the end of year
        t-1 plus both parts of the inflow by cohort during year t. Multiplied with the one-step
        survival rate of the lifetime model, it gives the stock by cohort at the end of year t.
        The remainder is the outflow by cohort during year t.

        All amounts within this routine are over the whole period of a time step, not annual.

        Args:
            get_newest_cohort_inflow: Returns the newest cohort inflow of year index i_t, given
                i_t and the amount present by cohort in year i_t before adding this inflow, for
                the cohorts 0 to i_t.
            use_inflow_by_cohort: Whether ``inflow_by_cohort`` is part of the older inflow by
                cohort.
        """
        survival_rate = self.lifetime_model.survival_rate
        # shape (t, c, ...): inflow by cohort, known before the time stepping
        older_inflow_by_cohort = self._older_inflow_by_cohort()
        if use_inflow_by_cohort:
            older_inflow_by_cohort += self._to_whole_period(self.inflow_by_cohort.values)
        # shape (t, ...): inflow of cohort t in year t, determined during the time stepping
        newest_cohort_inflow = np.zeros(self.shape)
        # shape (t, c, ...): amount of cohort c present at any point during year t, i.e. the stock
        # at the end of year t-1 plus all inflow during year t. It is initialized with the older
        # inflow; the stock of the previous year and the newest cohort inflow are added below.
        present_by_cohort = older_inflow_by_cohort.copy()
        self._stock_by_cohort = np.zeros(self._shape_cohort)
        for i_t in range(self._n_t):
            # only cohorts c <= t can be present in year t
            existing = slice(None, i_t + 1)
            if i_t > 0:
                present_by_cohort[i_t, :i_t, ...] += self._stock_by_cohort[i_t - 1, :i_t, ...]
            newest_cohort_inflow[i_t, ...] = get_newest_cohort_inflow(
                i_t, present_by_cohort[i_t, existing, ...]
            )
            present_by_cohort[i_t, i_t, ...] += newest_cohort_inflow[i_t, ...]
            self._stock_by_cohort[i_t, existing, ...] = (
                present_by_cohort[i_t, existing, ...] * survival_rate[i_t, existing, ...]
            )
        inflow_whole_period = newest_cohort_inflow + older_inflow_by_cohort.sum(axis=1)
        # what is present during year t, but not left at its end, leaves during year t
        # (computed in place, present_by_cohort is not needed afterwards)
        present_by_cohort -= self._stock_by_cohort
        present_by_cohort /= self._t.interval_lengths.reshape((-1,) + (1,) * len(self.shape))
        self._outflow_by_cohort = present_by_cohort
        self.inflow.values[...] = self._to_annual(inflow_whole_period)
        self.stock.values[...] = self._stock_by_cohort.sum(axis=1)
        self.outflow.values[...] = self._outflow_by_cohort.sum(axis=1)

    def _older_inflow_by_cohort(self) -> np.ndarray:
        """Inflow by cohort over the whole period which is known before the time stepping,
        see :py:meth:`_compute_cohorts`. Contains the initial stock, if set, but not
        ``inflow_by_cohort``.
        """
        older_inflow_by_cohort = np.zeros(self._shape_cohort)
        if self._initial_stock_by_cohort is None:
            return older_inflow_by_cohort
        iiy = self._initial_year_index
        self._check_consistent_with_survival(
            self._initial_stock_by_cohort, self.lifetime_model.sf[iiy, ...], "initial_stock"
        )
        # The initial stock enters during the initial year, such that after the losses of that
        # year, the stock by cohort at the end of the initial year equals the initial stock.
        survival_rate = self.lifetime_model.survival_rate[iiy, ...]
        np.divide(
            self._initial_stock_by_cohort,
            survival_rate,
            out=older_inflow_by_cohort[iiy, ...],
            where=survival_rate > 0,
        )
        return older_inflow_by_cohort

    @staticmethod
    def _check_consistent_with_survival(values: np.ndarray, sf: np.ndarray, passed_as: str):
        if np.any((values > 0) & (sf == 0)):
            raise ValueError(
                f"The given {passed_as} is inconsistent with the lifetime model: survival function "
                f"is zero for some time/cohort combinations of the {passed_as}, which means that "
                f"no surviving {passed_as} can exist for these combinations."
            )

    def _is_initial_year(self, i_t: int) -> bool:
        return self._initial_stock_by_cohort is not None and i_t == self._initial_year_index

    def compute(self):
        """Compute the dynamic stock model in the mode given by which of stock, inflow and
        inflow_by_cohort is set (see :py:attr:`flodym.FlodymArray.is_set`).

        If none of them is set, but exactly one of them has non-zero values, that one is used,
        with a warning. To recompute or switch modes, use the explicit methods
        :py:meth:`compute_inflow_driven`, :py:meth:`compute_stock_driven` and
        :py:meth:`compute_inflow_by_cohort_driven`.
        """
        driving_array = self._choose_driving_array()
        if driving_array == "stock":
            self.compute_stock_driven()
        elif driving_array == "inflow":
            self.compute_inflow_driven()
        else:
            self.compute_inflow_by_cohort_driven()

    def _process_compute_mode(self) -> str:
        """Computation mode used by :py:meth:`_compute_process`: ``"stock"``, ``"inflow"`` or
        ``"inflow_by_cohort"``, as chosen by :py:meth:`compute` from which array is set.
        If none is set, ``"inflow"``, with the stock inflow computed from the process inflows."""
        set_arrays = [name for name, a in self._driving_array_candidates().items() if a.is_set]
        if len(set_arrays) > 1:
            # raises an error with explanation
            self._choose_driving_array()
        return set_arrays[0] if set_arrays else "inflow"

    def _compute_process(self, computation: "ProcessComputation"):
        """Compute the stock and the unknown in- and outflows of its process.

        If the stock is already computed, it is not computed again. Otherwise, the computation
        mode is given by :py:meth:`_process_compute_mode`. In inflow-driven mode, if the stock
        inflow is not set, it is the total of the process inflows (see
        :py:meth:`flodym.Process.compute`). The unknown in- and outflows of the process are then
        computed from the stock inflow and outflow, respectively.
        """
        computation.require_flows()
        inflow_total = self.inflow
        if not self.is_computed:
            mode = self._process_compute_mode()
            if mode == "inflow" and not self.inflow.is_set:
                # keep the total, which may have more dimensions than the stock inflow
                inflow_total = computation.compute_total(("in",))
                self._set_from_total(self.inflow, inflow_total, "inflow")
            if mode == "stock":
                self.compute_stock_driven()
            elif mode == "inflow":
                self.compute_inflow_driven()
            else:
                self.compute_inflow_by_cohort_driven()
        computation.compute_side(inflow_total, "in")
        computation.compute_side(self.outflow, "out")

    def _driving_array_candidates(self) -> dict[str, StockArray]:
        candidates = {"stock": self.stock, "inflow": self.inflow}
        if self.inflow_by_cohort is not None:
            candidates["inflow_by_cohort"] = self.inflow_by_cohort
        return candidates

    def _choose_driving_array(self) -> str:
        candidates = self._driving_array_candidates()
        set_arrays = [name for name, array in candidates.items() if array.is_set]
        if len(set_arrays) == 1:
            return set_arrays[0]
        if len(set_arrays) > 1:
            raise ValueError(
                f"Exactly one of stock, inflow and inflow_by_cohort of stock '{self.name}' must be "
                f"set (is_set=True) to choose the computation mode, but set are: {set_arrays}. "
                "To recompute or switch modes, use compute_inflow_driven(), compute_stock_driven() "
                "or compute_inflow_by_cohort_driven(), or call mark_all_unset() first."
            )
        non_zero_arrays = [name for name, array in candidates.items() if np.any(array.values != 0)]
        if len(non_zero_arrays) == 1:
            logging.warning(
                f"None of stock, inflow and inflow_by_cohort of stock '{self.name}' is set "
                f"(is_set=True). Computing from {non_zero_arrays[0]}, the only one with non-zero "
                f"values. Call mark_set() on it, or use compute_{non_zero_arrays[0]}_driven()."
            )
            return non_zero_arrays[0]
        raise ValueError(
            f"None of stock, inflow and inflow_by_cohort of stock '{self.name}' is set "
            f"(is_set=True), and {len(non_zero_arrays)} of them have non-zero values, so the "
            "computation mode cannot be chosen. Call mark_set() on the given one, or use "
            "compute_inflow_driven(), compute_stock_driven() or compute_inflow_by_cohort_driven()."
        )

    @_wrap_compute
    def compute_inflow_driven(self):
        """Compute stock and outflow from the given inflow and the lifetime model.

        The inflow need not be marked as set (see ``config.error_behaviors.unset_array_used``).
        """
        self._check_array_known(
            self.inflow, "Inflow", "This will lead to a zero stock and outflow."
        )
        if self._initial_stock_by_cohort is None:
            self._compute_inflow_driven_vectorized()
            return
        self._check_inflow_at_initial_year()
        self._compute_cohorts(self._newest_cohort_inflow_from_inflow)

    def _compute_inflow_driven_vectorized(self):
        """Same result as the time stepping in :py:meth:`_compute_cohorts`, but faster.
        Only valid without initial stock and inflow by cohort."""
        inflow_whole_period = self._to_whole_period(self.inflow.values)
        self._stock_by_cohort = np.einsum(
            "c...,tc...->tc...", inflow_whole_period, self.lifetime_model.sf
        )
        self._outflow_by_cohort = np.einsum(
            "c...,tc...,t->tc...",
            inflow_whole_period,
            self.lifetime_model.pdf,
            1.0 / self._t.interval_lengths,
        )
        self.stock.values[...] = self._stock_by_cohort.sum(axis=1)
        self.outflow.values[...] = self._outflow_by_cohort.sum(axis=1)

    @_wrap_compute
    def compute_stock_driven(self):
        """Compute inflow and outflow from the given stock and the lifetime model.

        This involves solving the lower triangular equation system A*x=b, where A is the survival
        function matrix, x is the inflow vector, and b is the stock vector.
        The stock need not be marked as set (see ``config.error_behaviors.unset_array_used``).
        """
        self._check_array_known(self.stock, "Stock", "This will lead to a zero inflow and outflow.")
        if self._initial_stock_by_cohort is not None:
            self._check_stock_at_initial_year()
        self._compute_cohorts(self._newest_cohort_inflow_from_stock)

    @_wrap_compute
    def compute_inflow_by_cohort_driven(self):
        """Compute inflow, stock and outflow from the given ``inflow_by_cohort`` and the lifetime
        model. Requires ``cohort_dim``.

        ``inflow_by_cohort`` need not be marked as set (see
        ``config.error_behaviors.unset_array_used``).
        """
        self._check_inflow_by_cohort_known()
        self._check_consistent_with_survival(
            self.inflow_by_cohort.values, self.lifetime_model.sf, "inflow_by_cohort"
        )
        self._compute_cohorts(self._no_newest_cohort_inflow, use_inflow_by_cohort=True)

    def _no_newest_cohort_inflow(self, i_t: int, present_by_cohort: np.ndarray) -> np.ndarray:
        """All inflow is part of the inflow by cohort, including that of the newest cohort."""
        return np.zeros(self._shape_no_t)

    def _newest_cohort_inflow_from_inflow(
        self, i_t: int, present_by_cohort: np.ndarray
    ) -> np.ndarray:
        """Prescribed inflow of the newest cohort over the whole period."""
        newest_cohort_inflow = self.inflow.values[i_t, ...] * self._t.interval_lengths[i_t]
        if not self._is_initial_year(i_t):
            return newest_cohort_inflow
        # So far, all cohorts are only present through the inflow of the initial stock.
        if self._initial_stock_included_in_inflow:
            # The prescribed inflow contains the inflow of the initial stock of all cohorts
            initial_stock_inflow = present_by_cohort.sum(axis=0)
            if np.any(newest_cohort_inflow < initial_stock_inflow):
                logging.warning(
                    f"Prescribed inflow in the initial stock year {self._initial_stock_year} is "
                    "smaller than the inflow of the initial stock it is declared to include. "
                    "The inflow of the initial stock will be used."
                )
        else:
            # The prescribed inflow and the initial stock overlap only in the newest cohort
            initial_stock_inflow = present_by_cohort[i_t, ...]
        # First take maximum of prescribed inflow and initial stock inflow,
        # then subtract initial stock contribution:
        # max(pi, isi) - isi = pi - min(pi, isi)
        newest_cohort_inflow -= np.minimum(newest_cohort_inflow, initial_stock_inflow)
        return newest_cohort_inflow

    def _newest_cohort_inflow_from_stock(
        self, i_t: int, present_by_cohort: np.ndarray
    ) -> np.ndarray:
        """Inflow of the newest cohort over the whole period, such that the total stock equals the
        prescribed stock.
        """
        # Maths behind implementation:
        # Solve square linear equation system
        #   sf * inflow = stock
        # where sf is a lower triangular matrix (since year >= cohort)
        # => in every row i:
        #   sum_{j=1...i} (sf_i,j inflow_j) = stock_i
        # solve for inflow_i:
        #   inflow_i = ( stock_i - sum_{j=1...i-1}(sf_i,j * inflow_j) ) / sf_ii
        # where the sum is the surviving stock of the older cohorts.
        survival_rate = self.lifetime_model.survival_rate[i_t, : i_t + 1, ...]
        surviving_stock = (present_by_cohort * survival_rate).sum(axis=0)
        newest_cohort_inflow = (self.stock.values[i_t, ...] - surviving_stock) / survival_rate[
            i_t, ...
        ]
        if self._is_initial_year(i_t):
            # use the maximum of prescribed and initial stock
            newest_cohort_inflow = np.maximum(newest_cohort_inflow, 0.0)
        return newest_cohort_inflow

    def _check_inflow_at_initial_year(self):
        if np.any(self.inflow.values[: self._initial_year_index, ...] > 0):
            raise ValueError(
                f"Prescribed inflow before the initial stock year {self._initial_stock_year} is non-zero."
            )
        if not self._initial_stock_included_in_inflow and np.any(
            self.inflow.values[self._initial_year_index, ...] > 0
        ):
            logging.warning(
                f"Prescribed inflow in the initial stock year {self._initial_stock_year} is non-zero. "
                f"This creates ambiguity: Initial stock and prescribed inflow both contribute to the stock of the cohort of that year. "
                f"The maximum of both will be used. "
                f"Moreover, the inflow in that year will be overwritten to introduce all initial stock, to ensure mass balance."
            )

    def _check_stock_at_initial_year(self):
        if np.any(self.stock.values[: self._initial_year_index, ...] > 0):
            raise ValueError(
                f"Prescribed stock before the initial stock year {self._initial_stock_year} is non-zero."
            )
        prescribed_stock = self.stock.values[self._initial_year_index, ...]
        if np.all(prescribed_stock == 0):
            logging.debug(
                f"Prescribed stock in the initial stock year {self._initial_stock_year} is zero and will be fully replaced by the initial stock."
            )
        elif not np.all(prescribed_stock == self._initial_stock_by_cohort.sum(axis=0)):
            logging.warning(
                f"Prescribed stock in the initial stock year {self._initial_stock_year} is non-zero and different from the total initial stock. "
                f"This creates ambiguity: Initial stock and prescribed stock both contribute to the stock of that year. "
                f"The maximum of both will be used."
            )

    def set_initial_stock(
        self, initial_stock: FlodymArray, initial_year: int, included_in_inflow: bool = False
    ):
        """Set an initial stock by cohort at the end of a given year.

        The initial stock enters the stock during the initial year, and is part of the inflow of
        that year. Prescribed inflow or stock before the initial year must be zero.
        For stock-driven models, the prescribed stock after the initial year is the total stock,
        including the survivors of the initial stock.

        Args:
            initial_stock (FlodymArray): Initial stock to be set. Must have same dimensions as the stock, except for time dimension.
            initial_year (int): Year in which the initial stock is given. Must be an item of the time dimension.
            included_in_inflow (bool): Only relevant for inflow-driven computation. If False
                (default), the prescribed inflow in the initial year is the inflow of the newest
                cohort, and the maximum of it and the initial stock of that cohort is used. If True,
                the prescribed inflow in the initial year already contains the inflow of the
                initial stock of all cohorts, as written by a previous computation, e.g. when an
                inflow corrected after a stock-driven computation is used to recompute
                inflow-driven.
        """
        if initial_stock.dims.letters != self.dims_cohort.drop(self.time_letter).letters:
            raise ValueError(
                f"Initial stock dimensions {initial_stock.dims.letters} do not match expected dims {self.dims_cohort.drop(self.time_letter).letters}."
            )
        if initial_year not in self.dims[self.time_letter].items:
            raise ValueError(
                f"Initial year {initial_year} is not in time dimension items {self.dims[self.time_letter].items}."
            )
        self._initial_stock_by_cohort = initial_stock.values.copy()
        self._initial_stock_year = initial_year
        self._initial_stock_included_in_inflow = included_in_inflow

    @property
    def _initial_year_index(self):
        return self.dims[self.time_letter].items.index(self._initial_stock_year)

    def copy(self) -> "Stock":
        """Return a copy of the Stock, as :py:meth:`flodym.Stock.copy`, but additionally
        giving the copy its own independent ``lifetime_model`` so that, for example, calling
        ``set_prms`` on the copy does not affect the original.
        """
        new_stock = super().copy()
        new_stock.lifetime_model = self.lifetime_model.model_copy(deep=True)
        if self.inflow_by_cohort is not None:
            new_stock.inflow_by_cohort = self.inflow_by_cohort.copy()
        if self._initial_stock_by_cohort is not None:
            new_stock._initial_stock_by_cohort = self._initial_stock_by_cohort.copy()
        new_stock._initial_stock_year = self._initial_stock_year
        new_stock._initial_stock_included_in_inflow = self._initial_stock_included_in_inflow
        return new_stock

    def __str__(self):
        base = super().__str__()
        lifetime_model = self.lifetime_model.__class__.__name__
        return base + "\n  Lifetime model: " + lifetime_model


class InflowDrivenDSM(DynamicStockModel):
    """Inflow driven model.
    Given inflow and lifetime distribution calculate stocks and outflows.
    Equivalent to :py:meth:`flodym.DynamicStockModel.compute_inflow_driven`.
    """

    def compute(self):
        """Determine stocks and outflows and store values in the class instance."""
        self.compute_inflow_driven()

    def _process_compute_mode(self) -> str:
        return "inflow"


class StockDrivenDSM(DynamicStockModel):
    """Stock driven model.
    Given total stock and lifetime distribution, calculate inflows and outflows.
    Equivalent to :py:meth:`flodym.DynamicStockModel.compute_stock_driven`.
    """

    def compute(self):
        """Determine inflows and outflows and store values in the class instance."""
        self.compute_stock_driven()

    def _process_compute_mode(self) -> str:
        if not self.stock.is_set:
            raise UnderdeterminedError(
                self.process,
                f"The stock of the stock-driven stock '{self.name}' is not set (is_set=False).",
            )
        return "stock"
