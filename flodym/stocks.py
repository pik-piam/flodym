"""Home to various `Stock` classes,
including flow-driven stocks and dynamic (lifetime-based) stocks.
"""

from abc import abstractmethod
import numpy as np
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, model_validator
from typing import Optional, Union, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .processes import Process
from .flodym_arrays import StockArray, FlodymArray
from .dimensions import DimensionSet
from .lifetime_models import LifetimeModel, UnevenTimeDim
from .config import config, handle_error, ErrorBehavior


def stock_compute_decorator(func):
    """Adds checks before and after every stock compute routine
    """

    def wrapper(self: 'Stock', *args, **kwargs):
        self._check_needed_arrays()
        func(self, *args, **kwargs)
        self.mark_computed()
        if config.checks.mass_balance_stocks:
            self.check_mass_balance()
    wrapper.is_decorated = True

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
    process: Optional['Process'] = None
    """Process the stock is associated with, if any. Needed for example for the mass balance."""
    time_letter: str = "t"
    """Letter of the time dimension in the dimensions set, to make sure it's the first one."""
    _t: UnevenTimeDim = None

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
    def add_to_process(self):
        if self.process is not None:
            self.process.stock = self
        return self

    @model_validator(mode="after")
    def check_compute_decorator(self):
        if not getattr(self.compute, 'is_decorated', False):
            raise RuntimeError(
                "Stock.compute method must have the stock_compute_decorator applied to it."
            )
        return self

    @model_validator(mode="after")
    def init_t(self):
        self._t = UnevenTimeDim(dim=self.dims[self.time_letter])
        return self

    @abstractmethod
    @stock_compute_decorator
    def compute(self):
        # Add stock_compute_decorator to all subclasses!
        pass

    @abstractmethod
    def _check_needed_arrays(self):
        pass

    @abstractmethod
    def compute_process(self):
        """Compute the stock based on the process inflows and outflows.
        This is called by the process when it computes its total inflow/outflow.
        """
        pass

    def mark_computed(self):
        self.inflow.mark_set()
        self.outflow.mark_set()
        self.stock.mark_set()

    @property
    def shape(self) -> tuple:
        """Shape of the stock, inflow, outflow arrays, defined by the dimensions."""
        return self.dims.shape

    @property
    def process_id(self) -> int:
        """ID of the process the stock is associated with."""
        return self.process.id

    @property
    def net_inflow(self) -> FlodymArray:
        return self.inflow - self.outflow

    @property
    def time_interval_length(self) -> FlodymArray:
        t = UnevenTimeDim(dim=self.dims[self.time_letter])
        return FlodymArray(dims=self.dims[self.time_letter,], values=t.interval_lengths)

    def to_stock_type(self, desired_stock_type: type, **kwargs):
        """Return an object of a new stock type with values and dimensions the same as the original.
        `**kwargs` can be used to pass additional model attributes as required by the desired stock
        type, if these are not contained in the original stock type.
        """
        return desired_stock_type(**self.__dict__, **kwargs)

    def tolerance(self) -> float:
        if config.absolute_tolerance is not None:
            return config.absolute_tolerance
        return config.relative_tolerance * max(
                self.inflow._absolute_float_precision,
                self.outflow._absolute_float_precision,
                self.stock._absolute_float_precision,
            )

    def check_mass_balance(self, tolerance: float = None, error_behavior: ErrorBehavior = None):
        """Compute mass balance, and check whether it is within a certain tolerance.
        Throw an error if it isn't.

        Args:
            tolerance (float, optional): The tolerance for the mass balance check.
                If None, takes the global config setting, which defaults to 100 times the numpy
                float precision, multiplied by the maximum absolute flow value.
            error_behavior (ErrorBehavior): What to do if the mass balance check fails.
                If None, takes the global config setting, which defaults to raising an error
        """
        max_error = np.max(np.abs(self.balance.values))
        if tolerance is None:
            tolerance = config.absolute_tolerance
        if error_behavior is None:
            error_behavior = config.error_behaviors.mass_balance
        if tolerance is None:
            tolerance = self.tolerance()
        if max_error > tolerance:
            message = f"In stock {self.name}: Mass balance check failed (error = {max_error})"
            handle_error(
                behavior=error_behavior,
                message=message,
            )
        else:
            logging.info(f"In stock {self.name}: Success - Mass balance is consistent!")

    @property
    def balance(self) -> FlodymArray:
        """Check whether inflow, outflow, and stock are balanced.
        If possible, the method returns the vector 'Balance', where Balance = inflow - outflow - stock_change
        """
        net_addition_to_stock = self.net_inflow * self.time_interval_length
        return net_addition_to_stock - self.stock_change

    @property
    def stock_change(self) -> FlodymArray:
        return self.stock.apply(np.diff, kwargs={"axis": 0, "prepend": 0})

    @property
    def is_computed(self) -> bool:
        """Check whether the stock has been computed, i.e. whether the stock, inflow, and outflow arrays
        are all set.
        """
        return (
            self.inflow.is_set
            and self.outflow.is_set
            and self.stock.is_set
        )

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
            not self.inflow.is_set
            and not self.outflow.is_set
        ):
            logging.warning("Neither inflow and outflow are set (is_set=False). If this is intended, perform mark_set() on one of them.")

    @stock_compute_decorator
    def compute(self):
        self._check_needed_arrays()
        annual_net_inflow = self.inflow.values - self.outflow.values
        net_inflow_whole_period = self._to_whole_period(annual_net_inflow)
        self.stock.values[...] = np.cumsum(net_inflow_whole_period, axis=0)

    def compute_process(self):
        if self.process.inflows:
            self.process.compute_total(try_sides=["in"])
            if self.inflow.dims - self.process._total.dims:
                names = ", ".join((self.inflow.dims - self.process._total.dims).names)
                raise ValueError(
                    f"In Process {self.process.name}: Stock inflow has dimensions {names} not contained in the "
                    "dimensions of the summed inflow Flows. Consider using less dimensions for the "
                    "stock or a preceding process with a dimension_splitter."
                )
            self.inflow[...] = self.process._total

        if self.process.outflows:
            self.process.compute_total(try_sides=["out"])
            if self.outflow.dims - self.process._total.dims:
                names = ", ".join((self.outflow.dims - self.process._total.dims).names)
                raise ValueError(
                    f"In Process {self.process.name}: Stock outflow has dimensions {names} not contained in the "
                    "dimensions of the summed outflow Flows. Consider using less dimensions for the "
                    "stock or a neighboring process with a dimension_splitter."
                )
            self.outflow[...] = self.process._total

        self.process.check_shares()
        self.compute()



class DynamicStockModel(Stock):
    """Parent class for dynamic stock models, which are based on stocks having a specified
    lifetime (distribution).
    """

    lifetime_model: Union[LifetimeModel, type]
    """Lifetime model, which contains the lifetime distribution function.
    Can be input either as a LifetimeModel subclass, or as an instance of a
    LifetimeModel subclass. For available subclasses, see `flodym.lifetime_models`.
    """
    _outflow_by_cohort: np.ndarray = None
    _stock_by_cohort: np.ndarray = None

    @model_validator(mode="after")
    def init_cohort_arrays(self):
        self._stock_by_cohort = np.zeros(self._shape_cohort)
        self._outflow_by_cohort = np.zeros(self._shape_cohort)
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

    def get_outflow_by_cohort(self) -> np.ndarray:
        """Outflow by cohort, i.e. the outflow of each production year at each time step."""
        return self._outflow_by_cohort

    def get_stock_by_cohort(self) -> np.ndarray:
        """Stock by cohort, i.e. the stock of each production year at each time step."""
        return self._stock_by_cohort

    def _compute_outflow(self):
        self._outflow_by_cohort = np.einsum(
            "c...,tc...->tc...", self.inflow.values, self.lifetime_model.pdf
        )
        self.outflow.values[...] = self._outflow_by_cohort.sum(axis=1)

    def __str__(self):
        base = super().__str__()
        lifetime_model = self.lifetime_model.__class__.__name__
        return base + "\n  Lifetime model: " + lifetime_model


class InflowDrivenDSM(DynamicStockModel):
    """Inflow driven model.
    Given inflow and lifetime distribution calculate stocks and outflows.
    """

    def _check_needed_arrays(self):
        super()._check_needed_arrays()
        if not self.inflow.is_set:
            logging.warning("Inflow is not set (is_set=False). If this is intended, perform mark_set() on it.")

    @stock_compute_decorator
    def compute(self):
        """Determine stocks and outflows and store values in the class instance."""
        self._compute_stock()
        self._compute_outflow()

    def _compute_stock(self):
        # for non-contiguous years, yearly inflow is multiplied with time interval length
        inflow_per_period = self._to_whole_period(self.inflow.values)
        self._stock_by_cohort = np.einsum(
            "c...,tc...->tc...", inflow_per_period, self.lifetime_model.sf
        )
        self.stock.values[...] = self._stock_by_cohort.sum(axis=1)

    def compute_process(self):
        if self.inflow.is_set:
            self.compute()
            self.process._total = self.inflow
            self.process.compute_flows(sides=["in"])
            self.process.check_shares(sides=["in"])
        else:
            self.process.compute_total(try_sides=["in"])
            self.inflow[...] = self.process._total
            self.compute()

        self.process._total = self.outflow
        self.process.apply_dimension_splitter(sides=["out"])
        self.process.compute_flows(sides=["out"])
        self.process.check_shares(sides=["out"])


class StockDrivenDSM(DynamicStockModel):
    """Stock driven model.
    Given total stock and lifetime distribution, calculate inflows and outflows.
    This involves solving the lower triangular equation system A*x=b,
    where A is the survival function matrix, x is the inflow vector, and b is the stock vector.
    """

    def _check_needed_arrays(self):
        super()._check_needed_arrays()
        if not self.stock.is_set:
            logging.warning("Stock is not set (is_set=False). If this is intended, perform mark_set() on it.")

    @stock_compute_decorator
    def compute(self):
        """Determine inflows and outflows and store values in the class instance."""
        self._compute_cohorts_and_inflow()
        self._compute_outflow()

    def _compute_cohorts_and_inflow(self):
        """With given total stock and lifetime distribution,
        the method builds the stock by cohort and the inflow.
        This involves solving the lower triangular equation system A*x=b,
        where A is the survival function matrix, x is the inflow vector, and b is the stock vector.
        """
        # Maths behind implementation:
        # Solve square linear equation system
        #   sf * inflow = stock
        # where sf is a lower triangular matrix (since year >= cohort)
        # => in every row i:
        #   sum_{j=1...i} (sf_i,j inflow_j) = stock_i
        # solve for inflow_i:
        #   inflow_i = ( stock_i - sum_{j=1...i-1}(sf_i,j * inflow_j) ) / sf_ii
        inflow_whole_period = np.zeros_like(self.inflow.values)
        for i in range(self._n_t):
            stock_i = self.stock.values[i, ...]
            sf_ij = self.lifetime_model.sf[i, :i, ...]
            inflow_j = inflow_whole_period[:i, ...]
            sf_ii = self.lifetime_model.sf[i, i, ...]

            inflow_whole_period[i, ...] = (stock_i - (sf_ij * inflow_j).sum(axis=0)) / sf_ii
        self.inflow.values[...] = self._to_annual(inflow_whole_period)
        self._stock_by_cohort = np.einsum(
            "c...,tc...->tc...", self.inflow.values, self.lifetime_model.sf
        )

    def compute_process(self):

        self.compute()

        self.process._total = self.outflow
        self.process.apply_dimension_splitter(sides=["out"])
        self.process.compute_flows(sides=["out"])
        self.process.check_shares(sides=["out"])

        self.process._total = self.inflow
        self.process.apply_dimension_splitter(sides=["in"])
        self.process.compute_flows(sides=["in"])
        self.process.check_shares(sides=["in"])


class FlexibleDSM(DynamicStockModel):
    """Computes either stock-driven or inflow-driven dynamic stock model, depending on which of the
    stock or inflow is set.
    """

    compute_stock_driven = StockDrivenDSM.compute
    compute_inflow_driven = InflowDrivenDSM.compute

    _compute_cohorts_and_inflow = StockDrivenDSM._compute_cohorts_and_inflow
    _compute_stock = InflowDrivenDSM._compute_stock

    compute_process_stock_driven = StockDrivenDSM.compute_process
    compute_process_inflow_driven = InflowDrivenDSM.compute_process

    def compute(self):
        if self.stock.is_set:
            self.compute_stock_driven()
        elif self.inflow.is_set:
            self.compute_inflow_driven()
        else:
            raise ValueError("Either stock or inflow must be set for FlexibleDSM.compute().")

    # replaces decorator, since inner functions are already decorated
    compute.is_decorated = True

    def compute_process(self):
        if self.stock.is_set:
            self.compute_process_stock_driven()
        else:
            self.compute_process_inflow_driven()
