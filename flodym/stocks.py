"""Home to various `Stock` classes,
including flow-driven stocks and dynamic (lifetime-based) stocks.
"""

from abc import abstractmethod
import numpy as np
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, model_validator, Field
from typing import Optional, Union
import logging

from .processes import Process
from .flodym_arrays import StockArray, FlodymArray
from .dimensions import Dimension, DimensionSet
from .lifetime_models import LifetimeModel, UnevenTimeDim


def stock_compute_decorator(func):
    """Adds checks before and after every stock compute routine"""

    def wrapper(self: "Stock", *args, **kwargs):
        self._check_needed_arrays()
        func(self, *args, **kwargs)
        self.mark_computed()

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
    process: Optional[Process] = None
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
    def check_compute_decorator(self):
        if not getattr(self.compute, "is_decorated", False):
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

    def to_stock_type(self, desired_stock_type: type, **kwargs):
        """Return an object of a new stock type with values and dimensions the same as the original.
        `**kwargs` can be used to pass additional model attributes as required by the desired stock
        type, if these are not contained in the original stock type.
        """
        return desired_stock_type(**self.__dict__, **kwargs)

    def check_stock_balance(self):
        balance = self.get_stock_balance()
        balance = np.max(np.abs(balance).sum(axis=0))
        if balance > 1:  # 1 tonne accuracy
            raise RuntimeError("Stock balance for dynamic stock model is too high: " + str(balance))
        elif balance > 0.001:
            print("Stock balance for model dynamic stock model is noteworthy: " + str(balance))

    def get_stock_balance(self) -> np.ndarray:
        """Check whether inflow, outflow, and stock are balanced.
        If possible, the method returns the vector 'Balance', where Balance = inflow - outflow - stock_change
        """
        dsdt = np.diff(
            self.stock.values, axis=0, prepend=0
        )  # stock_change(t) = stock(t) - stock(t-1)
        return self.inflow.values - self.outflow.values - dsdt

    @property
    def is_computed(self) -> bool:
        """Check whether the stock has been computed, i.e. whether the stock, inflow, and outflow arrays
        are all set.
        """
        return self.inflow.is_set and self.outflow.is_set and self.stock.is_set

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
        if not self.inflow.is_set and not self.outflow.is_set:
            logging.warning(
                "Neither inflow and outflow are set (is_set=False). If this is intended, perform mark_set() on one of them."
            )

    @stock_compute_decorator
    def compute(self):
        annual_net_inflow = self.inflow.values - self.outflow.values
        net_inflow_whole_period = self._to_whole_period(annual_net_inflow)
        self.stock.values[...] = np.cumsum(net_inflow_whole_period, axis=0)


class DynamicStockModel(Stock):
    """Parent class for dynamic stock models, which are based on stocks having a specified
    lifetime (distribution).
    """

    lifetime_model: Union[LifetimeModel, type]
    """Lifetime model, which contains the lifetime distribution function.
    Can be input either as a LifetimeModel subclass, or as an instance of a
    LifetimeModel subclass. For available subclasses, see `flodym.lifetime_models`.
    """
    cohort_dim: Optional[Dimension] = None

    _outflow_by_cohort: np.ndarray = None
    _stock_by_cohort: np.ndarray = None
    _dims_cohort: DimensionSet = None
    _initial_stock_dsm: "InflowByCohortDrivenDSM" = None
    _initial_stock_year: int = None

    @model_validator(mode="after")
    def validate_cohort_dim(self):
        if self.cohort_dim is not None:
            t = self.dims[self.time_letter]
            c = self.cohort_dim
            if c.letter == t.letter or c.name == t.name:
                raise ValueError(
                    "Cohort dimension letter and name must be different from time dimension letter and name."
                )
            if c.items != t.items:
                raise ValueError(
                    "Cohort dimension size must be the same as time dimension size."
                )
            if c.letter in self.dims.letters or c.name in self.dims.names:
                raise ValueError("Cohort dimension must not be part of the stock dimensions.")
            self._dims_cohort = t + c + self.dims.drop(self.time_letter, inplace=False)
        return self

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

    def _check_cohort_dim(self, application: str):
        if self.cohort_dim is None:
            raise ValueError(f"Cohort dimension must be provided at DSM initialization for {application} to work.")

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
            return StockArray(dims=self._dims_cohort, values=values, name=f"{self.name}_{name}")

    def _compute_outflow(self):
        self._outflow_by_cohort = np.einsum(
            "c...,tc...->tc...", self.inflow.values, self.lifetime_model.pdf
        )
        self.outflow.values[...] = self._outflow_by_cohort.sum(axis=1)

    def set_initial_stock(self, initial_stock: FlodymArray, initial_year: int):
        """Set an initial stock at a given year.
        The initial stock is added to the stock by cohort internally.

        Args:
            initial_stock (FlodymArray): Initial stock to be set. Must have same dimensions as the stock, except for time dimension.
            initial_year (int): Year in which the initial stock is given. Must be an item of the time dimension.
        """
        self._check_cohort_dim("set_initial_stock")
        if initial_stock.dims.letters != self._dims_cohort.drop(self.time_letter).letters:
            raise ValueError(
                f"Initial stock dimensions {initial_stock.dims.letters} do not match expected dims {self._dims_cohort.drop(self.time_letter).letters}."
            )
        if initial_year not in self.dims[self.time_letter].items:
            raise ValueError(
                f"Initial year {initial_year} is not in time dimension items {self.dims[self.time_letter].items}."
            )

        inflow_by_cohort = StockArray(dims=self._dims_cohort, name=f"{self.name}_inflow_by_cohort")
        inflow_by_cohort[{self.time_letter: initial_year}] = self._to_annual(initial_stock.values)
        if np.any((inflow_by_cohort.values > 0) & (self.lifetime_model.sf == 0)):
            raise ValueError("Initial stock is inconsistent with the lifetime model: survival function is zero for some cohort/inflow year combinations.")
        inflow_by_cohort.values[...] /= np.where(self.lifetime_model.sf == 0, 1, self.lifetime_model.sf)
        self._initial_stock_dsm = InflowByCohortDrivenDSM(
            dims=self.dims,
            cohort_dim=self.cohort_dim,
            lifetime_model=self.lifetime_model,
            inflow_by_cohort=inflow_by_cohort,
            name=f"{self.name}_ISD",
        )
        self._initial_stock_year = initial_year

    @property
    def _initial_year_index(self):
        return self.dims[self.time_letter].items.index(self._initial_stock_year)

    def _add_initial_stock_to_inflow(self):
        if self._initial_stock_dsm is not None:
            self._initial_stock_dsm.compute(stop_after="inflow")
            if np.any(self.inflow.values[:self._initial_year_index+1, ...] > 0):
                raise ValueError(f"Prescribed inflow before or in the initial stock year {self._initial_stock_year} is non-zero.")
            self.inflow.values[...] += self._initial_stock_dsm.inflow.values

    def _add_initial_stock_to_stock(self) -> np.ndarray:
        if self._initial_stock_dsm is not None:
            if np.any(self.stock.values[:self._initial_year_index+1, ...] > 0):
                raise ValueError(f"Prescribed stock before or in the initial stock year {self._initial_stock_year} is non-zero.")
            # Initial stock only gives the remaining(!) stock by cohort in one year;
            # From this, we first need to reconstruct what the total stock was over historic time
            # This needs two steps:
            # 1) Construct what inflow in a year would lead to the given initial stock cohort
            #    surviving
            # 2) Sum this up over time in an inflow-driven way to get total stock over historic time
            self._initial_stock_dsm.compute(stop_after="stock")
            iiy = self._initial_year_index
            self.stock.values[:iiy+1,...] = self._initial_stock_dsm.stock.values[:iiy+1,...]

    def __str__(self):
        base = super().__str__()
        lifetime_model = self.lifetime_model.__class__.__name__
        return base + "\n  Lifetime model: " + lifetime_model


class InflowDrivenDSM(DynamicStockModel):
    """Inflow driven model.
    Given inflow and lifetime distribution calculate stocks and outflows.
    """

    def _check_needed_arrays(self):
        DynamicStockModel._check_needed_arrays()
        if not self.inflow.is_set:
            logging.warning(
                "Inflow is not set (is_set=False). If this is intended, perform mark_set() on it."
            )

    @stock_compute_decorator
    def compute(self):
        """Determine stocks and outflows and store values in the class instance."""
        self._add_initial_stock_to_inflow()
        self._compute_stock()
        self._compute_outflow()

    def _compute_stock(self):
        # for non-contiguous years, yearly inflow is multiplied with time interval length
        inflow_per_period = self._to_whole_period(self.inflow.values)
        self._stock_by_cohort = np.einsum(
            "c...,tc...->tc...", inflow_per_period, self.lifetime_model.sf
        )
        self.stock.values[...] = self._stock_by_cohort.sum(axis=1)


class StockDrivenDSM(DynamicStockModel):
    """Stock driven model.
    Given total stock and lifetime distribution, calculate inflows and outflows.
    This involves solving the lower triangular equation system A*x=b,
    where A is the survival function matrix, x is the inflow vector, and b is the stock vector.
    """

    def _check_needed_arrays(self):
        DynamicStockModel._check_needed_arrays()
        if not self.stock.is_set:
            logging.warning(
                "Stock is not set (is_set=False). If this is intended, perform mark_set() on it."
            )

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

        self._add_initial_stock_to_stock()

        for i in range(self._n_t):
            stock_i = self.stock.values[i, ...]
            sf_ij = self.lifetime_model.sf[i, :i, ...]
            inflow_j = inflow_whole_period[:i, ...]
            sf_ii = self.lifetime_model.sf[i, i, ...]

            inflow_whole_period[i, ...] = (stock_i - (sf_ij * inflow_j).sum(axis=0)) / sf_ii
        self.inflow.values[...] = self._to_annual(inflow_whole_period)
        self._stock_by_cohort = np.einsum(
            "c...,tc...->tc...", inflow_whole_period, self.lifetime_model.sf
        )


class InflowByCohortDrivenDSM(InflowDrivenDSM):

    inflow_by_cohort: StockArray = None
    cohort_dim: Dimension  # require for this subclass

    @model_validator(mode="after")
    def init_cohort_arrays(self):
        if inflow_by_cohort is None:
            inflow_by_cohort = StockArray(dims=self._dims_cohort, name=f"{self.name}_inflow_by_cohort")
        else:
            if inflow_by_cohort.dims.letters != self._dims_cohort.letters:
                raise ValueError(
                    f"Inflow by cohort dimensions {inflow_by_cohort.dims.letters} do not match expected dims {self._dims_cohort.letters}."
                )
        return self

    def _check_needed_arrays(self):
        DynamicStockModel._check_needed_arrays()
        if not self.inflow_by_cohort.is_set:
            logging.warning(
                "Inflow_by_cohort is not set (is_set=False). If this is intended, perform mark_set() on it."
            )
        if self.inflow.is_set:
            logging.warning(
                "Inflow is not set (is_set=True). It will be overwritten. Use either inflow or inflow_by_cohort."
            )

    def _compute_inflow(self):
        self.inflow.values[...] = np.einsum(
            "tc...,tc...->c...", self.inflow_by_cohort, 1/self.lifetime_model.sf
        )

    @stock_compute_decorator
    def compute(self, stop_after: str = None):
        """Determine stocks and outflows and store values in the class instance."""
        self._compute_inflow()
        if stop_after == "inflow":
            return
        self._compute_stock()
        if stop_after == "stock":
            return
        self._compute_outflow()


class FlexibleDSM(DynamicStockModel):
    """Computes either stock-driven or inflow-driven dynamic stock model, depending on which of the
    stock or inflow is set.
    """

    compute_stock_driven = StockDrivenDSM.compute
    compute_inflow_driven = InflowDrivenDSM.compute

    _compute_cohorts_and_inflow = StockDrivenDSM._compute_cohorts_and_inflow
    _compute_stock = InflowDrivenDSM._compute_stock

    def compute(self):
        if self.stock.is_set:
            self.compute_stock_driven()
        elif self.inflow.is_set:
            self.compute_inflow_driven()
        else:
            raise ValueError("Either stock or inflow must be set for FlexibleDSM.compute().")

    # replaces decorator, since inner functions are already decorated
    compute.is_decorated = True
