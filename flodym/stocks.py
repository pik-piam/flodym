"""Home to various `Stock` classes,
including flow-driven stocks and dynamic (lifetime-based) stocks.
"""

from abc import abstractmethod
import numpy as np
from scipy.linalg import solve_triangular
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, model_validator
from typing import Optional, Union
import logging

from .processes import Process
from .flodym_arrays import StockArray, FlodymArray
from .dimensions import DimensionSet
from .lifetime_models import LifetimeModel, UnevenTimeDim


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
    def init_t(self):
        self._t = UnevenTimeDim(dim=self.dims[self.time_letter])
        return self

    @abstractmethod
    def compute(self):
        # always add this check first
        self._check_needed_arrays()

    @abstractmethod
    def _check_needed_arrays(self):
        pass

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

    def compute(self):
        self._check_needed_arrays()
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
        if np.allclose(self.inflow.values, np.zeros(self.shape)):
            logging.warning("Inflow is zero. This will lead to a zero stock and outflow.")

    def compute(self):
        """Determine stocks and outflows and store values in the class instance."""
        self._check_needed_arrays()
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

    solver: str = "manual"
    """Algorithm to use for solving the equation system.  Options are: "manual" (default), which uses
    an own python implementation, and "lapack", which calls the lapack trtrs routine via scipy.
    The lapack implementation may be more precise. Speed depends on the dimensionality,
    but the manual implementation is usually faster.
    """

    @model_validator(mode="after")
    def init_solver(self):
        if self.solver not in ["manual", "lapack"]:
            raise ValueError("Solver must be either 'manual' or 'lapack'.")
        return self

    def _check_needed_arrays(self):
        super()._check_needed_arrays()
        if np.allclose(self.stock.values, np.zeros(self.shape)):
            logging.warning("Stock is zero. This will lead to a zero inflow and outflow.")

    def compute(self):
        """Determine inflows and outflows and store values in the class instance."""
        self._check_needed_arrays()
        self._compute_cohorts_and_inflow()
        self._compute_outflow()

    def _compute_cohorts_and_inflow(self):
        """With given total stock and lifetime distribution,
        the method builds the stock by cohort and the inflow.
        This involves solving the lower triangular equation system A*x=b,
        where A is the survival function matrix, x is the inflow vector, and b is the stock vector.
        """
        if self.solver == "manual":
            self._compute_inflow_manual()
        elif self.solver == "lapack":
            self._compute_inflow_lapack()
        else:
            raise ValueError(f"Unknown engine: {self.solver}")

        self._stock_by_cohort = np.einsum(
            "c...,tc...->tc...", self.inflow.values, self.lifetime_model.sf
        )

    def _compute_inflow_manual(self) -> tuple[np.ndarray]:
        """With given total stock and lifetime distribution,
        the method builds the stock by cohort and the inflow,
        using a manual algorithm for solving of the equation system (see "solver" doc for details).
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

    def _compute_inflow_lapack(self) -> tuple[np.ndarray]:
        """With given total stock and lifetime distribution,
        the method builds the stock by cohort and the inflow,
        using lapack for solving of the equation system (see "engine" doc for details).
        """
        sf = self.lifetime_model.sf
        slt = (slice(None),)
        inflow_whole_period = np.zeros_like(self.inflow.values)
        for i in np.ndindex(self._shape_no_t):
            inflow_whole_period[slt + i] = solve_triangular(
                sf[2 * slt + i], self.stock.values[slt + i], lower=True
            )
        self.inflow.values[...] = self._to_annual(inflow_whole_period)
