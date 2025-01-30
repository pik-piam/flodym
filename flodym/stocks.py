"""Home to various `Stock` classes,
including flow-driven stocks and dynamic (lifetime-based) stocks.
"""

from abc import abstractmethod
import numpy as np
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, model_validator
from typing import Optional, Union
import logging

from .processes import Process
from .flodym_arrays import StockArray
from .dimensions import DimensionSet
from .lifetime_models import LifetimeModel


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
        return self.dims.shape()

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


class SimpleFlowDrivenStock(Stock):
    """Given inflows and outflows, the stock can be calculated."""

    def _check_needed_arrays(self):
        if (
            np.max(np.abs(self.inflow.values)) < 1e-10
            and np.max(np.abs(self.outflow.values)) < 1e-10
        ):
            logging.warning("Inflow and Outflow are zero. This will lead to a zero stock.")

    def compute(self):
        self._check_needed_arrays()
        self.stock.values[...] = np.cumsum(self.inflow.values - self.outflow.values, axis=0)


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
        self.compute_stock_by_cohort()
        self.compute_outflow_by_cohort()
        self.stock.values[...] = self._stock_by_cohort.sum(axis=1)
        self.outflow.values[...] = self._outflow_by_cohort.sum(axis=1)

    def compute_stock_by_cohort(self) -> np.ndarray:
        """With given inflow and lifetime distribution, the method builds the stock by cohort.
        s_c[t,c] = i[c] * sf[t,c] for all t, c
        from the perspective of the stock the inflow has the dimension age-cohort,
        as each inflow(t) is added to the age-cohort c = t
        """
        self._stock_by_cohort = np.einsum(
            "c...,tc...->tc...", self.inflow.values, self.lifetime_model.sf
        )

    def compute_outflow_by_cohort(self) -> np.ndarray:
        """Compute outflow by cohort from changes in the stock by cohort and the known inflow."""
        self._outflow_by_cohort[1:, :, ...] = -np.diff(self._stock_by_cohort, axis=0)
        # allow for outflow in year 0 already
        self._outflow_by_cohort[self._t_diag_indices] = self.inflow.values - np.moveaxis(
            self._stock_by_cohort.diagonal(0, 0, 1), -1, 0
        )


class StockDrivenDSM(DynamicStockModel):
    """Stock driven model.
    Given total stock and lifetime distribution, calculate inflows and outflows.
    """

    def _check_needed_arrays(self):
        super()._check_needed_arrays()
        if np.allclose(self.stock.values, np.zeros(self.shape)):
            logging.warning("Stock is zero. This will lead to a zero inflow and outflow.")

    def compute(self):
        """Determine inflows and outflows and store values in the class instance."""
        self._check_needed_arrays()
        self.compute_inflow_and_outflow()
        self.outflow.values[...] = self._outflow_by_cohort.sum(axis=1)

    def compute_inflow_and_outflow(self) -> tuple[np.ndarray]:
        """With given total stock and lifetime distribution,
        the method builds the stock by cohort and the inflow."""
        sf = self.lifetime_model.sf
        # construct the sf of a product of cohort tc remaining in the stock in year t
        # First year:
        self.inflow.values[0, ...] = np.where(
            sf[0, 0, ...] != 0.0, self.stock.values[0] / sf[0, 0], 0.0
        )
        # Future decay of age-cohort of year 0.
        self._stock_by_cohort[:, 0, ...] = self.inflow.values[0, ...] * sf[:, 0, ...]
        self._outflow_by_cohort[0, 0, ...] = (
            self.inflow.values[0, ...] - self._stock_by_cohort[0, 0, ...]
        )
        # all other years:
        for m in range(1, self._n_t):  # for all years m, starting in second year
            # 1) Compute outflow from previous age-cohorts up to m-1
            # outflow table is filled row-wise, for each year m.
            self._outflow_by_cohort[m, 0:m, ...] = (
                self._stock_by_cohort[m - 1, 0:m, ...] - self._stock_by_cohort[m, 0:m, ...]
            )
            self.inflow_from_balance(m)

    def inflow_from_balance(self, m: int) -> np.ndarray:
        """determine inflow from mass balance and do not correct negative inflow"""

        sf = self.lifetime_model.sf
        # allow for outflow during first year by rescaling with 1/sf[m,m]
        self.inflow.values[m, ...] = np.where(
            sf[m, m, ...] != 0.0,
            (self.stock.values[m, ...] - self._stock_by_cohort[m, :, ...].sum(axis=0))
            / sf[m, m, ...],
            0.0,
        )
        # 3) Add new inflow to stock and determine future decay of new age-cohort
        self._stock_by_cohort[m::, m, ...] = self.inflow.values[m, ...] * sf[m::, m, ...]
        self._outflow_by_cohort[m, m, ...] = self.inflow.values[m, ...] * (1 - sf[m, m, ...])


class StockDrivenDSM_NIC(StockDrivenDSM):

    def inflow_from_balance(self, m):
        is_negative_inflow = self.check_negative_inflow(m)
        if is_negative_inflow:
            self.inflow_from_balance_correction(m)
        else:
            super().inflow_from_balance(m)

    def check_negative_inflow(self, m: int) -> bool:
        """Check if inflow is negative."""
        inflow_test = self.stock.values[m, ...] - self._stock_by_cohort[m, :, ...].sum(axis=0)
        return inflow_test < 0

    def inflow_from_balance_correction(self, m: int) -> np.ndarray:
        """determine inflow from mass balance and correct negative inflow

        NOTE: This method of negative inflow correction is only one of many plausible methods of increasing the
        outflow to keep matching stock levels. It assumes that the surplus stock is removed in the year that
        it becomes obsolete. Each cohort loses the same fraction. Modellers need to try out whether this
        method leads to justifiable results. In some situations it is better to change the lifetime assumption
        than using the NegativeInflowCorrect option.
        """

        # if the stock declines faster than according to the lifetime model, this option allows to extract
        # additional stock items.
        # The negative inflow correction implemented here was developed in a joined effort by Sebastiaan Deetman
        # and Stefan Pauliuk.

        # delta = -inflow
        delta = self._stock_by_cohort[m, :, ...].sum(axis=0) - self.stock.values[m, ...]
        # Set inflow to 0 and distribute mass balance gap onto remaining cohorts:
        self.inflow.values[m, ...] = 0
        delta_percent = np.where(
            self._stock_by_cohort[m, :, ...].sum(axis=0) != 0,
            delta / self._stock_by_cohort[m, :, ...].sum(axis=0),
            0.0,
        )
        # - Distribute gap equally across all cohorts (each cohort is adjusted by the same %, based on
        #   surplus with regards to the prescribed stock)
        # - delta_percent is a % value <= 100%
        # - correct for outflow and stock in current and future years
        # - adjust the entire stock AFTER year m as well, stock is lowered in year m, so future cohort
        #   survival also needs to decrease.

        # increase outflow according to the lost fraction of the stock, based on Delta_c
        self._outflow_by_cohort[m, :, ...] = self._outflow_by_cohort[m, :, ...] + (
            self._stock_by_cohort[m, :, ...] * delta_percent
        )
        # shrink future description of stock from previous age-cohorts by factor Delta_percent in current
        # AND future years.
        self._stock_by_cohort[m::, 0:m, ...] = self._stock_by_cohort[m::, 0:m, ...] * (
            1 - delta_percent
        )
