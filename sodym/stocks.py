from abc import abstractmethod
import numpy as np
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, model_validator
from typing import Optional
from .survival_functions import SurvivalModel
from .named_dim_arrays import StockArray, Process


class Stock(PydanticBaseModel):
    """Stock objects are components of an MFASystem, where materials can accumulate over time.
    They consist of three :py:class:`sodym.named_dim_arrays.NamedDimArrays` : stock (the accumulation), inflow, outflow.

    The base class only allows to compute the stock from known inflow and outflow.
    The subclasses allows computations using a lifetime distribution function, which is necessary if not both
    inflow and outflow are known.
    """
    model_config = ConfigDict(protected_namespaces=(), arbitrary_types_allowed=True)

    stock: Optional[StockArray] = None
    inflow: Optional[StockArray] = None
    outflow: Optional[StockArray] = None
    name: str
    process_name: Optional[str] = None
    process: Process

    @model_validator(mode="after")
    def check_process_names(self):
        if self.process_name and self.process.name != self.process_name:
            raise ValueError("Missmatching process names in Stock object")
        self.process_name = self.process.name
        return self

    @abstractmethod
    def compute(self):
        pass

    @property
    def shape(self):
        return self.stock.dims.shape()

    @property
    def process_id(self):
        return self.process.id

    def check_stock_balance(self):
        balance = self.get_stock_balance()
        balance = np.max(np.abs(balance).sum(axis=0))
        if balance > 1:  # 1 tonne accuracy
            raise RuntimeError("Stock balance for dynamic stock model is too high: " + str(balance))
        elif balance > 0.001:
            print("Stock balance for model dynamic stock model is noteworthy: " + str(balance))

    def get_stock_balance(self):
        """Check whether inflow, outflow, and stock are balanced.
        If possible, the method returns the vector 'Balance', where Balance = inflow - outflow - stock_change
        """
        dsdt = np.diff(self.stock.values, axis=0, prepend=0)  #stock_change(t) = stock(t) - stock(t-1)
        return self.inflow.values - self.outflow.values - dsdt


class FlowDrivenStock(Stock):
    def compute(self):
        stock_vals = np.cumsum(self.inflow.values - self.outflow.values, axis=self.stock.dims.index("t"))
        self.stock = StockArray(
            dims=self.inflow.dims, name=f'{self.name}_stock', values=stock_vals
        )


class DynamicStockModel(Stock):
    survival_model: SurvivalModel

    @property
    def n_t(self):
        return list(self.shape)[0]

    @property
    def shape_cohort(self):
        return (self.n_t, ) + self.shape

    @property
    def shape_no_t(self):
        return tuple(list(self.shape)[1:])

    @property
    def t_diag_indices(self):
        return np.diag_indices(self.n_t) + (slice(None),) * len(self.shape_no_t)


class InflowDrivenDSM(DynamicStockModel):
    """Inflow driven model
    Given: inflow, lifetime dist.
    Default order of methods:
    1) determine stock by cohort
    2) determine total stock
    2) determine outflow by cohort
    3) determine total outflow
    4) check mass balance.
    """
    inflow: StockArray

    @property
    def shape(self):
        return self.inflow.dims.shape()

    def compute(self):
        stock_by_cohort = self.compute_i_lt__2__sc()
        outflow_by_cohort = self.compute_sc__2__oc(stock_by_cohort)
        stock_vals = stock_by_cohort.sum(axis=1)
        outflow_vals = outflow_by_cohort.sum(axis=1)
    
        self.stock = StockArray(
            dims=self.inflow.dims, values=stock_vals, name=f'{self.name}_stock',
        )
        self.outflow = StockArray(
            dims=self.inflow.dims, values=outflow_vals, name=f'{self.name}_outflow'
        )

    def compute_i_lt__2__sc(self):
        """With given inflow and lifetime distribution, the method builds the stock by cohort."""
        stock_by_cohort = np.einsum("c...,tc...->tc...", self.inflow.values, self.survival_model.sf)
        # This command means: s_c[t,c] = i[c] * sf[t,c] for all t, c
        # from the perspective of the stock the inflow has the dimension age-cohort,
        # as each inflow(t) is added to the age-cohort c = t
        return stock_by_cohort

    def compute_sc__2__oc(self, stock_by_cohort):
        """Compute outflow by cohort from stock by cohort."""
        outflow_by_cohort = np.zeros(self.shape_cohort)
        outflow_by_cohort[1:, :, ...] = -np.diff(stock_by_cohort, axis=0)
        outflow_by_cohort[self.t_diag_indices] = self.inflow.values - np.moveaxis(
            stock_by_cohort.diagonal(0, 0, 1), -1, 0
        )  # allow for outflow in year 0 already
        return outflow_by_cohort


class StockDrivenDSM(DynamicStockModel):
    """Stock driven model
    Given: total stock, lifetime dist.
    Default order of methods:
    1) determine inflow, outflow by cohort, and stock by cohort
    2) determine total outflow
    3) determine stock change
    4) check mass balance.
    """
    stock: StockArray

    def compute(self):
        inflow_vals, outflow_by_cohort, stock_by_cohort = self.compute_s_lt__2__sc_oc_i()
        outflow_vals = outflow_by_cohort.sum(axis=1)
        self.inflow = StockArray(
            dims=self.stock.dims, values=inflow_vals, name=f'{self.name}_inflow',
        )
        self.outflow = StockArray(
            dims=self.stock.dims, values=outflow_vals, name=f'{self.name}_outflow'
        )

    def compute_s_lt__2__sc_oc_i(self, do_correct_negative_inflow=False):
        """With given total stock and lifetime distribution, the method builds the stock by cohort and the inflow."""
        stock_by_cohort = np.zeros(self.shape_cohort)
        outflow_by_cohort = np.zeros(self.shape_cohort)
        inflow = np.zeros(self.shape)
        sf = self.survival_model.sf
        # construct the sf of a product of cohort tc remaining in the stock in year t
        # First year:
        inflow[0, ...] = np.where(sf[0, 0, ...] != 0.0, self.stock.values[0] / sf[0, 0], 0.0)
        stock_by_cohort[:, 0, ...] = (
            inflow[0, ...] * sf[:, 0, ...]
        )  # Future decay of age-cohort of year 0.
        outflow_by_cohort[0, 0, ...] = inflow[0, ...] - stock_by_cohort[0, 0, ...]
        # all other years:
        for m in range(1, self.n_t):  # for all years m, starting in second year
            # 1) Compute outflow from previous age-cohorts up to m-1
            outflow_by_cohort[m, 0:m, ...] = (
                stock_by_cohort[m - 1, 0:m, ...] - stock_by_cohort[m, 0:m, ...]
            )  # outflow table is filled row-wise, for each year m.
            # 2) Determine inflow from mass balance:
            if not do_correct_negative_inflow:  # if no correction for negative inflows is made
                inflow[m, ...] = np.where(
                    sf[m, m, ...] != 0.0,
                    (self.stock.values[m, ...] - stock_by_cohort[m, :, ...].sum(axis=0)) / sf[m, m, ...],
                    0.0,
                )  # allow for outflow during first year by rescaling with 1/sf[m,m]
                # 3) Add new inflow to stock and determine future decay of new age-cohort
                stock_by_cohort[m::, m, ...] = inflow[m, ...] * sf[m::, m, ...]
                outflow_by_cohort[m, m, ...] = inflow[m, ...] * (1 - sf[m, m, ...])
            # 2a) Correct remaining stock in cases where inflow would be negative:
            else:
                # if the stock declines faster than according to the lifetime model, this option allows to extract
                # additional stock items.
                # The negative inflow correction implemented here was developed in a joined effort by Sebastiaan Deetman
                # and Stefan Pauliuk.
                inflow_test = self.stock.values[m, ...] - stock_by_cohort[m, :, ...].sum(axis=0)
                if inflow_test < 0:  # if stock-driven model would yield negative inflow
                    delta = -1 * inflow_test  # Delta > 0!
                    inflow[m, ...] = 0  # Set inflow to 0 and distribute mass balance gap onto remaining cohorts:
                    delta_percent = np.where(
                        stock_by_cohort[m, :, ...].sum(axis=0) != 0,
                        delta / stock_by_cohort[m, :, ...].sum(axis=0),
                        0.0,
                    )
                    # - Distribute gap equally across all cohorts (each cohort is adjusted by the same %, based on
                    #   surplus with regards to the prescribed stock)
                    # - delta_percent is a % value <= 100%
                    # - correct for outflow and stock in current and future years
                    # - adjust the entire stock AFTER year m as well, stock is lowered in year m, so future cohort
                    #   survival also needs to decrease.

                    # increase outflow according to the lost fraction of the stock, based on Delta_c
                    outflow_by_cohort[m, :, ...] = outflow_by_cohort[m, :, ...] + (
                        stock_by_cohort[m, :, ...] * delta_percent
                    )
                    # shrink future description of stock from previous age-cohorts by factor Delta_percent in current
                    # AND future years.
                    stock_by_cohort[m::, 0:m, ...] = (
                        stock_by_cohort[m::, 0:m, ...] * (1 - delta_percent)
                    )
                else:  # If no negative inflow would occur
                    inflow[m, ...] = np.where(
                        sf[m, m, ...] != 0,  # Else, inflow is 0.
                        (self.stock.values[m, ...] - stock_by_cohort[m, :, ...].sum(axis=0))
                        / sf[m, m, ...],  # allow for outflow during first year by rescaling with 1/sf[m,m]
                        0.0,
                    )
                    # Add new inflow to stock and determine future decay of new age-cohort
                    stock_by_cohort[m::, m, ...] = inflow[m, ...] * sf[m::, m, ...]
                    outflow_by_cohort[m, m, ...] = inflow[m, ...] * (1 - sf[m, m, ...])
                # NOTE: This method of negative inflow correction is only of of many plausible methods of increasing the
                # outflow to keep matching stock levels. It assumes that the surplus stock is removed in the year that
                # it becomes obsolete. Each cohort loses the same fraction. Modellers need to try out whether this
                # method leads to justifiable results. In some situations it is better to change the lifetime assumption
                # than using the NegativeInflowCorrect option.

        return inflow, outflow_by_cohort, stock_by_cohort