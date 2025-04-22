import numpy as np
import pytest
import time

from flodym.dimensions import Dimension, DimensionSet
from flodym.flodym_arrays import StockArray
from flodym.stocks import InflowDrivenDSM, StockDrivenDSM
from flodym.lifetime_models import LogNormalLifetime

dim_list = [
    Dimension(
        name="time",
        letter="t",
        items=list(range(1900, 2101)),
        dtype=int,
    ),
    Dimension(
        name="product",
        letter="a",
        items=["automotive", "construction"],
        dtype=str,
    ),
]

dims = DimensionSet(dim_list=dim_list)


def test_stocks():
    inflow_values = np.exp(-np.linspace(-2, 2, 201) ** 2)
    inflow_values = np.stack([inflow_values, inflow_values]).T
    inflow = StockArray(dims=dims, values=inflow_values)

    lifetime_model = LogNormalLifetime(dims=dims, time_letter="t", mean=60, std=25)
    inflow_driven_dsm = InflowDrivenDSM(
        dims=dims,
        inflow=inflow,
        lifetime_model=lifetime_model,
        time_letter="t",
    )
    inflow_driven_dsm.compute()
    stock_fda = inflow_driven_dsm.stock

    stock_driven_dsm = StockDrivenDSM(
        dims=dims,
        stock=stock_fda,
        lifetime_model=lifetime_model,
        time_letter="t",
        solver="manual",
    )
    stock_driven_dsm.compute()
    inflow_post = stock_driven_dsm.inflow
    assert np.allclose(inflow.values, inflow_post.values)

    stock_driven_dsm = StockDrivenDSM(
        dims=dims,
        stock=stock_fda,
        lifetime_model=lifetime_model,
        time_letter="t",
        solver="lapack",
    )
    stock_driven_dsm.compute()
    inflow_post_invert = stock_driven_dsm.inflow
    assert np.allclose(inflow.values, inflow_post_invert.values)
    # return inflow, inflow_post, inflow_post_invert


def test_lifetime_quadrature():
    # Put in constant inflow and check stationary stock values

    # Long lifetimes:
    # Inflow at start/end of time step under/overestimate stock by half a year,
    # others should work well
    inflow, stocks = get_stocks_by_quadrature(mean=30, std=10)
    targets = {
        "ltm_start": 29.5,
        "ltm_end": 30.5,
        "ltm_middle": 30,
        "ltm_2": 30,
        "ltm_6": 30,
    }
    eps = 0.01
    for name, stock in stocks.items():
        assert np.abs(stock["automotive"].values[-1] - targets[name]) < eps

    # Short lifetimes:
    # only high-order quadrature should work well
    inflow, stocks = get_stocks_by_quadrature(mean=0.3, std=0.1)
    for name, stock in stocks.items():
        if name == "ltm_6":
            assert np.abs(stock["automotive"].values[-1] - 0.3) < eps
        else:
            assert np.abs(stock["automotive"].values[-1] - 0.3) > eps


def get_stocks_by_quadrature(mean, std):
    inflow_values = np.exp(-np.linspace(-2, 2, 201) ** 2)
    inflow_values = np.stack([inflow_values, inflow_values]).T
    inflow_values = np.ones_like(inflow_values)
    inflow = StockArray(dims=dims, values=inflow_values)

    lifetime_models = {
        "ltm_start": LogNormalLifetime(
            dims=dims, time_letter="t", mean=mean, std=std, inflow_at="start"
        ),
        "ltm_end": LogNormalLifetime(
            dims=dims, time_letter="t", mean=mean, std=std, inflow_at="end"
        ),
        "ltm_middle": LogNormalLifetime(
            dims=dims, time_letter="t", mean=mean, std=std, inflow_at="middle"
        ),
        "ltm_2": LogNormalLifetime(
            dims=dims, time_letter="t", mean=mean, std=std, n_pts_per_interval=2
        ),
        "ltm_6": LogNormalLifetime(
            dims=dims, time_letter="t", mean=mean, std=std, n_pts_per_interval=6
        ),
    }
    stocks = {}
    for name, lifetime_model in lifetime_models.items():
        inflow_driven_dsm = InflowDrivenDSM(
            dims=dims,
            inflow=inflow,
            lifetime_model=lifetime_model,
            time_letter="t",
        )
        inflow_driven_dsm.compute()
        stocks[name] = inflow_driven_dsm.stock
    return inflow, stocks


def test_unequal_time_steps():
    """Unequal time step lengths should not influence the development of the stock apart from numerical errors."""
    # case with all years
    t_all = Dimension(
        name="time",
        letter="t",
        items=list(range(1900, 2101)),
        dtype=int,
    )
    # case with only a few select unequally spaced years
    t_select_items = [i for i in t_all.items if i % 13 == 0 or i % 7 == 0]
    t_select = Dimension(
        name="time",
        letter="s",
        items=t_select_items,
        dtype=int,
    )
    stocks = []
    for t in t_all, t_select:
        dims = DimensionSet(dim_list=[t])
        lifetime_model = LogNormalLifetime(
            dims=dims, time_letter=t.letter, mean=5, std=2, n_pts_per_interval=6
        )
        inflow_values = np.ones(t.len)
        inflow = StockArray(dims=dims, values=inflow_values)
        inflow_driven_dsm = InflowDrivenDSM(
            dims=dims,
            inflow=inflow,
            lifetime_model=lifetime_model,
            time_letter=t.letter,
        )
        inflow_driven_dsm.compute()
        stocks.append(inflow_driven_dsm.stock)
    stocks_all = stocks[0][{"t": t_select}]
    values_all = stocks_all.values[-10:]
    stocks_select = stocks[1]
    values_select = stocks_select.values[-10:]
    assert np.max(np.abs(values_all - values_select)) < 0.01
