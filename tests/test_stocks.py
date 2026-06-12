import numpy as np
import pytest
import time

from flodym.dimensions import Dimension, DimensionSet
from flodym.flodym_arrays import StockArray
from flodym.mfa_definition import StockDefinition
from flodym.stock_helper import make_empty_stocks
from flodym.stocks import InflowDrivenDSM, StockDrivenDSM, SimpleFlowDrivenStock
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


def test_simple_stock_copy():
    """Copying a base (flow-driven) stock yields independent dims and arrays."""
    inflow = StockArray(dims=dims, values=np.ones((dims["t"].len, 2)))
    outflow = StockArray(dims=dims, values=np.zeros((dims["t"].len, 2)))
    stock = SimpleFlowDrivenStock(dims=dims, inflow=inflow, outflow=outflow, time_letter="t")
    stock.compute()

    stock_copy = stock.copy()

    # same type, equal values, but distinct objects
    assert isinstance(stock_copy, SimpleFlowDrivenStock)
    assert stock_copy is not stock
    assert stock_copy.dims is not stock.dims
    assert stock_copy.stock is not stock.stock
    assert stock_copy.stock.values is not stock.stock.values
    assert np.allclose(stock_copy.stock.values, stock.stock.values)

    # in-place mutation of the copy does not propagate back to the original
    stock_copy.stock.values[...] = 999.0
    assert not np.allclose(stock.stock.values, stock_copy.stock.values)


def test_dynamic_stock_copy():
    """Copying a dynamic stock yields independent arrays and an independent lifetime model."""
    inflow_values = np.exp(-np.linspace(-2, 2, 201) ** 2)
    inflow_values = np.stack([inflow_values, inflow_values]).T
    inflow = StockArray(dims=dims, values=inflow_values)
    lifetime_model = LogNormalLifetime(dims=dims, time_letter="t", mean=60, std=25)
    dsm = InflowDrivenDSM(dims=dims, inflow=inflow, lifetime_model=lifetime_model, time_letter="t")
    dsm.compute()

    dsm_copy = dsm.copy()

    # subclass preserved, values equal, objects distinct (incl. lifetime model)
    assert isinstance(dsm_copy, InflowDrivenDSM)
    assert dsm_copy is not dsm
    assert dsm_copy.inflow is not dsm.inflow
    assert dsm_copy.stock.values is not dsm.stock.values
    assert dsm_copy.lifetime_model is not dsm.lifetime_model
    assert np.allclose(dsm_copy.stock.values, dsm.stock.values)

    # in-place array mutation stays local to the copy
    original_inflow = dsm.inflow.values.copy()
    dsm_copy.inflow.values[...] = 0.0
    assert np.allclose(dsm.inflow.values, original_inflow)

    # changing the copy's lifetime model and recomputing leaves the original untouched
    stock_before = dsm.stock.values.copy()
    dsm_copy.inflow.values[...] = inflow_values  # restore so recompute is meaningful
    dsm_copy.lifetime_model.set_prms(mean=10, std=5)
    dsm_copy.compute()
    assert np.allclose(dsm.stock.values, stock_before)
    assert not np.allclose(dsm_copy.stock.values, stock_before)


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


def test_make_empty_stocks_raises_clear_error_for_missing_nondefault_time_letter():
    time_dim = Dimension(name="time", letter="s", items=[2000, 2005, 2010], dtype=int)
    product_dim = Dimension(name="product", letter="a", items=["automotive"], dtype=str)
    stock_dims = DimensionSet(dim_list=[time_dim, product_dim])
    stock_definition = StockDefinition(
        name="stock_with_nondefault_time",
        dim_letters=("s", "a"),
        subclass=InflowDrivenDSM,
        lifetime_model_class=LogNormalLifetime,
    )

    with pytest.raises(ValueError, match="time_letter"):
        make_empty_stocks([stock_definition], processes={}, dims=stock_dims)


def test_make_empty_stocks_accepts_explicit_nondefault_time_letter():
    time_dim = Dimension(name="time", letter="s", items=[2000, 2005, 2010], dtype=int)
    product_dim = Dimension(name="product", letter="a", items=["automotive"], dtype=str)
    stock_dims = DimensionSet(dim_list=[time_dim, product_dim])
    stock_definition = StockDefinition(
        name="stock_with_nondefault_time",
        dim_letters=("s", "a"),
        time_letter="s",
        subclass=InflowDrivenDSM,
        lifetime_model_class=LogNormalLifetime,
    )

    stocks = make_empty_stocks([stock_definition], processes={}, dims=stock_dims)

    assert stocks["stock_with_nondefault_time"].time_letter == "s"
