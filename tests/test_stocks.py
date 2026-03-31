import numpy as np
import pytest
import time

from flodym.dimensions import Dimension, DimensionSet
from flodym.flodym_arrays import StockArray
from flodym.stocks import InflowDrivenDSM, StockDrivenDSM, InflowByCohortDrivenDSM
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


def test_inflow_by_cohort_driven_dsm():
    """Test InflowByCohortDrivenDSM basic functionality."""
    # Create simple dimensions for testing
    time_items = list(range(2000, 2011))
    dims_test = DimensionSet(
        dim_list=[
            Dimension(name="time", letter="t", items=time_items, dtype=int),
            Dimension(name="product", letter="p", items=["A", "B"], dtype=str),
        ]
    )

    # Create cohort dimension
    cohort_dim = Dimension(name="cohort", letter="c", items=time_items, dtype=int)

    # Create lifetime model
    lifetime_model = LogNormalLifetime(dims=dims_test, time_letter="t", mean=5, std=2)

    # Create InflowByCohortDrivenDSM
    dsm = InflowByCohortDrivenDSM(
        dims=dims_test,
        cohort_dim=cohort_dim,
        lifetime_model=lifetime_model,
        time_letter="t",
        name="test_dsm",
    )

    # Set inflow by cohort - diagonal pattern (inflow only occurs in cohort year)
    dsm.inflow_by_cohort.values[...] = 0.0
    for i in range(len(time_items)):
        dsm.inflow_by_cohort.values[i, i, :] = 1.0  # 1 unit per product per year

    # Compute the DSM
    dsm.compute()

    # Check that inflow was computed correctly (diagonal extraction)
    assert dsm.inflow.is_set
    assert dsm.stock.is_set
    assert dsm.outflow.is_set

    # Verify inflow values are approximately 1.0 at each time step
    assert np.allclose(dsm.inflow.values, 1.0)

    # Verify stock balance
    dsm.check_stock_balance()


def test_inflow_driven_dsm_with_initial_stock():
    """Test InflowDrivenDSM with initial stock set via set_initial_stock."""
    # Create dimensions
    time_items = list(range(2000, 2021))
    dims_test = DimensionSet(
        dim_list=[
            Dimension(name="time", letter="t", items=time_items, dtype=int),
            Dimension(name="product", letter="p", items=["A", "B"], dtype=str),
        ]
    )

    # Create cohort dimension (required for initial stock functionality)
    cohort_dim = Dimension(name="cohort", letter="c", items=time_items, dtype=int)

    # Create lifetime model
    lifetime_model = LogNormalLifetime(dims=dims_test, time_letter="t", mean=8, std=3)

    # Create InflowDrivenDSM with cohort_dim
    dsm = InflowDrivenDSM(
        dims=dims_test,
        cohort_dim=cohort_dim,
        lifetime_model=lifetime_model,
        time_letter="t",
        name="test_dsm_initial",
    )

    # Set initial stock at year 2005
    initial_year = 2005
    # Initial stock dims should be (cohort, product) = _dims_cohort.drop(time_letter)
    initial_stock_dims = DimensionSet(
        dim_list=[cohort_dim] + list(dims_test.drop("t", inplace=False).dim_list)
    )
    initial_stock = StockArray(dims=initial_stock_dims, name="initial_stock")
    initial_stock.values[...] = 0.0
    # Set stock from various cohorts
    for i in range(5):  # 5 historical cohorts
        initial_stock.values[i, :] = 10.0 - i  # Decreasing stock from older cohorts

    dsm.set_initial_stock(initial_stock, initial_year)

    # Set inflow for years AFTER initial year only
    # Inflow before and at initial year will be computed from initial stock
    dsm.inflow.values[...] = 0.0
    initial_year_idx = time_items.index(initial_year)
    dsm.inflow.values[initial_year_idx + 1 :, :] = 2.0  # 2 units per year after initial
    dsm.inflow.mark_set()  # Mark inflow as set

    # Compute
    dsm.compute()

    # Verify that stock is set
    assert dsm.stock.is_set
    assert dsm.inflow.is_set
    assert dsm.outflow.is_set

    # Stock at initial year should be non-zero
    assert dsm.stock.values[initial_year_idx, :].sum() > 0

    # Stock before initial year should also be non-zero (reconstructed from initial stock)
    assert dsm.stock.values[initial_year_idx - 1, :].sum() > 0

    # Verify stock balance
    dsm.check_stock_balance()


def test_stock_driven_dsm_with_initial_stock():
    """Test StockDrivenDSM with initial stock set via set_initial_stock."""
    # Create dimensions
    time_items = list(range(2000, 2021))
    dims_test = DimensionSet(
        dim_list=[
            Dimension(name="time", letter="t", items=time_items, dtype=int),
            Dimension(name="product", letter="p", items=["A", "B"], dtype=str),
        ]
    )

    # Create cohort dimension (required for initial stock functionality)
    cohort_dim = Dimension(name="cohort", letter="c", items=time_items, dtype=int)

    # Create lifetime model
    lifetime_model = LogNormalLifetime(dims=dims_test, time_letter="t", mean=8, std=3)

    # Create StockDrivenDSM with cohort_dim
    dsm = StockDrivenDSM(
        dims=dims_test,
        cohort_dim=cohort_dim,
        lifetime_model=lifetime_model,
        time_letter="t",
        name="test_stock_driven_initial",
    )

    # Set initial stock at year 2005
    initial_year = 2005
    # Initial stock dims should be (cohort, product) = _dims_cohort.drop(time_letter)
    initial_stock_dims = DimensionSet(
        dim_list=[cohort_dim] + list(dims_test.drop("t", inplace=False).dim_list)
    )
    initial_stock = StockArray(dims=initial_stock_dims, name="initial_stock")
    initial_stock.values[...] = 0.0
    # Set stock from various cohorts
    for i in range(5):  # 5 historical cohorts
        initial_stock.values[i, :] = 10.0 - i  # Decreasing stock from older cohorts

    dsm.set_initial_stock(initial_stock, initial_year)

    # Set total stock for all years AFTER initial year only
    dsm.stock.values[...] = 0.0
    initial_year_idx = time_items.index(initial_year)
    # Stock before and at initial year will be computed from initial stock
    # We only set stock after the initial year
    for i in range(initial_year_idx + 1, len(time_items)):
        dsm.stock.values[i, :] = (
            50.0 + (i - initial_year_idx) * 2
        )  # Growing stock after initial year
    dsm.stock.mark_set()  # Mark stock as set

    # Compute
    dsm.compute()

    # Verify that inflow is set
    assert dsm.inflow.is_set
    assert dsm.stock.is_set
    assert dsm.outflow.is_set

    # Inflow should be non-zero
    assert dsm.inflow.values.sum() > 0

    # Verify stock balance
    dsm.check_stock_balance()


def test_get_stock_by_cohort():
    """Test get_stock_by_cohort method."""
    # Create dimensions
    time_items = list(range(2000, 2011))
    dims_test = DimensionSet(
        dim_list=[
            Dimension(name="time", letter="t", items=time_items, dtype=int),
            Dimension(name="product", letter="p", items=["A", "B"], dtype=str),
        ]
    )

    # Create cohort dimension
    cohort_dim = Dimension(name="cohort", letter="c", items=time_items, dtype=int)

    # Create lifetime model
    lifetime_model = LogNormalLifetime(dims=dims_test, time_letter="t", mean=5, std=2)

    # Create InflowDrivenDSM with cohort dimension
    dsm = InflowDrivenDSM(
        dims=dims_test,
        cohort_dim=cohort_dim,
        lifetime_model=lifetime_model,
        time_letter="t",
        name="test_cohort",
    )

    # Set inflow
    dsm.inflow.values[...] = 1.0

    # Compute
    dsm.compute()

    # Get stock by cohort
    stock_by_cohort = dsm.get_stock_by_cohort()

    # Verify it's a StockArray
    assert isinstance(stock_by_cohort, StockArray)

    # Verify dimensions include both time and cohort
    assert "t" in stock_by_cohort.dims.letters
    assert "c" in stock_by_cohort.dims.letters

    # Verify shape is correct (n_t, n_t, ...)
    assert stock_by_cohort.values.shape[0] == len(time_items)
    assert stock_by_cohort.values.shape[1] == len(time_items)

    # Verify that stock by cohort sums to total stock along cohort dimension
    total_stock_from_cohort = stock_by_cohort.values.sum(axis=1)
    assert np.allclose(total_stock_from_cohort, dsm.stock.values)

    # Verify that future cohorts (t < c) have zero stock
    for t_idx in range(len(time_items)):
        for c_idx in range(t_idx + 1, len(time_items)):
            assert np.allclose(stock_by_cohort.values[t_idx, c_idx, :], 0.0)


def test_get_outflow_by_cohort():
    """Test get_outflow_by_cohort method."""
    # Create dimensions
    time_items = list(range(2000, 2011))
    dims_test = DimensionSet(
        dim_list=[
            Dimension(name="time", letter="t", items=time_items, dtype=int),
            Dimension(name="product", letter="p", items=["A", "B"], dtype=str),
        ]
    )

    # Create cohort dimension
    cohort_dim = Dimension(name="cohort", letter="c", items=time_items, dtype=int)

    # Create lifetime model
    lifetime_model = LogNormalLifetime(dims=dims_test, time_letter="t", mean=5, std=2)

    # Create InflowDrivenDSM with cohort dimension
    dsm = InflowDrivenDSM(
        dims=dims_test,
        cohort_dim=cohort_dim,
        lifetime_model=lifetime_model,
        time_letter="t",
        name="test_cohort",
    )

    # Set inflow
    dsm.inflow.values[...] = 1.0

    # Compute
    dsm.compute()

    # Get outflow by cohort
    outflow_by_cohort = dsm.get_outflow_by_cohort()

    # Verify it's a StockArray
    assert isinstance(outflow_by_cohort, StockArray)

    # Verify dimensions include both time and cohort
    assert "t" in outflow_by_cohort.dims.letters
    assert "c" in outflow_by_cohort.dims.letters

    # Verify shape is correct (n_t, n_t, ...)
    assert outflow_by_cohort.values.shape[0] == len(time_items)
    assert outflow_by_cohort.values.shape[1] == len(time_items)

    # Verify that outflow by cohort sums to total outflow along cohort dimension
    total_outflow_from_cohort = outflow_by_cohort.values.sum(axis=1)
    assert np.allclose(total_outflow_from_cohort, dsm.outflow.values)

    # Verify that future cohorts (t < c) have zero outflow
    for t_idx in range(len(time_items)):
        for c_idx in range(t_idx + 1, len(time_items)):
            assert np.allclose(outflow_by_cohort.values[t_idx, c_idx, :], 0.0)
