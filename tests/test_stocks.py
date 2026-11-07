import logging

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from flodym import (
    Dimension,
    DimensionSet,
    DynamicStockModel,
    ExponentialLifetime,
    FixedLifetime,
    FlodymArray,
    FoldedNormalLifetime,
    InflowDrivenDSM,
    Leaching,
    LogNormalLifetime,
    NormalLifetime,
    SimpleFlowDrivenStock,
    StockArray,
    StockDefinition,
    StockDrivenDSM,
    WeibullLifetime,
    WeibullMeanStdLifetime,
    make_empty_stocks,
    make_processes,
)

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
    inflow_values = np.exp(-(np.linspace(-2, 2, 201) ** 2))
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
    )
    stock_driven_dsm.compute()
    inflow_post = stock_driven_dsm.inflow
    assert np.allclose(inflow.values, inflow_post.values)


def test_simple_stock_copy():
    """Copying a base (flow-driven) stock yields independent dims and arrays."""
    inflow = StockArray(dims=dims, values=np.ones((dims["t"].len, 2)))
    outflow = StockArray(dims=dims, values=np.zeros((dims["t"].len, 2)))
    stock = SimpleFlowDrivenStock(dims=dims, inflow=inflow, outflow=outflow, time_letter="t")
    stock.compute()

    stock_copy = stock.copy()

    # same type, but distinct objects
    assert isinstance(stock_copy, SimpleFlowDrivenStock)
    assert stock_copy is not stock

    # scalar attributes are preserved on the copy
    assert stock_copy.name == stock.name
    assert stock_copy.time_letter == stock.time_letter

    # dims and all three arrays are distinct objects with equal values
    assert stock_copy.dims is not stock.dims
    assert stock_copy.dims.letters == stock.dims.letters
    assert stock_copy.dims.names == stock.dims.names
    for letter in stock.dims.letters:
        assert stock_copy.dims[letter].items == stock.dims[letter].items
    for attr in ("stock", "inflow", "outflow"):
        original_array = getattr(stock, attr)
        copied_array = getattr(stock_copy, attr)
        assert copied_array is not original_array
        assert copied_array.values is not original_array.values
        assert np.allclose(copied_array.values, original_array.values)

    # in-place mutation of the copy does not propagate back to the original
    for attr in ("stock", "inflow", "outflow"):
        getattr(stock_copy, attr).values[...] = 999.0
        assert not np.any(getattr(stock, attr).values == 999.0)
        getattr(stock, attr).values[...] = -1.0
        assert not np.any(getattr(stock_copy, attr).values == -1.0)


def test_dynamic_stock_copy():
    """Copying a dynamic stock yields independent arrays and an independent lifetime model."""
    inflow_values = np.exp(-(np.linspace(-2, 2, 201) ** 2))
    inflow_values = np.stack([inflow_values, inflow_values]).T
    inflow = StockArray(dims=dims, values=inflow_values)
    lifetime_model = LogNormalLifetime(dims=dims, time_letter="t", mean=60, std=25)
    dsm = InflowDrivenDSM(dims=dims, inflow=inflow, lifetime_model=lifetime_model, time_letter="t")
    dsm.compute()

    dsm_copy = dsm.copy()

    # same type, but distinct objects
    assert isinstance(dsm_copy, InflowDrivenDSM)
    assert dsm_copy is not dsm

    # scalar attributes are preserved on the copy
    assert dsm_copy.name == dsm.name
    assert dsm_copy.time_letter == dsm.time_letter

    # dims and all three arrays are distinct objects with equal values
    assert dsm_copy.dims is not dsm.dims
    assert dsm_copy.dims.letters == dsm.dims.letters
    assert dsm_copy.dims.names == dsm.dims.names
    for letter in dsm.dims.letters:
        assert dsm_copy.dims[letter].items == dsm.dims[letter].items
    for attr in ("stock", "inflow", "outflow"):
        original_array = getattr(dsm, attr)
        copied_array = getattr(dsm_copy, attr)
        assert copied_array is not original_array
        assert copied_array.values is not original_array.values
        assert np.allclose(copied_array.values, original_array.values)

    # the lifetime model is a distinct object whose parameters start out equal
    assert dsm_copy.lifetime_model is not dsm.lifetime_model
    for prm_name, prm in dsm.lifetime_model.prms.items():
        assert np.allclose(dsm_copy.lifetime_model.prms[prm_name], prm)

    # changing the copy's lifetime model and recomputing leaves the original untouched
    stock_before = dsm.stock.values.copy()
    dsm_copy.lifetime_model.set_prms(mean=10, std=5)
    dsm_copy.compute()
    assert np.allclose(dsm.stock.values, stock_before)
    assert not np.allclose(dsm_copy.stock.values, stock_before)

    # in-place mutation of the copy does not propagate back to the original
    for attr in ("stock", "inflow", "outflow"):
        getattr(dsm_copy, attr).values[...] = 999.0
        assert not np.any(getattr(dsm, attr).values == 999.0)
        getattr(dsm, attr).values[...] = -1.0
        assert not np.any(getattr(dsm_copy, attr).values == -1.0)


def test_lifetime_quadrature():
    # Put in constant inflow and check stationary stock values

    # Long lifetimes:
    # Inflow at start/end of time step under/overestimate stock by half a year,
    # others should work well
    _inflow, stocks = get_stocks_by_quadrature(mean=30, std=10)
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
    _inflow, stocks = get_stocks_by_quadrature(mean=0.3, std=0.1)
    for name, stock in stocks.items():
        if name == "ltm_6":
            assert np.abs(stock["automotive"].values[-1] - 0.3) < eps
        else:
            assert np.abs(stock["automotive"].values[-1] - 0.3) > eps


def get_stocks_by_quadrature(mean, std):
    inflow_values = np.exp(-(np.linspace(-2, 2, 201) ** 2))
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
    """Test basic functionality of the inflow by cohort driven computation."""
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

    dsm = DynamicStockModel(
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
    dsm.compute_inflow_by_cohort_driven()

    # Check that inflow was computed correctly (diagonal extraction)
    assert dsm.inflow.is_set
    assert dsm.stock.is_set
    assert dsm.outflow.is_set

    # Verify inflow values are approximately 1.0 at each time step
    assert np.allclose(dsm.inflow.values, 1.0)

    # Verify stock balance
    dsm.check_mass_balance()


def test_inflow_driven_dsm_with_initial_stock():
    """Test InflowDrivenDSM with initial stock set via set_initial_stock."""
    # Create dimensions
    time_items = list(range(2000, 2021))
    dims_test = DimensionSet(
        dim_list=[
            Dimension(name="time", letter="t", items=time_items, dtype=int),
            # Dimension(name="product", letter="p", items=["A", "B"], dtype=str),
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
    # Initial stock dims should be (cohort, product) = dims_cohort.drop(time_letter)
    initial_stock_dims = DimensionSet(
        dim_list=[cohort_dim] + list(dims_test.drop("t", inplace=False).dim_list)
    )
    initial_stock = StockArray(dims=initial_stock_dims, name="initial_stock")
    initial_stock.values[...] = 0.0
    # Set stock from various cohorts
    for i in range(5):  # 5 historical cohorts
        initial_stock.values[i, ...] = 10.0 - i  # Decreasing stock from older cohorts
        # sum is 10 + 9 + 8 + 7 + 6 = 40 units at initial year

    dsm.set_initial_stock(initial_stock, initial_year)

    # Set inflow for years AFTER initial year only
    # Inflow before and at initial year will be computed from initial stock
    dsm.inflow.values[...] = 0.0
    initial_year_idx = time_items.index(initial_year)
    dsm.inflow.values[initial_year_idx + 1 :, ...] = 2.0  # 2 units per year after initial
    dsm.inflow.mark_set()  # Mark inflow as set

    # Compute
    dsm.compute()

    # Verify that stock is set
    assert dsm.stock.is_set
    assert dsm.inflow.is_set
    assert dsm.outflow.is_set

    # Stock at initial year should be non-zero
    assert dsm.stock.values[initial_year_idx, ...].sum() > 0

    # Verify stock balance
    dsm.check_mass_balance()


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
    # Initial stock dims should be (cohort, product) = dims_cohort.drop(time_letter)
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
    dsm.check_mass_balance()


time_items_cohort_tests = list(range(2000, 2021))
dims_cohort_tests = DimensionSet(
    dim_list=[
        Dimension(name="time", letter="t", items=time_items_cohort_tests, dtype=int),
        Dimension(name="product", letter="p", items=["A", "B"], dtype=str),
    ]
)
cohort_dim_tests = Dimension(name="cohort", letter="c", items=time_items_cohort_tests, dtype=int)
initial_year_tests = 2005
initial_year_index_tests = time_items_cohort_tests.index(initial_year_tests)


def _dsm_for_cohort_tests(dsm_cls, **kwargs):
    lifetime_model = LogNormalLifetime(dims=dims_cohort_tests, time_letter="t", mean=8, std=3)
    return dsm_cls(
        dims=dims_cohort_tests,
        cohort_dim=cohort_dim_tests,
        lifetime_model=lifetime_model,
        time_letter="t",
        **kwargs,
    )


def _initial_stock_for_cohort_tests():
    initial_stock_dims = DimensionSet(
        dim_list=[cohort_dim_tests] + list(dims_cohort_tests.drop("t", inplace=False).dim_list)
    )
    initial_stock = StockArray(dims=initial_stock_dims, name="initial_stock")
    for i in range(initial_year_index_tests + 1):
        initial_stock.values[i, :] = [10.0 - i, 5.0 + i]
    return initial_stock


def test_inflow_driven_dsm_matches_survival_function():
    """Time stepping gives the same cohorts as the direct product with the cumulative survival
    and the outflow pmf."""
    rng = np.random.default_rng(0)
    inflow = StockArray(dims=dims_cohort_tests, values=rng.random(dims_cohort_tests.shape))
    dsm = _dsm_for_cohort_tests(InflowDrivenDSM, inflow=inflow)
    dsm.compute()

    cum_surv = dsm.lifetime_model.cumulative_survival
    outflow_pmf = dsm.lifetime_model.outflow_pmf
    np.testing.assert_allclose(
        dsm.get_stock_by_cohort().values, np.einsum("c...,tc...->tc...", inflow.values, cum_surv)
    )
    np.testing.assert_allclose(
        dsm.get_outflow_by_cohort().values,
        np.einsum("c...,tc...->tc...", inflow.values, outflow_pmf),
        atol=1e-14,
    )
    np.testing.assert_allclose(dsm.inflow.values, inflow.values)


def test_inflow_driven_dsm_initial_stock_survival():
    """The initial stock by cohort is the stock at the end of the initial year, and decays
    according to the survival function afterwards."""
    dsm = _dsm_for_cohort_tests(InflowDrivenDSM)
    initial_stock = _initial_stock_for_cohort_tests()
    dsm.set_initial_stock(initial_stock, initial_year_tests)
    dsm.inflow.values[initial_year_index_tests + 1 :, ...] = 2.0
    dsm.inflow.mark_set()
    dsm.compute()

    iiy = initial_year_index_tests
    cum_surv = dsm.lifetime_model.cumulative_survival
    stock_by_cohort = dsm.get_stock_by_cohort().values
    expected = np.einsum(
        "c...,tc...->tc...",
        initial_stock.values[: iiy + 1],
        cum_surv[iiy:, : iiy + 1] / cum_surv[iiy, : iiy + 1],
    )
    np.testing.assert_allclose(stock_by_cohort[iiy:, : iiy + 1], expected)
    assert np.all(stock_by_cohort[:iiy] == 0.0)
    dsm.check_mass_balance()


def test_stock_driven_dsm_initial_stock_reproduces_stock():
    """With an initial stock, the prescribed stock after the initial year is the total stock."""
    dsm = _dsm_for_cohort_tests(StockDrivenDSM)
    initial_stock = _initial_stock_for_cohort_tests()
    dsm.set_initial_stock(initial_stock, initial_year_tests)
    iiy = initial_year_index_tests
    n_after = len(time_items_cohort_tests) - iiy - 1
    dsm.stock.values[iiy + 1 :, :] = 80.0 + 2.0 * np.arange(n_after)[:, np.newaxis]
    prescribed_stock = dsm.stock.values.copy()
    dsm.stock.mark_set()
    dsm.compute()

    stock_by_cohort = dsm.get_stock_by_cohort().values
    np.testing.assert_allclose(stock_by_cohort[iiy], initial_stock.values)
    np.testing.assert_allclose(dsm.stock.values[iiy + 1 :], prescribed_stock[iiy + 1 :])
    dsm.check_mass_balance()


dims_with_cohort_tests = DimensionSet(
    dim_list=[dims_cohort_tests["t"], cohort_dim_tests, dims_cohort_tests["p"]]
)


def _driving_array_for_cohort_tests(driven_by: str) -> StockArray:
    rng = np.random.default_rng(1)
    if driven_by == "inflow_by_cohort":
        # inflow only into cohorts which already exist
        values = rng.random(dims_with_cohort_tests.shape)
        values *= np.tri(len(time_items_cohort_tests))[:, :, np.newaxis]
        return StockArray(dims=dims_with_cohort_tests, values=values)
    values = rng.random(dims_cohort_tests.shape)
    if driven_by == "stock":
        values = np.cumsum(values, axis=0)
    return StockArray(dims=dims_cohort_tests, values=values)


_explicit_compute_methods = {
    "inflow": "compute_inflow_driven",
    "stock": "compute_stock_driven",
    "inflow_by_cohort": "compute_inflow_by_cohort_driven",
}


@pytest.mark.parametrize("driven_by", ["inflow", "stock", "inflow_by_cohort"])
def test_compute_chooses_mode_like_explicit_method(driven_by):
    results = []
    for method in _explicit_compute_methods[driven_by], "compute":
        dsm = _dsm_for_cohort_tests(
            DynamicStockModel, **{driven_by: _driving_array_for_cohort_tests(driven_by)}
        )
        getattr(dsm, method)()
        results.append(dsm)
    for attr in ("stock", "inflow", "outflow"):
        np.testing.assert_allclose(
            getattr(results[0], attr).values, getattr(results[1], attr).values
        )


@pytest.mark.parametrize("driven_by", [["stock", "inflow"], ["inflow", "inflow_by_cohort"]])
def test_compute_raises_if_several_arrays_set(driven_by):
    dsm = _dsm_for_cohort_tests(
        DynamicStockModel, **{name: _driving_array_for_cohort_tests(name) for name in driven_by}
    )
    with pytest.raises(ValueError, match="Exactly one of stock, inflow and inflow_by_cohort"):
        dsm.compute()


def test_compute_falls_back_to_non_zero_array(caplog):
    dsm = _dsm_for_cohort_tests(DynamicStockModel)
    dsm.inflow.values[...] = _driving_array_for_cohort_tests("inflow").values
    with caplog.at_level(logging.WARNING):
        dsm.compute()
    assert "Computing from inflow" in caplog.text
    reference = _dsm_for_cohort_tests(
        DynamicStockModel, inflow=_driving_array_for_cohort_tests("inflow")
    )
    reference.compute_inflow_driven()
    np.testing.assert_allclose(dsm.stock.values, reference.stock.values)


def test_compute_raises_if_nothing_given():
    dsm = _dsm_for_cohort_tests(DynamicStockModel)
    with pytest.raises(ValueError, match="cannot be chosen"):
        dsm.compute()


def test_recompute():
    dsm = _dsm_for_cohort_tests(DynamicStockModel, inflow=_driving_array_for_cohort_tests("inflow"))
    dsm.compute()
    with pytest.raises(ValueError, match="compute_inflow_driven"):
        dsm.compute()

    # explicit methods can be called repeatedly
    stock = dsm.stock.values.copy()
    dsm.compute_stock_driven()
    np.testing.assert_allclose(dsm.inflow.values, _driving_array_for_cohort_tests("inflow").values)
    np.testing.assert_allclose(dsm.stock.values, stock)

    # or the choice of compute() can be reset
    dsm.mark_all_unset()
    dsm.stock[...] = stock
    dsm.compute()
    np.testing.assert_allclose(dsm.inflow.values, _driving_array_for_cohort_tests("inflow").values)


def test_stock_driven_then_corrected_inflow_driven():
    """Compute stock-driven, set negative inflows to zero, and recompute inflow-driven."""
    stock_values = _driving_array_for_cohort_tests("stock").values
    stock_values[10:] -= 3.0 * np.arange(len(time_items_cohort_tests) - 10)[:, np.newaxis]
    dsm = _dsm_for_cohort_tests(
        DynamicStockModel, stock=StockArray(dims=dims_cohort_tests, values=stock_values)
    )
    dsm.compute_stock_driven()
    assert np.any(dsm.inflow.values < 0)
    corrected_inflow = np.maximum(dsm.inflow.values, 0.0)
    dsm.inflow[...] = corrected_inflow
    dsm.compute_inflow_driven()

    reference = _dsm_for_cohort_tests(
        InflowDrivenDSM, inflow=StockArray(dims=dims_cohort_tests, values=corrected_inflow)
    )
    reference.compute()
    np.testing.assert_allclose(dsm.inflow.values, corrected_inflow)
    for attr in ("stock", "outflow"):
        np.testing.assert_allclose(getattr(dsm, attr).values, getattr(reference, attr).values)


@pytest.mark.parametrize("included_in_inflow", [True, False])
def test_stock_driven_then_inflow_driven_with_initial_stock(included_in_inflow):
    """The inflow of a stock-driven computation with initial stock contains the inflow of the
    initial stock. With included_in_inflow=True, recomputing inflow-driven reproduces the stock."""
    dsm = _dsm_for_cohort_tests(DynamicStockModel)
    dsm.set_initial_stock(
        _initial_stock_for_cohort_tests(), initial_year_tests, included_in_inflow=included_in_inflow
    )
    iiy = initial_year_index_tests
    n_after = len(time_items_cohort_tests) - iiy - 1
    dsm.stock.values[iiy + 1 :, :] = 80.0 + 2.0 * np.arange(n_after)[:, np.newaxis]
    dsm.compute_stock_driven()
    stock_driven_stock = dsm.stock.values.copy()
    stock_driven_inflow = dsm.inflow.values.copy()

    dsm.compute_inflow_driven()
    if included_in_inflow:
        np.testing.assert_allclose(dsm.stock.values, stock_driven_stock)
        np.testing.assert_allclose(dsm.inflow.values, stock_driven_inflow)
    else:
        # the initial stock of the older cohorts is counted twice
        assert np.all(dsm.stock.values[iiy:] > stock_driven_stock[iiy:])
    dsm.check_mass_balance()


def test_user_subclass_of_dynamic_stock_model_marks_set():
    class MyDSM(DynamicStockModel):
        def compute(self):
            self.compute_inflow_driven()

    dsm = _dsm_for_cohort_tests(MyDSM)
    dsm.inflow.values[...] = 1.0
    dsm.compute()
    assert dsm.is_computed


@pytest.mark.parametrize(
    "time_items",
    [time_items_cohort_tests, [2000, 2001, 2003, 2004, 2007, 2010, 2011, 2015, 2020]],
    ids=["equal_time_steps", "unequal_time_steps"],
)
def test_inflow_by_cohort_driven_dsm_matches_inflow_driven_dsm(time_items):
    """Compares the vectorized inflow-driven computation with the time stepping."""
    time_dim = Dimension(name="time", letter="t", items=time_items, dtype=int)
    dims_test = DimensionSet(dim_list=[time_dim, dims_cohort_tests["p"]])
    cohort_dim = Dimension(name="cohort", letter="c", items=time_items, dtype=int)
    rng = np.random.default_rng(2)
    inflow_values = rng.random(dims_test.shape)

    results = []
    for driven_by in "inflow", "inflow_by_cohort":
        dsm = DynamicStockModel(
            dims=dims_test,
            cohort_dim=cohort_dim,
            lifetime_model=LogNormalLifetime(dims=dims_test, mean=8, std=3),
        )
        if driven_by == "inflow":
            dsm.inflow[...] = inflow_values
        else:
            t_diag_indices = np.diag_indices(len(time_items)) + (slice(None),)
            dsm.inflow_by_cohort.values[t_diag_indices] = inflow_values
            dsm.inflow_by_cohort.mark_set()
        dsm.compute()
        results.append(dsm)

    for attr in ("stock", "inflow", "outflow"):
        np.testing.assert_allclose(
            getattr(results[0], attr).values, getattr(results[1], attr).values, atol=1e-14
        )
    for method in "get_stock_by_cohort", "get_outflow_by_cohort":
        np.testing.assert_allclose(
            getattr(results[0], method)().values, getattr(results[1], method)().values, atol=1e-14
        )


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


def test_lifetime_ext(plot=False):
    EXT_FAC = 2

    # Nurture

    dsm = _get_dsm_with_lifetime_ext(EXT_FAC, nurture=True)

    assert_almost_equal(
        dsm.stock[{"t": 4, "p": "Base"}].values, dsm.stock[{"t": 4, "p": "Sudden ext"}].values
    )

    assert_almost_equal(
        dsm.stock[{"t": 4, "p": "Base"}].values, dsm.stock[{"t": 4, "p": "Smooth ext"}].values
    )

    base = dsm.stock[{"t": 15, "p": "Base"}].values
    all_ext = dsm.stock[{"t": 15, "p": "All ext"}].values
    sudden_ext = dsm.stock[{"t": 15, "p": "Sudden ext"}].values
    smooth_ext = dsm.stock[{"t": 15, "p": "Smooth ext"}].values

    assert all_ext > smooth_ext > sudden_ext > base

    # Nature

    dsm = _get_dsm_with_lifetime_ext(EXT_FAC, nurture=False)

    assert_almost_equal(
        dsm.stock[{"t": 4, "p": "Base"}].values, dsm.stock[{"t": 4, "p": "Sudden ext"}].values
    )

    assert_almost_equal(
        dsm.stock[{"t": 4, "p": "Base"}].values, dsm.stock[{"t": 4, "p": "Smooth ext"}].values
    )

    base = dsm.stock[{"t": 20, "p": "Base"}].values
    all_ext = dsm.stock[{"t": 20, "p": "All ext"}].values
    sudden_ext = dsm.stock[{"t": 20, "p": "Sudden ext"}].values
    smooth_ext = dsm.stock[{"t": 20, "p": "Smooth ext"}].values

    assert all_ext > sudden_ext > base
    assert all_ext > smooth_ext > base

    assert (
        dsm.stock[{"t": 9, "p": "Sudden ext"}].values
        < dsm.stock[{"t": 9, "p": "Smooth ext"}].values
    )


def _get_dsm_with_lifetime_ext(EXT_FAC, nurture=True):
    dim_list = [
        Dimension(
            name="time",
            letter="t",
            items=list(range(31)),
            dtype=int,
        ),
        Dimension(
            name="product",
            letter="p",
            items=["Base", "All ext", "Sudden ext", "Smooth ext"],
            dtype=str,
        ),
    ]

    dims = DimensionSet(dim_list=dim_list)

    inflow = StockArray(dims=dims)

    if nurture:
        # effect of extension through repair etc is clearest for only one cohort
        inflow[{"t": 0}] = 1
    else:
        # extension through product design needs several inflow years
        inflow[...] = 1

    factor = FlodymArray(dims=dims)
    factor[...] = 1.0
    factor["All ext"] = EXT_FAC
    factor[{"p": "Sudden ext", "t": range(10, 31)}] = EXT_FAC
    # blend from 5 to 14 years for product D
    x_clip = np.clip((np.arange(31) - 5) / 10, 0, 1)
    factor["Smooth ext"] = 1 + (EXT_FAC - 1) * x_clip

    ext_prm_name = "lt_factor_by_year" if nurture else "lt_factor_by_cohort"
    lifetime_model = WeibullLifetime(
        dims=dims, time_letter="t", weibull_scale=10, weibull_shape=2, **{ext_prm_name: factor}
    )

    dsm = InflowDrivenDSM(
        dims=dims,
        inflow=inflow,
        lifetime_model=lifetime_model,
        time_letter="t",
    )
    dsm.compute()
    return dsm


def _nurture_lifetime_model(lt_cls, **kwargs):
    dims = DimensionSet(
        dim_list=[
            Dimension(name="time", letter="t", items=[0, 1, 2, 4, 5, 8, 9, 10, 13, 20], dtype=int),
            Dimension(name="product", letter="p", items=["a", "b", "c"], dtype=str),
        ]
    )
    rng = np.random.default_rng(0)
    shape = (dims["t"].len, dims["p"].len)
    factor = FlodymArray(dims=dims, values=rng.uniform(1.0, 2.0, shape))
    if lt_cls is WeibullLifetime:
        prms = {
            "weibull_scale": rng.uniform(3, 8, shape),
            "weibull_shape": rng.uniform(1, 3, shape),
        }
    elif lt_cls in (FixedLifetime, ExponentialLifetime):
        prms = {"mean": rng.uniform(2, 6, shape)}
    elif lt_cls is Leaching:
        prms = {"loss_rate": rng.uniform(0.05, 0.3, shape)}
    elif kwargs.get("relative_std", False):
        prms = {"mean": rng.uniform(3, 8, shape), "std": rng.uniform(0.2, 0.5, shape)}
    else:
        prms = {"mean": rng.uniform(3, 8, shape), "std": rng.uniform(1, 2, shape)}
    return lt_cls(dims=dims, time_letter="t", lt_factor_by_year=factor, **prms, **kwargs)


def _nurture_cum_surv_per_cohort(lt):
    """Reference: the nurture cumulative survival computed one cohort at a time."""
    n_t = lt._n_t
    names_to_scale = lt._get_prm_names_to_scale()
    prms_orig = {name: getattr(lt, name).copy() for name in names_to_scale}
    points, weights = lt._get_quad_points_and_weights()
    cum_surv = np.zeros(lt._shape_cohort)
    for i_t in range(n_t):
        for name in names_to_scale:
            lt._scaled_prms[name] = lt._scale_prm(prms_orig[name], lt.lt_factor_by_year[i_t])
        curr = np.zeros((min(2, i_t + 1), n_t) + lt._shape_no_t)
        for point, weight in zip(points, weights):
            for i_c in range(i_t + 1):
                ages = lt._tile(
                    lt._t.bounds[max(i_t, 1) : i_t + 2] - lt._quad_point_time(i_c, point)
                )
                curr[:, i_c, ...] += weight * lt._survival_by_cohort(ages, i_c)
        cum_surv[i_t, i_t] = curr[-1, i_t]
        if i_t > 0:
            prev, now = curr[0, :i_t], curr[1, :i_t]
            ratio = np.divide(now, prev, out=np.zeros_like(now), where=prev > 0)
            cum_surv[i_t, :i_t] = cum_surv[i_t - 1, :i_t] * ratio
    return cum_surv


@pytest.mark.parametrize(
    "lt_cls, kwargs",
    [
        (NormalLifetime, {}),
        (FoldedNormalLifetime, {}),
        (LogNormalLifetime, {}),
        (LogNormalLifetime, {"relative_std": True}),
        (WeibullLifetime, {}),
        (WeibullMeanStdLifetime, {}),
        (WeibullMeanStdLifetime, {"relative_std": True}),
        (FixedLifetime, {}),
        (ExponentialLifetime, {}),
        (Leaching, {}),
    ],
)
@pytest.mark.parametrize("n_pts", [1, 3])
def test_nurture_matches_per_cohort_loop(lt_cls, kwargs, n_pts):
    lt = _nurture_lifetime_model(lt_cls, n_pts_per_interval=n_pts, **kwargs)
    cum_surv = lt.cumulative_survival.copy()
    np.testing.assert_allclose(cum_surv, _nurture_cum_surv_per_cohort(lt), rtol=1e-12, atol=1e-15)


def test_nurture_keeps_lifetime_parameters():
    lt = _nurture_lifetime_model(WeibullLifetime)
    scale = lt.weibull_scale.copy()
    first = lt.cumulative_survival.copy()
    np.testing.assert_array_equal(lt.weibull_scale, scale)

    # recomputing, e.g. for a second scenario, starts from the same parameters
    lt.reset_cached_arrays()
    np.testing.assert_array_equal(lt.cumulative_survival, first)


def test_process_id_and_to_stock_type():
    time_dims = DimensionSet(dim_list=[dims["t"]])
    processes = make_processes(["sysenv", "a"])
    stock = SimpleFlowDrivenStock(
        dims=time_dims,
        inflow=StockArray(dims=time_dims, values=np.ones(time_dims.shape)),
        outflow=StockArray(dims=time_dims, values=np.zeros(time_dims.shape)),
        process=processes["a"],
        name="s",
    )
    assert stock.process_id == processes["a"].id

    dsm = stock.to_stock_type(InflowDrivenDSM, lifetime_model=FixedLifetime(dims=time_dims, mean=3))
    assert isinstance(dsm, InflowDrivenDSM)
    assert dsm.name == "s"
    assert dsm.process is processes["a"]
    assert np.array_equal(dsm.inflow.values, stock.inflow.values)
    dsm.compute()
    assert_almost_equal(dsm.stock.values[:5], [1.0, 2.0, 3.0, 3.0, 3.0])
