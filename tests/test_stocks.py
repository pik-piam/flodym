import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from flodym import (
    Dimension,
    DimensionSet,
    FlodymArray,
    InflowDrivenDSM,
    StockDrivenDSM,
    SimpleFlowDrivenStock,
    StockArray,
    StockDefinition,
    WeibullLifetime,
    NormalLifetime,
    FoldedNormalLifetime,
    FixedLifetime,
    LogNormalLifetime,
    make_empty_stocks,
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
    elif lt_cls is FixedLifetime:
        prms = {"mean": rng.uniform(2, 6, shape)}
    else:
        prms = {"mean": rng.uniform(3, 8, shape), "std": rng.uniform(1, 2, shape)}
    return lt_cls(dims=dims, time_letter="t", lt_factor_by_year=factor, **prms, **kwargs)


def _nurture_sf_per_cohort(lt):
    """Reference: the nurture survival factor computed one cohort at a time."""
    n_t = lt._n_t
    prms_orig = {name: getattr(lt, name).copy() for name in lt._prm_names_to_scale}
    points, weights = lt._get_quad_points_and_weights()
    sf = np.zeros(lt._shape_cohort)
    for i_t in range(n_t):
        for name in lt._prm_names_to_scale:
            lt._scaled_prms[name] = prms_orig[name] * lt.lt_factor_by_year[i_t]
        curr = np.zeros((min(2, i_t + 1), n_t) + lt._shape_no_t)
        for point, weight in zip(points, weights):
            for i_c in range(i_t + 1):
                ages = lt._tile(
                    lt._t.bounds[max(i_t, 1) : i_t + 2] - lt._quad_point_time(i_c, point)
                )
                curr[:, i_c, ...] += weight * lt._survival_by_cohort(ages, i_c)
        sf[i_t, i_t] = curr[-1, i_t]
        if i_t > 0:
            prev, now = curr[0, :i_t], curr[1, :i_t]
            ratio = np.divide(now, prev, out=np.zeros_like(now), where=prev > 0)
            sf[i_t, :i_t] = sf[i_t - 1, :i_t] * ratio
    return sf


@pytest.mark.parametrize(
    "lt_cls",
    [NormalLifetime, FoldedNormalLifetime, LogNormalLifetime, WeibullLifetime, FixedLifetime],
)
@pytest.mark.parametrize("n_pts", [1, 3])
def test_nurture_matches_per_cohort_loop(lt_cls, n_pts):
    lt = _nurture_lifetime_model(lt_cls, n_pts_per_interval=n_pts)
    sf = lt.sf.copy()
    np.testing.assert_allclose(sf, _nurture_sf_per_cohort(lt), rtol=1e-12, atol=1e-15)


def test_nurture_keeps_lifetime_parameters():
    lt = _nurture_lifetime_model(WeibullLifetime)
    scale = lt.weibull_scale.copy()
    first = lt.sf.copy()
    np.testing.assert_array_equal(lt.weibull_scale, scale)

    # recomputing, e.g. for a second scenario, starts from the same parameters
    lt.reset_cached_arrays()
    np.testing.assert_array_equal(lt.sf, first)
