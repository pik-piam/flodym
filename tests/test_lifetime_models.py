import numpy as np
import pytest
import scipy.stats

from flodym import (
    CompetingRisksLifetime,
    Dimension,
    DimensionSet,
    ExponentialLifetime,
    FixedLifetime,
    FlodymArray,
    FoldedNormalLifetime,
    InflowDrivenDSM,
    Leaching,
    LogNormalLifetime,
    NormalLifetime,
    StockArray,
    TabulatedLifetime,
    WeibullLifetime,
    WeibullMeanStdLifetime,
)

DIMS = DimensionSet(
    dim_list=[
        Dimension(name="time", letter="t", items=list(range(2000, 2031)), dtype=int),
        Dimension(name="product", letter="p", items=["a", "b"], dtype=str),
    ]
)
MEAN = FlodymArray(dims=DIMS["p",], values=np.array([8.0, 15.0]))
FACTOR = FlodymArray(dims=DIMS, values=np.linspace(1.0, 2.0, DIMS["t"].len)[:, None] * [1, 1])


@pytest.mark.parametrize(
    "lt_cls", [NormalLifetime, FoldedNormalLifetime, LogNormalLifetime, WeibullMeanStdLifetime]
)
@pytest.mark.parametrize("factor_name", [None, "lt_factor_by_cohort", "lt_factor_by_year"])
def test_relative_std_equals_absolute_std(lt_cls, factor_name):
    factor_kwargs = {} if factor_name is None else {factor_name: FACTOR}
    rel = lt_cls(dims=DIMS, mean=MEAN, std=0.3, relative_std=True, **factor_kwargs)
    abs_ = lt_cls(dims=DIMS, mean=MEAN, std=MEAN * 0.3, **factor_kwargs)
    np.testing.assert_allclose(
        rel.cumulative_survival, abs_.cumulative_survival, rtol=1e-12, atol=1e-14
    )


@pytest.mark.parametrize("cv", [0.1, 0.5, 1.0, 2.0])
def test_weibull_mean_std_reproduces_moments(cv):
    lt = WeibullMeanStdLifetime(dims=DIMS, mean=MEAN, std=MEAN * cv)
    lt.cumulative_survival
    shape = lt._weibull_shape
    scale = lt.mean / scipy.special.gamma(1 + 1 / shape)
    dist = scipy.stats.weibull_min(c=shape, scale=scale)
    np.testing.assert_allclose(dist.mean(), lt.mean, rtol=1e-10)
    np.testing.assert_allclose(dist.std(), lt.std, rtol=1e-10)
    if cv == 1.0:
        # Weibull with shape 1 is the exponential distribution
        np.testing.assert_allclose(shape, 1.0, rtol=1e-10)


def test_weibull_mean_std_matches_weibull():
    shape, scale = 2.5, 10.0
    dist = scipy.stats.weibull_min(c=shape, scale=scale)
    lt_ms = WeibullMeanStdLifetime(dims=DIMS, mean=dist.mean(), std=dist.std())
    lt = WeibullLifetime(dims=DIMS, weibull_shape=shape, weibull_scale=scale)
    np.testing.assert_allclose(
        lt_ms.cumulative_survival, lt.cumulative_survival, rtol=1e-10, atol=1e-14
    )


def test_weibull_mean_std_raises_for_invalid_cv():
    lt = WeibullMeanStdLifetime(dims=DIMS, mean=MEAN, std=0.0)
    with pytest.raises(ValueError, match="Coefficient of variation"):
        lt.cumulative_survival


def test_exponential_and_leaching():
    rate = 0.1
    expo = ExponentialLifetime(dims=DIMS, mean=10.0, inflow_at="start")
    ages = np.arange(DIMS["t"].len) + 1.0
    np.testing.assert_allclose(expo.cumulative_survival[:, 0, 0], np.exp(-ages / 10.0))

    leach = Leaching(dims=DIMS, loss_rate=rate, inflow_at="start")
    np.testing.assert_allclose(leach.cumulative_survival[:, 0, 0], (1 - rate) ** ages)
    np.testing.assert_allclose(leach.conditional_survival[1:, 0, 0], 1 - rate)

    # Leaching is an exponential lifetime, also when lifetime factors are applied
    expo_equiv_mean = -1 / np.log(1 - rate)
    for factor_name in ["lt_factor_by_cohort", "lt_factor_by_year"]:
        leach = Leaching(dims=DIMS, loss_rate=rate, **{factor_name: FACTOR})
        expo = ExponentialLifetime(dims=DIMS, mean=expo_equiv_mean, **{factor_name: FACTOR})
        np.testing.assert_allclose(
            leach.cumulative_survival, expo.cumulative_survival, rtol=1e-12, atol=1e-15
        )


def test_competing_risks():
    leach = Leaching(dims=DIMS, loss_rate=0.05)
    product = LogNormalLifetime(dims=DIMS, mean=MEAN, std=0.3, relative_std=True)
    combined = CompetingRisksLifetime(models=[product, leach])
    assert combined.dims.letters == product.dims.letters
    np.testing.assert_allclose(
        combined.cumulative_survival, product.cumulative_survival * leach.cumulative_survival
    )
    lower_tri = np.tril(np.ones((DIMS["t"].len,) * 2, dtype=bool))
    np.testing.assert_allclose(
        combined.conditional_survival[lower_tri],
        (product.conditional_survival * leach.conditional_survival)[lower_tri],
    )
    outflow_pmf_before = combined.outflow_pmf.copy()

    # changing parameters of a model updates the combined model, for each accessed array
    leach.loss_rate = 0.2
    assert not np.allclose(combined.outflow_pmf, outflow_pmf_before)
    np.testing.assert_allclose(
        combined.cumulative_survival, product.cumulative_survival * leach.cumulative_survival
    )
    product.mean = 20.0
    np.testing.assert_allclose(
        combined.conditional_survival[lower_tri],
        (product.conditional_survival * leach.conditional_survival)[lower_tri],
    )

    # replacing the models
    combined.models = [product, leach, ExponentialLifetime(dims=DIMS, mean=30.0)]
    expected = (
        product.cumulative_survival
        * leach.cumulative_survival
        * combined.models[2].cumulative_survival
    )
    np.testing.assert_allclose(combined.cumulative_survival, expected)


def test_competing_risks_in_stock():
    inflow = StockArray(dims=DIMS, values=np.ones(DIMS.shape))
    stocks = {}
    for loss_rate in [0.0, 0.1]:
        lifetime_model = CompetingRisksLifetime(
            models=[
                WeibullMeanStdLifetime(dims=DIMS, mean=MEAN, std=0.4, relative_std=True),
                Leaching(dims=DIMS, loss_rate=loss_rate),
            ]
        )
        dsm = InflowDrivenDSM(dims=DIMS, inflow=inflow, lifetime_model=lifetime_model)
        dsm.compute()
        np.testing.assert_allclose(
            dsm.stock.values[-1] + dsm.outflow.values.sum(axis=0), inflow.values.sum(axis=0)
        )
        stocks[loss_rate] = dsm.stock.values
    assert np.all(stocks[0.1][1:] < stocks[0.0][1:])


def test_competing_risks_invalid_input():
    product = LogNormalLifetime(dims=DIMS, mean=MEAN, std=2.0)
    with pytest.raises(ValueError, match="at least two"):
        CompetingRisksLifetime(models=[product])
    other_dims = DimensionSet(dim_list=[DIMS["t"]])
    with pytest.raises(ValueError, match="Dimensions"):
        CompetingRisksLifetime(models=[product, Leaching(dims=other_dims, loss_rate=0.1)])
    with pytest.raises(ValueError, match="individual lifetime models"):
        CompetingRisksLifetime(
            models=[product, Leaching(dims=DIMS, loss_rate=0.1)], lt_factor_by_year=FACTOR
        )
    combined = CompetingRisksLifetime(models=[product, Leaching(dims=DIMS)])
    with pytest.raises(ValueError, match="loss_rate"):
        combined._check_prms_set()


YEAR_FACTOR = FlodymArray(
    dims=DIMS["t",], values=1 + 0.5 * np.sin(np.linspace(0, 3, DIMS["t"].len)) ** 2
)
UNEVEN_DIMS = DimensionSet(
    dim_list=[
        Dimension(name="time", letter="t", items=[0, 1, 2, 4, 5, 8, 9, 10, 13, 20], dtype=int),
        Dimension(name="product", letter="p", items=["a", "b"], dtype=str),
    ]
)


@pytest.mark.parametrize("test_dims", [DIMS, UNEVEN_DIMS], ids=["even", "uneven"])
def test_hazard_factor_equals_lifetime_factor_for_weibull(test_dims):
    weibull_shape = 2.5
    factor_values = 1 + 0.1 * np.arange(test_dims["t"].len)
    lt_factor = FlodymArray(dims=test_dims["t",], values=factor_values)
    hazard_factor = FlodymArray(dims=test_dims["t",], values=factor_values**-weibull_shape)
    prms = {"dims": test_dims, "weibull_scale": 10.0, "weibull_shape": weibull_shape}
    extended = WeibullLifetime(lt_factor_by_year=lt_factor, **prms)
    hazard = WeibullLifetime(hazard_factor_by_year=hazard_factor, **prms)
    np.testing.assert_allclose(
        hazard.conditional_survival, extended.conditional_survival, atol=1e-14
    )
    np.testing.assert_allclose(hazard.cumulative_survival, extended.cumulative_survival, atol=1e-14)
    np.testing.assert_allclose(hazard.outflow_pmf, extended.outflow_pmf, atol=1e-14)


def test_hazard_factor_equals_lifetime_factor_for_exponential():
    extended = ExponentialLifetime(dims=DIMS, mean=10.0, lt_factor_by_year=YEAR_FACTOR)
    hazard = ExponentialLifetime(dims=DIMS, mean=10.0, hazard_factor_by_year=1 / YEAR_FACTOR)
    np.testing.assert_allclose(hazard.cumulative_survival, extended.cumulative_survival, atol=1e-14)


def test_hazard_factor():
    plain = LogNormalLifetime(dims=DIMS, mean=MEAN, std=0.3, relative_std=True)
    neutral = LogNormalLifetime(
        dims=DIMS, mean=MEAN, std=0.3, relative_std=True, hazard_factor_by_year=1.0
    )
    np.testing.assert_allclose(
        neutral.cumulative_survival, plain.cumulative_survival, rtol=1e-12, atol=1e-15
    )

    # no exits in year 2010: conditional survival of 1 in that year, unchanged otherwise
    hazard_factor = FlodymArray(dims=DIMS["t",], values=np.ones(DIMS["t"].len))
    hazard_factor[{"t": 2010}] = 0.0
    moratorium = LogNormalLifetime(
        dims=DIMS, mean=MEAN, std=0.3, relative_std=True, hazard_factor_by_year=hazard_factor
    )
    i_t = DIMS["t"].index(2010)
    lower_tri = np.tril(np.ones((DIMS["t"].len,) * 2, dtype=bool))
    np.testing.assert_array_equal(moratorium.conditional_survival[i_t, : i_t + 1], 1.0)
    other_years = lower_tri.copy()
    other_years[i_t, :] = False
    np.testing.assert_allclose(
        moratorium.conditional_survival[other_years], plain.conditional_survival[other_years]
    )
    # rates of future cohorts stay zero
    assert np.all(moratorium.conditional_survival[~lower_tri] == 0.0)

    # more exits for a factor greater than one
    moratorium.hazard_factor_by_year = 2.0
    plain_cum_surv = plain.cumulative_survival[lower_tri]
    cum_surv = moratorium.cumulative_survival[lower_tri]
    partly_survived = (plain_cum_surv > 1e-6) & (plain_cum_surv < 1 - 1e-6)
    assert np.all(cum_surv[partly_survived] < plain_cum_surv[partly_survived])
    assert np.all(cum_surv <= plain_cum_surv)

    with pytest.raises(ValueError, match="must not be negative"):
        LogNormalLifetime(
            dims=DIMS, mean=MEAN, std=2.0, hazard_factor_by_year=-1.0
        ).cumulative_survival


@pytest.mark.parametrize("inflow_at, exposed_fraction", [("start", 1.0), ("middle", 0.5)])
def test_removal_share(inflow_at, exposed_fraction):
    share = FlodymArray(dims=DIMS["t",])
    share[{"t": 2005}] = 0.3
    share[{"t": 2012}] = 1.0
    prms = {"dims": DIMS, "mean": MEAN, "std": 0.3, "relative_std": True, "inflow_at": inflow_at}
    plain = LogNormalLifetime(**prms)
    removed = LogNormalLifetime(removal_share_by_year=share, **prms)
    for year, value in [(2005, 0.3), (2012, 1.0)]:
        i_t = DIMS["t"].index(year)
        np.testing.assert_allclose(
            removed.conditional_survival[i_t, :i_t],
            plain.conditional_survival[i_t, :i_t] * (1 - value),
        )
        np.testing.assert_allclose(
            removed.conditional_survival[i_t, i_t],
            plain.conditional_survival[i_t, i_t] * (1 - exposed_fraction * value),
        )
    # phase-out in 2012: older cohorts are gone afterwards
    i_t = DIMS["t"].index(2012)
    np.testing.assert_array_equal(removed.cumulative_survival[i_t:, :i_t], 0.0)

    with pytest.raises(ValueError, match="between 0 and 1"):
        LogNormalLifetime(removal_share_by_year=1.5, **prms).cumulative_survival


def test_removal_share_in_stock():
    share = FlodymArray(dims=DIMS["t",])
    share[{"t": 2010}] = 0.5
    inflow = StockArray(dims=DIMS, values=np.ones(DIMS.shape))
    dsms = {}
    for name, kwargs in [("plain", {}), ("removed", {"removal_share_by_year": share})]:
        lifetime_model = LogNormalLifetime(dims=DIMS, mean=MEAN, std=3.0, **kwargs)
        dsm = InflowDrivenDSM(dims=DIMS, inflow=inflow, lifetime_model=lifetime_model)
        dsm.compute()
        np.testing.assert_allclose(
            dsm.stock.values[-1] + dsm.outflow.values.sum(axis=0), inflow.values.sum(axis=0)
        )
        dsms[name] = dsm
    i_t = DIMS["t"].index(2010)
    plain, removed = dsms["plain"], dsms["removed"]
    np.testing.assert_allclose(removed.stock.values[:i_t], plain.stock.values[:i_t])
    assert np.all(removed.outflow.values[i_t] > plain.outflow.values[i_t])
    assert np.all(removed.stock.values[i_t:] < plain.stock.values[i_t:])


def test_hazard_factor_on_competing_risks():
    def models(**kwargs):
        return [
            LogNormalLifetime(dims=DIMS, mean=MEAN, std=0.3, relative_std=True, **kwargs),
            Leaching(dims=DIMS, loss_rate=0.05, **kwargs),
        ]

    combined = CompetingRisksLifetime(models=models(), hazard_factor_by_year=YEAR_FACTOR)
    separate = CompetingRisksLifetime(models=models(hazard_factor_by_year=YEAR_FACTOR))
    np.testing.assert_allclose(
        combined.cumulative_survival, separate.cumulative_survival, rtol=1e-12, atol=1e-15
    )


@pytest.mark.parametrize(
    "old_name, new_name", [("sf", "cumulative_survival"), ("pdf", "outflow_pmf")]
)
def test_deprecated_array_names(old_name, new_name):
    lifetime_model = LogNormalLifetime(dims=DIMS, mean=MEAN, std=0.3, relative_std=True)
    with pytest.warns(DeprecationWarning, match=new_name):
        values = getattr(lifetime_model, old_name)
    np.testing.assert_array_equal(values, getattr(lifetime_model, new_name))


def test_get_arrays():
    lifetime_model = LogNormalLifetime(dims=DIMS, mean=MEAN, std=0.3, relative_std=True)
    for getter, values in [
        (lifetime_model.get_cumulative_survival, lifetime_model.cumulative_survival),
        (lifetime_model.get_conditional_survival, lifetime_model.conditional_survival),
        (lifetime_model.get_outflow_pmf, lifetime_model.outflow_pmf),
    ]:
        array = getter()
        assert array.dims.letters == ("t", "c", "p")
        np.testing.assert_array_equal(array.values, values)
        # the returned values are a copy
        array.values[...] = 0.0
        assert np.any(values != 0.0)

    cohort_dim = Dimension(name="production year", letter="y", items=DIMS["t"].items)
    cum_surv = lifetime_model.get_cumulative_survival(cohort_dim=cohort_dim)
    assert cum_surv.dims.letters == ("t", "y", "p")
    with pytest.raises(ValueError, match="must not be part of"):
        lifetime_model.get_cumulative_survival(
            cohort_dim=Dimension(name="other", letter="p", items=DIMS["t"].items)
        )
    with pytest.raises(ValueError, match="same items"):
        lifetime_model.get_cumulative_survival(
            cohort_dim=Dimension(name="cohort", letter="c", items=[1, 2])
        )


def test_tabulated_lifetime():
    reference = LogNormalLifetime(dims=DIMS, mean=MEAN, std=0.3, relative_std=True)
    cum_surv = reference.get_cumulative_survival()
    tabulated = TabulatedLifetime(dims=DIMS, cumulative_survival_table=cum_surv)
    for name in ["cumulative_survival", "conditional_survival", "outflow_pmf"]:
        np.testing.assert_allclose(getattr(tabulated, name), getattr(reference, name))

    # same stock results
    inflow = StockArray(dims=DIMS, values=np.ones(DIMS.shape))
    results = []
    for lifetime_model in [reference, tabulated]:
        dsm = InflowDrivenDSM(dims=DIMS, inflow=inflow, lifetime_model=lifetime_model)
        dsm.compute()
        results.append(dsm.stock.values)
    np.testing.assert_allclose(results[1], results[0])

    # cumulative survival without the product dimension, as FlodymArray and as np.ndarray
    single = LogNormalLifetime(dims=DIMS["t",], mean=8.0, std=2.0)
    single_cum_surv = single.cumulative_survival[..., np.newaxis]
    for cum_surv in [single.get_cumulative_survival(), single_cum_surv]:
        tabulated = TabulatedLifetime(dims=DIMS, cumulative_survival_table=cum_surv)
        np.testing.assert_allclose(tabulated.cumulative_survival, single_cum_surv * np.ones(2))

    # period effects apply to tabulated cumulative survival as well
    reference.hazard_factor_by_year = YEAR_FACTOR
    tabulated = TabulatedLifetime(
        dims=DIMS,
        cumulative_survival_table=LogNormalLifetime(
            dims=DIMS, mean=MEAN, std=0.3, relative_std=True
        ).get_cumulative_survival(),
        hazard_factor_by_year=YEAR_FACTOR,
    )
    np.testing.assert_allclose(
        tabulated.cumulative_survival, reference.cumulative_survival, rtol=1e-12, atol=1e-15
    )


def test_tabulated_lifetime_invalid_input():
    cum_surv = LogNormalLifetime(dims=DIMS, mean=MEAN, std=2.0).cumulative_survival
    too_large = cum_surv.copy()
    too_large[3, 2] = 1.5
    future_cohort = cum_surv.copy()
    future_cohort[2, 3] = 0.5
    increasing = cum_surv.copy()
    increasing[10, 2] = (increasing[9, 2, :] + 1) / 2
    for values, match in [
        (too_large, "between 0 and 1"),
        (future_cohort, "zero for cohorts after"),
        (increasing, "must not increase"),
    ]:
        with pytest.raises(ValueError, match=match):
            TabulatedLifetime(dims=DIMS, cumulative_survival_table=values)
    with pytest.raises(ValueError, match="Lifetime factors"):
        TabulatedLifetime(dims=DIMS, cumulative_survival_table=cum_surv, lt_factor_by_year=FACTOR)
    with pytest.raises(ValueError, match="must be set"):
        TabulatedLifetime(dims=DIMS)._check_prms_set()


LONG_DIMS = DimensionSet(
    dim_list=[Dimension(name="time", letter="t", items=list(range(201)), dtype=int)]
)


def test_mean_lifetime_by_cohort():
    exponential = ExponentialLifetime(dims=LONG_DIMS, mean=10.0)
    truncated = exponential.get_mean_lifetime_by_cohort(tail="none").values
    extrapolated = exponential.get_mean_lifetime_by_cohort(tail="constant_rate").values
    # discretization error of the time steps is about 0.1 %
    np.testing.assert_allclose(truncated[:100], 10.0, rtol=1e-3)
    np.testing.assert_allclose(extrapolated, 10.0, rtol=1e-3)
    assert np.all(truncated[100:] <= extrapolated[100:])
    assert truncated[-1] < 1.0

    weibull = WeibullMeanStdLifetime(dims=LONG_DIMS, mean=10.0, std=4.0)
    np.testing.assert_allclose(
        weibull.get_mean_lifetime_by_cohort(tail="none").values[:100], 10.0, rtol=1e-3
    )

    # inflow in the middle of the time step, leaving after 5 years: exact
    fixed = FixedLifetime(dims=LONG_DIMS, mean=5.0)
    np.testing.assert_allclose(fixed.get_mean_lifetime_by_cohort(tail="none").values[:150], 5.0)
    # nothing leaves the stock in the last time step for young cohorts
    assert np.isinf(fixed.get_mean_lifetime_by_cohort(tail="constant_rate").values[-1])

    # period effects and competing risks are included
    faster = ExponentialLifetime(dims=LONG_DIMS, mean=10.0, hazard_factor_by_year=2.0)
    np.testing.assert_allclose(
        faster.get_mean_lifetime_by_cohort(tail="constant_rate").values, 5.0, rtol=5e-3
    )
    loss_rate = 0.05
    combined = CompetingRisksLifetime(
        models=[exponential, Leaching(dims=LONG_DIMS, loss_rate=loss_rate)]
    )
    expected = 1 / (1 / 10.0 - np.log(1 - loss_rate))
    np.testing.assert_allclose(
        combined.get_mean_lifetime_by_cohort(tail="constant_rate").values, expected, rtol=5e-3
    )

    with pytest.raises(ValueError, match="tail must be"):
        exponential.get_mean_lifetime_by_cohort(tail="linear")


def test_incompatible_array_shape_raises_value_error():
    with pytest.raises(ValueError, match="incompatible dimensions"):
        NormalLifetime(dims=DIMS, mean=np.ones(5), std=1.0)


@pytest.mark.parametrize("n_pts_per_interval", [0, 11])
def test_invalid_n_pts_per_interval(n_pts_per_interval):
    lifetime_model = NormalLifetime(
        dims=DIMS, mean=MEAN, std=1.0, n_pts_per_interval=n_pts_per_interval
    )
    with pytest.raises(ValueError, match="n_pts_per_interval must be between 1 and 10"):
        lifetime_model.cumulative_survival


REDUCE_DIMS = DimensionSet(
    dim_list=[
        Dimension(name="time", letter="t", items=list(range(2000, 2021)), dtype=int),
        Dimension(name="region", letter="r", items=["x", "y", "z"], dtype=str),
        Dimension(name="product", letter="p", items=["a", "b"], dtype=str),
        Dimension(name="element", letter="e", items=["C", "H"], dtype=str),
    ]
)
REDUCE_PRMS = {
    FixedLifetime: {"mean": [5.0, 8.0, 12.0]},
    ExponentialLifetime: {"mean": [5.0, 8.0, 12.0]},
    Leaching: {"loss_rate": [0.05, 0.1, 0.2]},
    NormalLifetime: {"mean": [5.0, 8.0, 12.0], "std": 2.0},
    FoldedNormalLifetime: {"mean": [5.0, 8.0, 12.0], "std": 2.0},
    LogNormalLifetime: {"mean": [5.0, 8.0, 12.0], "std": 2.0},
    WeibullLifetime: {"weibull_scale": [5.0, 8.0, 12.0], "weibull_shape": 2.0},
    WeibullMeanStdLifetime: {"mean": [5.0, 8.0, 12.0], "std": 2.0},
}
# vary along time and product, constant along region and element
REDUCE_FACTOR = FlodymArray(
    dims=REDUCE_DIMS[("t", "p")],
    values=np.linspace(1.0, 1.5, REDUCE_DIMS["t"].len)[:, None] * [1.0, 0.8],
)


def _reduce_prms(lt_cls):
    region_dims = REDUCE_DIMS["r",]
    return {
        name: FlodymArray(dims=region_dims, values=np.array(value))
        if isinstance(value, list)
        else value
        for name, value in REDUCE_PRMS[lt_cls].items()
    }


@pytest.mark.parametrize("lt_cls", list(REDUCE_PRMS))
@pytest.mark.parametrize(
    "factor_name",
    [None, "lt_factor_by_year", "lt_factor_by_cohort", "hazard_factor_by_year"],
)
def test_reduced_dims_equal_full_dims(lt_cls, factor_name, monkeypatch):
    """The survival computed on the reduced dimensions equals the one on the full dimensions."""
    factor_kwargs = {} if factor_name is None else {factor_name: REDUCE_FACTOR}
    kwargs = {"dims": REDUCE_DIMS, "n_pts_per_interval": 2, **_reduce_prms(lt_cls), **factor_kwargs}
    reduced = lt_cls(**kwargs)
    region_and_product = slice(None), slice(None), slice(None), slice(0, 1)
    expected_index = (
        region_and_product if factor_name else region_and_product[:2] + (slice(0, 1),) * 2
    )
    assert reduced._reduce_index == expected_index

    full_index = property(lambda self: (slice(None),) * len(self.shape))
    with monkeypatch.context() as patch:
        patch.setattr(type(reduced), "_reduce_index", full_index)
        full = lt_cls(**kwargs)
        full_arrays = [full.cumulative_survival, full.conditional_survival, full.outflow_pmf]
        full_mean = full.get_mean_lifetime_by_cohort(tail="none").values

    reduced_arrays = [
        reduced.cumulative_survival,
        reduced.conditional_survival,
        reduced.outflow_pmf,
    ]
    for reduced_array, full_array in zip(reduced_arrays, full_arrays):
        assert reduced_array.shape == full_array.shape
        np.testing.assert_allclose(reduced_array, full_array, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        reduced.get_mean_lifetime_by_cohort(tail="none").values, full_mean, rtol=1e-12
    )


def test_removal_share_and_in_place_change_with_reduced_dims():
    share = FlodymArray(dims=REDUCE_DIMS["e",], values=np.array([0.0, 0.1]))
    model = ExponentialLifetime(dims=REDUCE_DIMS, mean=10.0, removal_share_by_year=share)
    assert model._reduce_index[1:] == (slice(0, 1), slice(0, 1), slice(None))
    cum_surv = model.cumulative_survival
    no_removal = ExponentialLifetime(dims=REDUCE_DIMS, mean=10.0).cumulative_survival
    np.testing.assert_allclose(cum_surv[..., 0], no_removal[..., 0])
    assert np.all(cum_surv[2:, 0, :, :, 1] < cum_surv[2:, 0, :, :, 0])

    # changing a parameter in place along a reduced axis is taken into account after the reset
    model.mean[:, 1, ...] = 5.0
    model.reset_cached_arrays()
    assert model._reduce_index[1:] == (slice(None), slice(0, 1), slice(None))
    assert np.all(model.cumulative_survival[2:, 0, 1] < model.cumulative_survival[2:, 0, 0])


def test_survival_arrays_are_read_only():
    model = LogNormalLifetime(dims=REDUCE_DIMS, mean=10.0, std=3.0)
    for array in [model.cumulative_survival, model.conditional_survival, model.outflow_pmf]:
        assert array.shape == (21, 21, 3, 2, 2)
        with pytest.raises(ValueError, match="read-only"):
            array[...] = 0.0
    model.get_cumulative_survival().values[...] = 0.0
