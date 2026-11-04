import numpy as np
import pytest
import scipy.stats

from flodym import (
    Dimension,
    DimensionSet,
    FlodymArray,
    InflowDrivenDSM,
    StockArray,
    NormalLifetime,
    FoldedNormalLifetime,
    LogNormalLifetime,
    WeibullLifetime,
    WeibullMeanStdLifetime,
    ExponentialLifetime,
    Leaching,
    CompetingRisksLifetime,
    TabulatedLifetime,
    FixedLifetime,
)

dims = DimensionSet(
    dim_list=[
        Dimension(name="time", letter="t", items=list(range(2000, 2031)), dtype=int),
        Dimension(name="product", letter="p", items=["a", "b"], dtype=str),
    ]
)
mean = FlodymArray(dims=dims["p",], values=np.array([8.0, 15.0]))
factor = FlodymArray(dims=dims, values=np.linspace(1.0, 2.0, dims["t"].len)[:, None] * [1, 1])


@pytest.mark.parametrize(
    "lt_cls", [NormalLifetime, FoldedNormalLifetime, LogNormalLifetime, WeibullMeanStdLifetime]
)
@pytest.mark.parametrize("factor_name", [None, "lt_factor_by_cohort", "lt_factor_by_year"])
def test_relative_std_equals_absolute_std(lt_cls, factor_name):
    factor_kwargs = {} if factor_name is None else {factor_name: factor}
    rel = lt_cls(dims=dims, mean=mean, std=0.3, relative_std=True, **factor_kwargs)
    abs_ = lt_cls(dims=dims, mean=mean, std=mean * 0.3, **factor_kwargs)
    np.testing.assert_allclose(rel.sf, abs_.sf, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("cv", [0.1, 0.5, 1.0, 2.0])
def test_weibull_mean_std_reproduces_moments(cv):
    lt = WeibullMeanStdLifetime(dims=dims, mean=mean, std=mean * cv)
    lt.sf
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
    lt_ms = WeibullMeanStdLifetime(dims=dims, mean=dist.mean(), std=dist.std())
    lt = WeibullLifetime(dims=dims, weibull_shape=shape, weibull_scale=scale)
    np.testing.assert_allclose(lt_ms.sf, lt.sf, rtol=1e-10, atol=1e-14)


def test_weibull_mean_std_raises_for_invalid_cv():
    lt = WeibullMeanStdLifetime(dims=dims, mean=mean, std=0.0)
    with pytest.raises(ValueError, match="Coefficient of variation"):
        lt.sf


def test_exponential_and_leaching():
    rate = 0.1
    expo = ExponentialLifetime(dims=dims, mean=10.0, inflow_at="start")
    ages = np.arange(dims["t"].len) + 1.0
    np.testing.assert_allclose(expo.sf[:, 0, 0], np.exp(-ages / 10.0))

    leach = Leaching(dims=dims, loss_rate=rate, inflow_at="start")
    np.testing.assert_allclose(leach.sf[:, 0, 0], (1 - rate) ** ages)
    np.testing.assert_allclose(leach.survival_rate[1:, 0, 0], 1 - rate)

    # Leaching is an exponential lifetime, also when lifetime factors are applied
    expo_equiv_mean = -1 / np.log(1 - rate)
    for factor_name in ["lt_factor_by_cohort", "lt_factor_by_year"]:
        leach = Leaching(dims=dims, loss_rate=rate, **{factor_name: factor})
        expo = ExponentialLifetime(dims=dims, mean=expo_equiv_mean, **{factor_name: factor})
        np.testing.assert_allclose(leach.sf, expo.sf, rtol=1e-12, atol=1e-15)


def test_competing_risks():
    leach = Leaching(dims=dims, loss_rate=0.05)
    product = LogNormalLifetime(dims=dims, mean=mean, std=0.3, relative_std=True)
    combined = CompetingRisksLifetime(models=[product, leach])
    assert combined.dims.letters == product.dims.letters
    np.testing.assert_allclose(combined.sf, product.sf * leach.sf)
    lower_tri = np.tril(np.ones((dims["t"].len,) * 2, dtype=bool))
    np.testing.assert_allclose(
        combined.survival_rate[lower_tri],
        (product.survival_rate * leach.survival_rate)[lower_tri],
    )
    pdf_before = combined.pdf.copy()

    # changing parameters of a model updates the combined model, for each accessed array
    leach.loss_rate = 0.2
    assert not np.allclose(combined.pdf, pdf_before)
    np.testing.assert_allclose(combined.sf, product.sf * leach.sf)
    product.mean = 20.0
    np.testing.assert_allclose(
        combined.survival_rate[lower_tri],
        (product.survival_rate * leach.survival_rate)[lower_tri],
    )

    # replacing the models
    combined.models = [product, leach, ExponentialLifetime(dims=dims, mean=30.0)]
    np.testing.assert_allclose(combined.sf, product.sf * leach.sf * combined.models[2].sf)


def test_competing_risks_in_stock():
    inflow = StockArray(dims=dims, values=np.ones(dims.shape))
    stocks = {}
    for loss_rate in [0.0, 0.1]:
        lifetime_model = CompetingRisksLifetime(
            models=[
                WeibullMeanStdLifetime(dims=dims, mean=mean, std=0.4, relative_std=True),
                Leaching(dims=dims, loss_rate=loss_rate),
            ]
        )
        dsm = InflowDrivenDSM(dims=dims, inflow=inflow, lifetime_model=lifetime_model)
        dsm.compute()
        np.testing.assert_allclose(
            dsm.stock.values[-1] + dsm.outflow.values.sum(axis=0), inflow.values.sum(axis=0)
        )
        stocks[loss_rate] = dsm.stock.values
    assert np.all(stocks[0.1][1:] < stocks[0.0][1:])


def test_competing_risks_invalid_input():
    product = LogNormalLifetime(dims=dims, mean=mean, std=2.0)
    with pytest.raises(ValueError, match="at least two"):
        CompetingRisksLifetime(models=[product])
    other_dims = DimensionSet(dim_list=[dims["t"]])
    with pytest.raises(ValueError, match="Dimensions"):
        CompetingRisksLifetime(models=[product, Leaching(dims=other_dims, loss_rate=0.1)])
    with pytest.raises(ValueError, match="individual lifetime models"):
        CompetingRisksLifetime(
            models=[product, Leaching(dims=dims, loss_rate=0.1)], lt_factor_by_year=factor
        )
    combined = CompetingRisksLifetime(models=[product, Leaching(dims=dims)])
    with pytest.raises(ValueError, match="loss_rate"):
        combined._check_prms_set()


year_factor = FlodymArray(
    dims=dims["t",], values=1 + 0.5 * np.sin(np.linspace(0, 3, dims["t"].len)) ** 2
)
uneven_dims = DimensionSet(
    dim_list=[
        Dimension(name="time", letter="t", items=[0, 1, 2, 4, 5, 8, 9, 10, 13, 20], dtype=int),
        Dimension(name="product", letter="p", items=["a", "b"], dtype=str),
    ]
)


@pytest.mark.parametrize("test_dims", [dims, uneven_dims], ids=["even", "uneven"])
def test_hazard_factor_equals_lifetime_factor_for_weibull(test_dims):
    weibull_shape = 2.5
    factor_values = 1 + 0.1 * np.arange(test_dims["t"].len)
    lt_factor = FlodymArray(dims=test_dims["t",], values=factor_values)
    hazard_factor = FlodymArray(dims=test_dims["t",], values=factor_values**-weibull_shape)
    prms = dict(dims=test_dims, weibull_scale=10.0, weibull_shape=weibull_shape)
    extended = WeibullLifetime(lt_factor_by_year=lt_factor, **prms)
    hazard = WeibullLifetime(hazard_factor_by_year=hazard_factor, **prms)
    np.testing.assert_allclose(hazard.survival_rate, extended.survival_rate, atol=1e-14)
    np.testing.assert_allclose(hazard.sf, extended.sf, atol=1e-14)
    np.testing.assert_allclose(hazard.pdf, extended.pdf, atol=1e-14)


def test_hazard_factor_equals_lifetime_factor_for_exponential():
    extended = ExponentialLifetime(dims=dims, mean=10.0, lt_factor_by_year=year_factor)
    hazard = ExponentialLifetime(dims=dims, mean=10.0, hazard_factor_by_year=1 / year_factor)
    np.testing.assert_allclose(hazard.sf, extended.sf, atol=1e-14)


def test_hazard_factor():
    plain = LogNormalLifetime(dims=dims, mean=mean, std=0.3, relative_std=True)
    neutral = LogNormalLifetime(
        dims=dims, mean=mean, std=0.3, relative_std=True, hazard_factor_by_year=1.0
    )
    np.testing.assert_allclose(neutral.sf, plain.sf, rtol=1e-12, atol=1e-15)

    # no exits in year 2010: survival rates of 1 in that year, unchanged otherwise
    hazard_factor = FlodymArray(dims=dims["t",], values=np.ones(dims["t"].len))
    hazard_factor[{"t": 2010}] = 0.0
    moratorium = LogNormalLifetime(
        dims=dims, mean=mean, std=0.3, relative_std=True, hazard_factor_by_year=hazard_factor
    )
    i_t = dims["t"].index(2010)
    lower_tri = np.tril(np.ones((dims["t"].len,) * 2, dtype=bool))
    np.testing.assert_array_equal(moratorium.survival_rate[i_t, : i_t + 1], 1.0)
    other_years = lower_tri.copy()
    other_years[i_t, :] = False
    np.testing.assert_allclose(
        moratorium.survival_rate[other_years], plain.survival_rate[other_years]
    )
    # rates of future cohorts stay zero
    assert np.all(moratorium.survival_rate[~lower_tri] == 0.0)

    # more exits for a factor greater than one
    moratorium.hazard_factor_by_year = 2.0
    plain_sf, sf = plain.sf[lower_tri], moratorium.sf[lower_tri]
    partly_survived = (plain_sf > 1e-6) & (plain_sf < 1 - 1e-6)
    assert np.all(sf[partly_survived] < plain_sf[partly_survived])
    assert np.all(sf <= plain_sf)

    with pytest.raises(ValueError, match="must not be negative"):
        LogNormalLifetime(dims=dims, mean=mean, std=2.0, hazard_factor_by_year=-1.0).sf


@pytest.mark.parametrize("inflow_at, exposed_fraction", [("start", 1.0), ("middle", 0.5)])
def test_removal_share(inflow_at, exposed_fraction):
    share = FlodymArray(dims=dims["t",])
    share[{"t": 2005}] = 0.3
    share[{"t": 2012}] = 1.0
    prms = dict(dims=dims, mean=mean, std=0.3, relative_std=True, inflow_at=inflow_at)
    plain = LogNormalLifetime(**prms)
    removed = LogNormalLifetime(removal_share_by_year=share, **prms)
    for year, value in [(2005, 0.3), (2012, 1.0)]:
        i_t = dims["t"].index(year)
        np.testing.assert_allclose(
            removed.survival_rate[i_t, :i_t], plain.survival_rate[i_t, :i_t] * (1 - value)
        )
        np.testing.assert_allclose(
            removed.survival_rate[i_t, i_t],
            plain.survival_rate[i_t, i_t] * (1 - exposed_fraction * value),
        )
    # phase-out in 2012: older cohorts are gone afterwards
    i_t = dims["t"].index(2012)
    np.testing.assert_array_equal(removed.sf[i_t:, :i_t], 0.0)

    with pytest.raises(ValueError, match="between 0 and 1"):
        LogNormalLifetime(removal_share_by_year=1.5, **prms).sf


def test_removal_share_in_stock():
    share = FlodymArray(dims=dims["t",])
    share[{"t": 2010}] = 0.5
    inflow = StockArray(dims=dims, values=np.ones(dims.shape))
    dsms = {}
    for name, kwargs in [("plain", {}), ("removed", {"removal_share_by_year": share})]:
        lifetime_model = LogNormalLifetime(dims=dims, mean=mean, std=3.0, **kwargs)
        dsm = InflowDrivenDSM(dims=dims, inflow=inflow, lifetime_model=lifetime_model)
        dsm.compute()
        np.testing.assert_allclose(
            dsm.stock.values[-1] + dsm.outflow.values.sum(axis=0), inflow.values.sum(axis=0)
        )
        dsms[name] = dsm
    i_t = dims["t"].index(2010)
    plain, removed = dsms["plain"], dsms["removed"]
    np.testing.assert_allclose(removed.stock.values[:i_t], plain.stock.values[:i_t])
    assert np.all(removed.outflow.values[i_t] > plain.outflow.values[i_t])
    assert np.all(removed.stock.values[i_t:] < plain.stock.values[i_t:])


def test_hazard_factor_on_competing_risks():
    def models(**kwargs):
        return [
            LogNormalLifetime(dims=dims, mean=mean, std=0.3, relative_std=True, **kwargs),
            Leaching(dims=dims, loss_rate=0.05, **kwargs),
        ]

    combined = CompetingRisksLifetime(models=models(), hazard_factor_by_year=year_factor)
    separate = CompetingRisksLifetime(models=models(hazard_factor_by_year=year_factor))
    np.testing.assert_allclose(combined.sf, separate.sf, rtol=1e-12, atol=1e-15)


def test_get_arrays():
    lifetime_model = LogNormalLifetime(dims=dims, mean=mean, std=0.3, relative_std=True)
    for getter, values in [
        (lifetime_model.get_sf, lifetime_model.sf),
        (lifetime_model.get_survival_rate, lifetime_model.survival_rate),
        (lifetime_model.get_pdf, lifetime_model.pdf),
    ]:
        array = getter()
        assert array.dims.letters == ("t", "c", "p")
        np.testing.assert_array_equal(array.values, values)
        # the returned values are a copy
        array.values[...] = 0.0
        assert np.any(values != 0.0)

    cohort_dim = Dimension(name="production year", letter="y", items=dims["t"].items)
    assert lifetime_model.get_sf(cohort_dim=cohort_dim).dims.letters == ("t", "y", "p")
    with pytest.raises(ValueError, match="must not be part of"):
        lifetime_model.get_sf(cohort_dim=Dimension(name="other", letter="p", items=dims["t"].items))
    with pytest.raises(ValueError, match="same items"):
        lifetime_model.get_sf(cohort_dim=Dimension(name="cohort", letter="c", items=[1, 2]))


def test_tabulated_lifetime():
    reference = LogNormalLifetime(dims=dims, mean=mean, std=0.3, relative_std=True)
    tabulated = TabulatedLifetime(dims=dims, survival_factor=reference.get_sf())
    for name in ["sf", "survival_rate", "pdf"]:
        np.testing.assert_allclose(getattr(tabulated, name), getattr(reference, name))

    # same stock results
    inflow = StockArray(dims=dims, values=np.ones(dims.shape))
    results = []
    for lifetime_model in [reference, tabulated]:
        dsm = InflowDrivenDSM(dims=dims, inflow=inflow, lifetime_model=lifetime_model)
        dsm.compute()
        results.append(dsm.stock.values)
    np.testing.assert_allclose(results[1], results[0])

    # survival factor without the product dimension, as FlodymArray and as np.ndarray
    single = LogNormalLifetime(dims=dims["t",], mean=8.0, std=2.0)
    for survival_factor in [single.get_sf(), single.sf[..., np.newaxis]]:
        tabulated = TabulatedLifetime(dims=dims, survival_factor=survival_factor)
        np.testing.assert_allclose(tabulated.sf, single.sf[..., np.newaxis] * np.ones(2))

    # period effects apply to tabulated survival factors as well
    reference.hazard_factor_by_year = year_factor
    tabulated = TabulatedLifetime(
        dims=dims,
        survival_factor=LogNormalLifetime(
            dims=dims, mean=mean, std=0.3, relative_std=True
        ).get_sf(),
        hazard_factor_by_year=year_factor,
    )
    np.testing.assert_allclose(tabulated.sf, reference.sf, rtol=1e-12, atol=1e-15)


def test_tabulated_lifetime_invalid_input():
    survival_factor = LogNormalLifetime(dims=dims, mean=mean, std=2.0).sf
    too_large = survival_factor.copy()
    too_large[3, 2] = 1.5
    future_cohort = survival_factor.copy()
    future_cohort[2, 3] = 0.5
    increasing = survival_factor.copy()
    increasing[10, 2] = (increasing[9, 2, :] + 1) / 2
    for values, match in [
        (too_large, "between 0 and 1"),
        (future_cohort, "zero for cohorts after"),
        (increasing, "must not increase"),
    ]:
        with pytest.raises(ValueError, match=match):
            TabulatedLifetime(dims=dims, survival_factor=values)
    with pytest.raises(ValueError, match="Lifetime factors"):
        TabulatedLifetime(dims=dims, survival_factor=survival_factor, lt_factor_by_year=factor)
    with pytest.raises(ValueError, match="must be set"):
        TabulatedLifetime(dims=dims)._check_prms_set()


long_dims = DimensionSet(
    dim_list=[Dimension(name="time", letter="t", items=list(range(201)), dtype=int)]
)


def test_mean_lifetime_by_cohort():
    exponential = ExponentialLifetime(dims=long_dims, mean=10.0)
    truncated = exponential.get_mean_lifetime_by_cohort(tail="none").values
    extrapolated = exponential.get_mean_lifetime_by_cohort(tail="constant_rate").values
    # discretization error of the time steps is about 0.1 %
    np.testing.assert_allclose(truncated[:100], 10.0, rtol=1e-3)
    np.testing.assert_allclose(extrapolated, 10.0, rtol=1e-3)
    assert np.all(truncated[100:] <= extrapolated[100:])
    assert truncated[-1] < 1.0

    weibull = WeibullMeanStdLifetime(dims=long_dims, mean=10.0, std=4.0)
    np.testing.assert_allclose(
        weibull.get_mean_lifetime_by_cohort(tail="none").values[:100], 10.0, rtol=1e-3
    )

    # inflow in the middle of the time step, leaving after 5 years: exact
    fixed = FixedLifetime(dims=long_dims, mean=5.0)
    np.testing.assert_allclose(fixed.get_mean_lifetime_by_cohort(tail="none").values[:150], 5.0)
    # nothing leaves the stock in the last time step for young cohorts
    assert np.isinf(fixed.get_mean_lifetime_by_cohort(tail="constant_rate").values[-1])

    # period effects and competing risks are included
    faster = ExponentialLifetime(dims=long_dims, mean=10.0, hazard_factor_by_year=2.0)
    np.testing.assert_allclose(
        faster.get_mean_lifetime_by_cohort(tail="constant_rate").values, 5.0, rtol=5e-3
    )
    loss_rate = 0.05
    combined = CompetingRisksLifetime(
        models=[exponential, Leaching(dims=long_dims, loss_rate=loss_rate)]
    )
    expected = 1 / (1 / 10.0 - np.log(1 - loss_rate))
    np.testing.assert_allclose(
        combined.get_mean_lifetime_by_cohort(tail="constant_rate").values, expected, rtol=5e-3
    )

    with pytest.raises(ValueError, match="tail must be"):
        exponential.get_mean_lifetime_by_cohort(tail="linear")
