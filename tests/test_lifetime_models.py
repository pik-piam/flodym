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


