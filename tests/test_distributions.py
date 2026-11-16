"""Tests of the distributions of uncertain parameter values."""

from typing import ClassVar

import numpy as np
import pytest
import scipy.stats
from pydantic import ValidationError

import flodym as fd
from flodym._sampling import unit_interval_numbers

TIME = fd.Dimension(name="Time", letter="t", items=[2000, 2001, 2002])
REGION = fd.Dimension(name="Region", letter="r", items=["north", "south"])


def sample_values(distribution: fd.Distribution, nominal, n_samples=50, seed=0):
    """Sampled values and their unit-interval numbers, by value (axis 0) and sample (axis 1).

    Each value gets its own draw, so the unit-interval numbers are those of the stream scheme.
    """
    nominal = np.atleast_1d(np.asarray(nominal, dtype=float))
    item = fd.Dimension(name="Item", letter="i", items=list(range(nominal.size)))
    parameter = fd.Parameter(name="p", dims=item, values=nominal)
    entry = fd.ParameterUncertainty(distribution=distribution, independent_over="all")
    definition = fd.UncertaintyDefinition(parameters={"p": entry})
    samples = definition.sample({"p": parameter}, n_samples=n_samples, seed=seed)["p"]
    unit_numbers = np.stack(
        [unit_interval_numbers(seed, "p", 0, i, nominal.size) for i in range(n_samples)], axis=-1
    )
    return samples.samples.values, unit_numbers


def sample_parameter(parameter: fd.Parameter, entry, n_samples=5, seed=0) -> np.ndarray:
    """Sampled values of one parameter, with the sample axis last."""
    uncertainty = fd.UncertaintyDefinition(parameters={parameter.name: entry})
    samples = uncertainty.sample({parameter.name: parameter}, n_samples=n_samples, seed=seed)
    return samples[parameter.name].samples.values


class ScaledBetaDistribution(fd.Distribution):
    """Beta distribution between the bounds, which it requires."""

    requires_bounds: ClassVar[bool] = True
    a: float
    b: float

    def frozen(self, nominal, lower, upper, a, b):
        return scipy.stats.beta(a, b, loc=lower, scale=upper - lower)

    def invalid_values(self, nominal, lower, upper, a, b):
        return {"a is not positive; give a > 0": a <= 0, "b is not positive; give b > 0": b <= 0}


def test_normal_matches_scipy():
    values, unit_numbers = sample_values(fd.NormalDistribution(std=0.5), [5.0, -2.0])
    expected = scipy.stats.norm(loc=[[5.0], [-2.0]], scale=0.5).ppf(unit_numbers)
    np.testing.assert_allclose(values, expected, rtol=1e-12)


def test_relative_std_equals_absolute_std():
    nominal = [5.0, -2.0, 0.0]
    values, unit_numbers = sample_values(fd.NormalDistribution(relative_std=0.1), nominal)
    expected = scipy.stats.norm(loc=[[5.0], [-2.0]], scale=[[0.5], [0.2]]).ppf(unit_numbers[:2])
    np.testing.assert_allclose(values[:2], expected, rtol=1e-12)
    # zero nominal value with a relative spread stays zero
    np.testing.assert_array_equal(values[2], 0.0)


def test_lognormal_has_nominal_mean_and_std():
    nominal = np.array([2.0, 10.0])
    frozen = fd.LogNormalDistribution(relative_std=0.3).frozen(
        nominal, std=None, relative_std=np.full(2, 0.3)
    )
    np.testing.assert_allclose(frozen.mean(), nominal, rtol=1e-12)
    np.testing.assert_allclose(frozen.std(), 0.3 * nominal, rtol=1e-12)
    values, unit_numbers = sample_values(fd.LogNormalDistribution(relative_std=0.3), nominal)
    np.testing.assert_allclose(values, frozen.ppf(unit_numbers.T).T, rtol=1e-12)


def test_lognormal_with_absolute_std():
    values, unit_numbers = sample_values(fd.LogNormalDistribution(std=1.0), [4.0])
    log_variance = np.log(1 + (1.0 / 4.0) ** 2)
    expected = scipy.stats.lognorm(s=np.sqrt(log_variance), scale=4.0 * np.exp(-log_variance / 2))
    np.testing.assert_allclose(values[0], expected.ppf(unit_numbers[0]), rtol=1e-12)


def test_uniform_between_bounds():
    values, unit_numbers = sample_values(fd.UniformDistribution(lower=1.0, upper=3.0), [2.0, 9.0])
    np.testing.assert_allclose(values, 1.0 + 2.0 * unit_numbers, rtol=1e-12)


def test_uniform_with_relative_bounds():
    distribution = fd.UniformDistribution(relative_lower=0.5, relative_upper=1.5)
    values, unit_numbers = sample_values(distribution, [2.0])
    np.testing.assert_allclose(values[0], 1.0 + 2.0 * unit_numbers[0], rtol=1e-12)


@pytest.mark.parametrize("mode", [None, 1.5])
def test_triangular_matches_scipy(mode):
    values, unit_numbers = sample_values(
        fd.TriangularDistribution(lower=1.0, upper=4.0, mode=mode), [2.0]
    )
    peak = 2.0 if mode is None else mode
    expected = scipy.stats.triang(c=(peak - 1.0) / 3.0, loc=1.0, scale=3.0).ppf(unit_numbers[0])
    np.testing.assert_allclose(values[0], expected, rtol=1e-12)


def test_truncated_normal_matches_truncnorm():
    distribution = fd.NormalDistribution(std=0.2, lower=0.7, upper=1.0)
    values, unit_numbers = sample_values(distribution, [0.92], n_samples=500)
    expected = scipy.stats.truncnorm(
        a=(0.7 - 0.92) / 0.2, b=(1.0 - 0.92) / 0.2, loc=0.92, scale=0.2
    ).ppf(unit_numbers[0])
    np.testing.assert_allclose(values[0], expected, rtol=1e-10)
    assert np.all((values >= 0.7) & (values <= 1.0))


NOMINAL = np.array([0.5, 0.9, 0.99])


@pytest.mark.parametrize(
    ("distribution", "lower", "upper"),
    [
        (fd.NormalDistribution(relative_std=0.5, lower=0.0, upper=1.0), 0.0, 1.0),
        (fd.LogNormalDistribution(relative_std=2.0, relative_upper=1.5), 0.0, 1.5 * NOMINAL),
        (
            fd.TriangularDistribution(relative_lower=0.5, relative_upper=1.1),
            0.5 * NOMINAL,
            1.1 * NOMINAL,
        ),
    ],
)
def test_sampled_values_respect_bounds(distribution, lower, upper):
    values, _ = sample_values(distribution, NOMINAL, n_samples=1000)
    assert np.all(values >= np.reshape(lower, (-1, 1)))
    assert np.all(values <= np.reshape(upper, (-1, 1)))
    assert np.any(values > NOMINAL[:, None])


@pytest.mark.parametrize(
    "distribution",
    [
        fd.NormalDistribution(std=0.0),
        fd.NormalDistribution(relative_std=0.0, lower=-1.0),
        fd.LogNormalDistribution(std=0.0),
        fd.LogNormalDistribution(relative_std=0.0),
    ],
)
def test_zero_spread_stays_nominal(distribution):
    values, _ = sample_values(distribution, [3.0, 0.5], n_samples=5)
    np.testing.assert_array_equal(values, np.tile([[3.0], [0.5]], (1, 5)))


def test_lognormal_with_all_nominal_values_zero_stays_zero():
    values, _ = sample_values(fd.LogNormalDistribution(relative_std=0.2), [0.0, 0.0], n_samples=5)
    np.testing.assert_array_equal(values, 0.0)


def test_lognormal_with_zero_nominal_value_stays_zero():
    values, _ = sample_values(fd.LogNormalDistribution(std=1.0), [0.0, 2.0], n_samples=20)
    np.testing.assert_array_equal(values[0], 0.0)
    assert np.all(values[1] > 0)
    assert np.unique(values[1]).size == 20


def test_bounds_without_width_give_the_bound():
    distribution = fd.UniformDistribution(relative_lower=0.8, relative_upper=1.2)
    values, _ = sample_values(distribution, [0.0, 1.0], n_samples=5)
    np.testing.assert_array_equal(values[0], 0.0)
    values, _ = sample_values(fd.TriangularDistribution(lower=2.0, upper=2.0), [2.0], n_samples=5)
    np.testing.assert_array_equal(values[0], 2.0)


def test_negative_nominal_value_raises_for_lognormal():
    parameter = fd.Parameter(name="flow", dims=REGION, values=np.array([1.0, -1.0]))
    with pytest.raises(
        ValueError, match=r"LogNormalDistribution of parameter 'flow' at Region = south: nominal"
    ):
        sample_parameter(parameter, fd.LogNormalDistribution(relative_std=0.1))


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"std": 1.0, "relative_std": 0.1},
        {"std": 1.0, "lower": 0.0, "relative_lower": 0.5},
        {"std": 1.0, "upper": 1.0, "relative_upper": 1.5},
    ],
)
@pytest.mark.parametrize("distribution_class", [fd.NormalDistribution, fd.LogNormalDistribution])
def test_invalid_spread_or_bounds_raise_at_construction(distribution_class, kwargs):
    with pytest.raises(ValidationError):
        distribution_class(**kwargs)


@pytest.mark.parametrize(
    ("construct", "message"),
    [
        (lambda: fd.NormalDistribution(std=-1), "std is negative; give a non-negative std"),
        (
            lambda: fd.LogNormalDistribution(relative_std=-0.1),
            "relative_std is negative; give a non-negative relative_std",
        ),
        (lambda: fd.UniformDistribution(lower=1, upper=0), "greater than upper bound; give lower"),
        (lambda: fd.NormalDistribution(std=float("nan")), "std is NaN; give a number"),
        (
            lambda: fd.TriangularDistribution(lower=0.0, upper=1.0, mode=2.0),
            "mode is not between the bounds; give a mode between lower and upper",
        ),
        (lambda: ScaledBetaDistribution(a=-1.0, b=2.0, lower=0, upper=1), "a is not positive"),
    ],
)
def test_invalid_numbers_raise_at_construction(construct, message):
    with pytest.raises(ValidationError, match=message):
        construct()


@pytest.mark.parametrize("distribution_class", [fd.UniformDistribution, fd.TriangularDistribution])
@pytest.mark.parametrize("kwargs", [{}, {"lower": 0.0}, {"relative_upper": 1.2}])
def test_missing_bounds_raise_at_construction(distribution_class, kwargs):
    with pytest.raises(ValidationError, match="requires a lower bound"):
        distribution_class(**kwargs)


def test_unknown_argument_raises_at_construction():
    with pytest.raises(ValidationError):
        fd.NormalDistribution(sigma=1.0)  # ty: ignore[unknown-argument]


def test_base_class_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        fd.Distribution()  # ty: ignore[abstract-class-instantiation]

    class Incomplete(fd.Distribution):
        spread: float

    with pytest.raises(TypeError, match="abstract"):
        Incomplete(spread=1.0)  # ty: ignore[abstract-class-instantiation]


@pytest.mark.parametrize(
    ("distribution", "location", "message"),
    [
        (
            fd.NormalDistribution(std=fd.FlodymArray(dims=REGION, values=np.array([0.1, -1.0]))),
            "south",
            "std is negative",
        ),
        (fd.NormalDistribution(std=1.0, relative_lower=2.0, upper=1.0), "south", "lower bound is"),
        (fd.TriangularDistribution(lower=0.0, upper=0.5), "south", r"mode \(the nominal value\)"),
        (fd.NormalDistribution(std=0.01, lower=0.0, upper=0.1), "south", "no probability"),
    ],
)
def test_invalid_values_raise_with_parameter_and_items(distribution, location, message):
    parameter = fd.Parameter(name="share", dims=REGION, values=np.array([0.2, 0.8]))
    with pytest.raises(
        ValueError, match=rf"of parameter 'share' at Region = {location}: {message}"
    ):
        sample_parameter(parameter, distribution)


def test_location_of_invalid_values_over_several_dimensions():
    parameter = fd.Parameter(
        name="p",
        dims=fd.DimensionSet(dim_list=[TIME, REGION]),
        values=np.array([[1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]]),
    )
    with pytest.raises(
        ValueError,
        match=r"at \(Time, Region\) = \(2000, south\), \(2001, north\), \(2001, south\) and 2 more",
    ):
        sample_parameter(parameter, fd.LogNormalDistribution(relative_std=0.1))


def test_nan_nominal_value_raises():
    parameter = fd.Parameter(name="share", dims=REGION, values=np.array([0.2, np.nan]))
    with pytest.raises(ValueError, match=r"parameter 'share' at Region = south: nominal value is"):
        sample_parameter(parameter, fd.NormalDistribution(std=0.1))


def test_distribution_parameters_as_arrays_over_a_subset_of_dims():
    dims = fd.DimensionSet(dim_list=[TIME, REGION])
    parameter = fd.Parameter(name="p", dims=dims, values=np.array([[1.0, 2.0]] * 3))
    std = fd.FlodymArray(dims=REGION, values=np.array([0.1, 0.4]))
    lower = fd.FlodymArray(dims=REGION, values=np.array([-10.0, 1.9]))
    samples = sample_parameter(parameter, fd.NormalDistribution(std=std, lower=lower), 20)
    # one draw for all values: the unit-interval number is the same for all values
    unit_numbers = scipy.stats.norm.cdf((samples[:, 0, :] - 1.0) / 0.1)
    np.testing.assert_allclose(unit_numbers, np.tile(unit_numbers[:1], (3, 1)), rtol=1e-12)
    truncated = scipy.stats.truncnorm(a=(1.9 - 2.0) / 0.4, b=np.inf, loc=2.0, scale=0.4)
    np.testing.assert_allclose(samples[:, 1, :], truncated.ppf(unit_numbers), rtol=1e-9)
    assert np.all(samples[:, 1, :] >= 1.9)


def test_distribution_parameter_over_parent_dimension():
    world = fd.Dimension(name="World region", letter="w", items=["EUR", "USA"])
    countries = fd.Dimension(
        name="Country",
        letter="c",
        items=["DEU", "FRA", "USA"],
        parent=world,
        parent_items=["EUR", "EUR", "USA"],
    )
    parameter = fd.Parameter(name="p", dims=countries, values=np.array([1.0, 1.0, 1.0]))
    std = fd.FlodymArray(dims=world, values=np.array([0.1, 0.0]))
    samples = sample_parameter(parameter, fd.NormalDistribution(std=std), 10)
    np.testing.assert_array_equal(samples[0], samples[1])
    np.testing.assert_array_equal(samples[2], 1.0)
    assert np.unique(samples[0]).size == 10


def test_distribution_array_with_unknown_dimension_raises():
    parameter = fd.Parameter(name="p", dims=TIME, values=np.ones(3))
    std = fd.FlodymArray(dims=REGION, values=np.array([0.1, 0.4]))
    with pytest.raises(ValueError, match=r"parameter 'p'.*std has the dimension 'Region'"):
        sample_parameter(parameter, fd.NormalDistribution(std=std))


def test_distribution_array_with_other_items_raises():
    parameter = fd.Parameter(name="p", dims=REGION, values=np.ones(2))
    other_regions = fd.Dimension(name="Region", letter="r", items=["east", "west"])
    std = fd.FlodymArray(dims=other_regions, values=np.array([0.1, 0.4]))
    with pytest.raises(ValueError, match="std has other items of dimension 'Region'"):
        sample_parameter(parameter, fd.NormalDistribution(std=std))


class BetaDistribution(fd.Distribution):
    """Beta distribution with the nominal value as mean, as in the Distribution docstring."""

    concentration: fd.FlodymArray | float

    def frozen(self, nominal, concentration):
        return scipy.stats.beta(nominal * concentration, (1 - nominal) * concentration)

    def invalid_values(self, nominal, concentration):
        return {"nominal value is not between 0 and 1": (nominal < 0) | (nominal > 1)}

    def no_spread(self, nominal, concentration):
        return (nominal == 0) | (nominal == 1)


def test_custom_distribution_relative_to_the_nominal_value():
    nominal = [0.3, 0.6, 1.0]
    values, unit_numbers = sample_values(
        BetaDistribution(concentration=20.0, upper=0.8), nominal, n_samples=10, seed=7
    )
    for row in range(2):
        frozen = scipy.stats.beta(20.0 * nominal[row], 20.0 * (1 - nominal[row]))
        expected = frozen.ppf(unit_numbers[row] * frozen.cdf(0.8))
        np.testing.assert_allclose(values[row], expected, rtol=1e-12)
    # without spread at 1, and then capped by the upper bound
    np.testing.assert_array_equal(values[2], 0.8)
    mean = BetaDistribution(concentration=20.0).frozen(np.array([0.3]), np.array([20.0])).mean()
    np.testing.assert_allclose(mean, 0.3)


def test_custom_checks_of_nominal_values_pass_at_construction():
    # created with numbers, the checks run with NaN nominal values, which they must not flag
    distribution = BetaDistribution(concentration=20.0)
    with pytest.raises(ValidationError, match="a is not positive; give a > 0"):
        ScaledBetaDistribution(a=0.0, b=2.0, lower=0.0, upper=1.0)
    assert distribution.concentration == 20.0


def test_custom_distribution_checks_values():
    parameter = fd.Parameter(name="share", dims=REGION, values=np.array([0.3, 1.6]))
    with pytest.raises(ValueError, match="at Region = south: nominal value is not between 0 and 1"):
        sample_parameter(parameter, BetaDistribution(concentration=20.0))


def test_custom_distribution_defined_by_its_bounds():
    with pytest.raises(ValidationError, match="requires a lower bound"):
        ScaledBetaDistribution(a=2.0, b=3.0, upper=1.0)
    distribution = ScaledBetaDistribution(a=2.0, b=3.0, relative_lower=0.5, upper=4.0)
    values, unit_numbers = sample_values(distribution, [2.0, 8.0], n_samples=10)
    expected = scipy.stats.beta(2.0, 3.0, loc=1.0, scale=3.0).ppf(unit_numbers[0])
    np.testing.assert_allclose(values[0], expected, rtol=1e-12)
    # lower and upper bound are equal
    np.testing.assert_array_equal(values[1], 4.0)


def test_custom_distribution_with_invalid_parameters_raises():
    parameter = fd.Parameter(name="share", dims=REGION, values=np.array([0.3, 0.6]))
    concentration = fd.FlodymArray(dims=REGION, values=np.array([2.0, -1.0]))
    with pytest.raises(ValueError, match=r"at Region = south: the distribution gives NaN values"):
        sample_parameter(parameter, BetaDistribution(concentration=concentration))


def test_custom_distribution_returning_no_scipy_distribution_raises():
    class NoScipy(fd.Distribution):
        def frozen(self, nominal):
            return None

    parameter = fd.Parameter(name="share", dims=REGION, values=np.array([0.3, 0.6]))
    with pytest.raises(TypeError, match=r"NoScipy.frozen\(\) must return a frozen scipy.stats"):
        sample_parameter(parameter, NoScipy())
