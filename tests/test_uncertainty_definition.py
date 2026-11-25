"""Tests of the declaration of uncertain parameters and of their sampling."""

import numpy as np
import pytest
import scipy.stats
from pydantic import ValidationError

import flodym as fd
from flodym._sampling import unit_interval_numbers
from flodym.example_objects import get_example_mfa

TIME = fd.Dimension(name="Time", letter="t", dtype=int, items=[2000, 2001, 2002, 2003])
REGION = fd.Dimension(name="Region", letter="r", items=["north", "south", "east"])
WORLD = fd.Dimension(name="World region", letter="w", items=["EUR", "USA"])
COUNTRY = fd.Dimension(
    name="Country",
    letter="c",
    items=["DEU", "FRA", "ITA", "USA"],
    parent=WORLD,
    parent_items=["EUR", "EUR", "EUR", "USA"],
)


def time_by_region(name: str = "p") -> fd.Parameter:
    dims = fd.DimensionSet(dim_list=[TIME, REGION])
    values = np.arange(1.0, 13.0).reshape(4, 3)
    return fd.Parameter(name=name, dims=dims, values=values)


def quantiles_of_normal(samples: fd.SampledArray, nominal: fd.FlodymArray, std: float):
    """Unit-interval numbers of normally sampled values, with the sample axis last."""
    return scipy.stats.norm.cdf((samples.samples.values - nominal.values[..., None]) / std)


def test_shorthand_forms():
    normal = fd.NormalDistribution(std=1.0)
    entry = fd.ParameterUncertainty(distribution=normal, where={"r": "north"})
    definition = fd.UncertaintyDefinition(
        parameters={"a": normal, "b": entry, "c": [entry, normal]},
        normalize_over={"a": "r"},
    )
    assert definition.parameters["a"] == [fd.ParameterUncertainty(distribution=normal)]
    assert definition.parameters["b"] == [entry]
    second_entry = definition.parameters["c"][1]
    assert isinstance(second_entry, fd.ParameterUncertainty)
    assert second_entry.distribution is normal
    assert definition.normalize_over == {"a": ("r",)}
    single_letter = fd.ParameterUncertainty(distribution=normal, independent_over="r")
    assert single_letter.independent_over == ("r",)
    everything = fd.ParameterUncertainty(distribution=normal, independent_over="all")
    assert everything.independent_over == "all"


def test_normalize_over_of_certain_parameter_raises():
    with pytest.raises(ValidationError, match="have no uncertainty"):
        fd.UncertaintyDefinition(
            parameters={"a": fd.NormalDistribution(std=1.0)}, normalize_over={"b": ("r",)}
        )


def test_one_draw_per_entry_by_default():
    parameter = time_by_region()
    definition = fd.UncertaintyDefinition(parameters={"p": fd.NormalDistribution(std=0.5)})
    samples = definition.sample({"p": parameter}, n_samples=50, seed=3)["p"]
    unit_numbers = quantiles_of_normal(samples, parameter, 0.5)
    np.testing.assert_allclose(unit_numbers, np.broadcast_to(unit_numbers[:1, :1], (4, 3, 50)))
    assert np.unique(unit_numbers[0, 0]).size == 50


def test_independent_over_own_dimension():
    parameter = time_by_region()
    definition = fd.UncertaintyDefinition(
        parameters={
            "p": fd.ParameterUncertainty(
                distribution=fd.NormalDistribution(std=0.5), independent_over=("r",)
            )
        }
    )
    unit_numbers = quantiles_of_normal(
        definition.sample({"p": parameter}, n_samples=20, seed=3)["p"], parameter, 0.5
    )
    # the same along time, different between regions
    np.testing.assert_allclose(unit_numbers, np.broadcast_to(unit_numbers[:1], (4, 3, 20)))
    assert not np.allclose(unit_numbers[0, 0], unit_numbers[0, 1])


def test_independent_over_parent_dimension():
    parameter = fd.Parameter(name="p", dims=COUNTRY, values=np.array([1.0, 2.0, 3.0, 4.0]))
    definition = fd.UncertaintyDefinition(
        parameters={
            "p": fd.ParameterUncertainty(
                distribution=fd.NormalDistribution(std=0.5), independent_over=("w",)
            )
        }
    )
    unit_numbers = quantiles_of_normal(
        definition.sample({"p": parameter}, n_samples=20, seed=3)["p"], parameter, 0.5
    )
    np.testing.assert_allclose(unit_numbers[:3], np.broadcast_to(unit_numbers[:1], (3, 20)))
    assert not np.allclose(unit_numbers[0], unit_numbers[3])


def test_independent_over_all():
    parameter = time_by_region()
    definition = fd.UncertaintyDefinition(
        parameters={
            "p": fd.ParameterUncertainty(
                distribution=fd.NormalDistribution(std=0.5), independent_over="all"
            )
        }
    )
    unit_numbers = quantiles_of_normal(
        definition.sample({"p": parameter}, n_samples=2, seed=3)["p"], parameter, 0.5
    )
    assert np.unique(unit_numbers).size == unit_numbers.size


def test_where_restricts_the_entry():
    mfa = get_example_mfa()
    definition = fd.UncertaintyDefinition(
        parameters={
            "demolition yield": fd.ParameterUncertainty(
                distribution=fd.UniformDistribution(lower=0.01, upper=0.05), where={"e": "Cu"}
            )
        }
    )
    samples = definition.sample(mfa.parameters, n_samples=100, seed=1)["demolition yield"]
    values = samples.samples.values
    np.testing.assert_array_equal(values[[0, 2]], np.full((2, 100), 0.95))
    assert np.all((values[1] >= 0.01) & (values[1] <= 0.05))
    assert samples.samples.dims.letters == ("e", "s")
    assert samples.samples.dims["s"].items == list(range(100))


def test_where_with_ancestor_items_and_several_entries():
    parameter = fd.Parameter(name="p", dims=COUNTRY, values=np.array([1.0, 2.0, 3.0, 4.0]))
    definition = fd.UncertaintyDefinition(
        parameters={
            "p": [
                fd.ParameterUncertainty(
                    distribution=fd.UniformDistribution(lower=10.0, upper=11.0),
                    where={"c": "EUR"},
                ),
                fd.ParameterUncertainty(
                    distribution=fd.UniformDistribution(lower=20.0, upper=21.0),
                    where={"c": ["USA"]},
                ),
            ]
        }
    )
    values = definition.sample({"p": parameter}, n_samples=10, seed=1)["p"].samples.values
    assert np.all((values[:3] >= 10) & (values[:3] <= 11))
    assert np.all((values[3] >= 20) & (values[3] <= 21))


def test_where_with_a_subset_dimension():
    later = fd.Dimension(name="Later", letter="l", dtype=int, items=[2002, 2003])
    definition = fd.UncertaintyDefinition(
        parameters={
            "p": fd.ParameterUncertainty(
                distribution=fd.NormalDistribution(std=1.0), where={"t": later}
            )
        }
    )
    parameter = time_by_region()
    values = definition.sample({"p": parameter}, n_samples=3, seed=0)["p"].samples.values
    np.testing.assert_array_equal(
        values[:2], np.broadcast_to(parameter.values[:2, :, None], (2, 3, 3))
    )
    assert np.all(values[2:] != parameter.values[2:, :, None])


def test_overlapping_entries_raise():
    definition = fd.UncertaintyDefinition(
        parameters={
            "p": [
                fd.ParameterUncertainty(
                    distribution=fd.NormalDistribution(std=1.0), where={"r": ["north", "south"]}
                ),
                fd.ParameterUncertainty(
                    distribution=fd.NormalDistribution(std=1.0), where={"r": "south"}
                ),
            ]
        }
    )
    with pytest.raises(ValueError, match="entries 0 and 1 apply to the same values"):
        definition.sample({"p": time_by_region()}, n_samples=1, seed=0)


def test_misspelled_parameter_names_get_suggestions():
    mfa = get_example_mfa()
    definition = fd.UncertaintyDefinition(
        parameters={"eol machine": fd.NormalDistribution(relative_std=0.1)}
    )
    with pytest.raises(ValueError, match="Did you mean 'eol machines' instead of 'eol machine'"):
        definition.sample(mfa.parameters, n_samples=1, seed=0)


@pytest.mark.parametrize(
    ("parameters", "error"),
    [
        (get_example_mfa(), "Pass the parameters of the MFA system"),
        ([1.0], "must be a mapping of parameter names to FlodymArrays"),
        ({"p": 1.0}, r"Parameters \['p'\] are not FlodymArrays"),
    ],
)
def test_parameters_of_wrong_type_raise(parameters, error):
    definition = fd.UncertaintyDefinition(parameters={"p": fd.NormalDistribution(std=1.0)})
    with pytest.raises(TypeError, match=error):
        definition.sample(parameters, n_samples=1, seed=0)


def test_unknown_parameter_names_are_all_listed():
    definition = fd.UncertaintyDefinition(
        parameters={
            "p": fd.NormalDistribution(std=1.0),
            "q": fd.NormalDistribution(std=1.0),
            "r": fd.NormalDistribution(std=1.0),
        }
    )
    with pytest.raises(ValueError, match=r"\['q', 'r'\] are not among the given parameters"):
        definition.sample({"p": time_by_region()}, n_samples=1, seed=0)


def test_parameter_which_is_not_set_raises():
    parameter = fd.Parameter(name="p", dims=REGION)
    definition = fd.UncertaintyDefinition(parameters={"p": fd.NormalDistribution(std=1.0)})
    with pytest.raises(ValueError, match=r"\['p'\] are not set"):
        definition.sample({"p": parameter}, n_samples=1, seed=0)


@pytest.mark.parametrize(
    ("entry", "normalize_over", "message"),
    [
        ({"where": {"x": "north"}}, {}, r"unknown dimensions \['x'\] in where"),
        ({"where": {"r": ["north", "west"]}}, {}, "'west' is not an item of Region; items: north"),
        ({"where": {"r": []}}, {}, "where selects no values"),
        ({"independent_over": ("x",)}, {}, "unknown dimension 'x' in independent_over"),
        ({}, {"p": ("x",)}, r"normalize_over of parameter 'p': unknown dimensions \['x'\]"),
    ],
)
def test_unknown_dimensions_and_items_raise(entry, normalize_over, message):
    definition = fd.UncertaintyDefinition(
        parameters={
            "p": fd.ParameterUncertainty(distribution=fd.NormalDistribution(std=1.0), **entry)
        },
        normalize_over=normalize_over,
    )
    with pytest.raises(ValueError, match=message):
        definition.sample({"p": time_by_region()}, n_samples=1, seed=0)


def test_normalize_over_keeps_sums_and_draws_independently():
    mfa = get_example_mfa()
    definition = fd.UncertaintyDefinition(
        parameters={
            "composition eol machines": fd.TriangularDistribution(
                relative_lower=0.8, relative_upper=1.2
            )
        },
        normalize_over={"composition eol machines": ("e",)},
    )
    samples = definition.sample(mfa.parameters, n_samples=200, seed=5)["composition eol machines"]
    values = samples.samples.values
    np.testing.assert_allclose(values.sum(axis=0), 1.0, rtol=1e-14)
    ratios = values / mfa.parameters["composition eol machines"].values[:, None]
    assert not np.allclose(ratios[0], ratios[1])
    # rescaled values differ from their nominal values
    assert np.all(np.std(values, axis=1) > 0)


def test_normalize_over_can_push_values_beyond_the_bounds():
    parameter = fd.Parameter(name="share", dims=REGION, values=np.array([0.7, 0.2, 0.1]))
    definition = fd.UncertaintyDefinition(
        parameters={"share": fd.NormalDistribution(relative_std=0.5, lower=0.0, upper=0.8)},
        normalize_over={"share": "r"},
    )
    values = definition.sample({"share": parameter}, n_samples=200, seed=1)["share"].samples.values
    np.testing.assert_allclose(values.sum(axis=0), 1.0, rtol=1e-14)
    # the bounds apply before the rescaling, which can push values beyond them
    assert np.any(values > 0.8)
    assert np.all(values >= 0.0)


def test_normalize_over_with_partly_certain_values():
    parameter = fd.Parameter(name="share", dims=REGION, values=np.array([0.5, 0.3, 0.2]))
    definition = fd.UncertaintyDefinition(
        parameters={
            "share": fd.ParameterUncertainty(
                distribution=fd.UniformDistribution(lower=0.1, upper=0.3), where={"r": "east"}
            )
        },
        normalize_over={"share": ("r",)},
    )
    values = definition.sample({"share": parameter}, n_samples=20, seed=5)["share"].samples.values
    np.testing.assert_allclose(values.sum(axis=0), 1.0, rtol=1e-14)
    # the certain values keep their ratio
    np.testing.assert_allclose(values[0] / values[1], 0.5 / 0.3, rtol=1e-12)


def test_normalize_over_nominal_sum_zero_raises():
    parameter = fd.Parameter(
        name="share",
        dims=fd.DimensionSet(dim_list=[TIME, REGION]),
        values=np.array([[0.5, 0.5, 0.0]] * 2 + [[0.0, 0.0, 0.0]] * 2),
    )
    definition = fd.UncertaintyDefinition(
        parameters={"share": fd.NormalDistribution(std=0.1)}, normalize_over={"share": ("r",)}
    )
    with pytest.raises(ValueError, match=r"sum to zero over \['r'\].*at \(Time\) = \(2002\)"):
        definition.sample({"share": parameter}, n_samples=1, seed=0)


def test_seed_is_required():
    definition = fd.UncertaintyDefinition(parameters={"p": fd.NormalDistribution(std=0.5)})
    with pytest.raises(TypeError, match="seed"):
        definition.sample({"p": time_by_region()}, n_samples=3)


def test_same_seed_same_values_other_seed_other_values():
    mfa = get_example_mfa()
    definition = fd.UncertaintyDefinition(
        parameters={"eol machines": fd.LogNormalDistribution(relative_std=0.2)}
    )
    first = definition.sample(mfa.parameters, n_samples=10, seed=1)["eol machines"]
    second = definition.sample(mfa.parameters, n_samples=10, seed=1)["eol machines"]
    other = definition.sample(mfa.parameters, n_samples=10, seed=2)["eol machines"]
    np.testing.assert_array_equal(first.samples.values, second.samples.values)
    assert not np.any(first.samples.values == other.samples.values)


def test_sample_does_not_depend_on_other_samples():
    mfa = get_example_mfa()
    definition = fd.UncertaintyDefinition(
        parameters={
            "eol machines": fd.LogNormalDistribution(relative_std=0.2),
            "shredder yield": fd.ParameterUncertainty(
                distribution=fd.NormalDistribution(relative_std=0.05, lower=0.0, upper=1.0),
                independent_over=("e",),
            ),
        }
    )
    many = definition.sample(mfa.parameters, n_samples=150, seed=4)
    few = definition.sample(mfa.parameters, n_samples=5, seed=4)
    single = definition.sample(mfa.parameters, sample_indices=[130], seed=4)
    for name in many:
        np.testing.assert_array_equal(many[name].samples.values[..., :5], few[name].samples.values)
        np.testing.assert_array_equal(
            many[name].samples.values[..., 130:131], single[name].samples.values
        )
    assert single["eol machines"].samples.dims["s"].items == [130]


def test_other_parameters_do_not_change_the_draws():
    mfa = get_example_mfa()
    eol_machines = fd.LogNormalDistribution(relative_std=0.2)
    alone = fd.UncertaintyDefinition(parameters={"eol machines": eol_machines})
    with_others = fd.UncertaintyDefinition(
        parameters={
            "shredder yield": fd.NormalDistribution(relative_std=0.05, upper=1.0),
            "eol buildings": fd.NormalDistribution(relative_std=0.05),
            "eol machines": eol_machines,
        }
    )
    reordered = fd.UncertaintyDefinition(
        parameters={
            "eol machines": eol_machines,
            "eol buildings": fd.NormalDistribution(relative_std=0.05),
        }
    )
    expected = alone.sample(mfa.parameters, n_samples=20, seed=8)["eol machines"].samples.values
    for definition in (with_others, reordered):
        values = definition.sample(mfa.parameters, n_samples=20, seed=8)["eol machines"]
        np.testing.assert_array_equal(values.samples.values, expected)


def test_unit_interval_numbers_are_stable():
    """Regression test of the stream scheme: these numbers must never change."""
    np.testing.assert_array_equal(
        unit_interval_numbers(42, "eol machines", 0, 0, 3),
        [0.7718557926282195, 0.5911684397030386, 0.8002469324042571],
    )
    np.testing.assert_array_equal(
        unit_interval_numbers(42, "eol machines", 1, 17, 2),
        [0.3904666813350777, 0.09086473448378746],
    )


def test_sample_values_follow_the_stream_scheme():
    parameter = fd.Parameter(name="eol machines", dims=REGION, values=np.array([1.0, 2.0, 3.0]))
    definition = fd.UncertaintyDefinition(
        parameters={"eol machines": fd.UniformDistribution(lower=0.0, upper=1.0)}
    )
    values = definition.sample({"eol machines": parameter}, sample_indices=[0], seed=42)
    np.testing.assert_allclose(
        values["eol machines"].samples.values[:, 0], 0.7718557926282195, rtol=1e-15
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "exactly one of n_samples and sample_indices"),
        ({"n_samples": 2, "sample_indices": [0]}, "exactly one of n_samples and sample_indices"),
        ({"n_samples": 0}, "positive integer"),
        ({"sample_indices": []}, "is empty"),
        ({"sample_indices": [1, -1]}, "non-negative integers"),
        ({"sample_indices": [1, 1]}, "repeated"),
    ],
)
def test_invalid_sample_counts_raise(kwargs, message):
    definition = fd.UncertaintyDefinition(parameters={"p": fd.NormalDistribution(std=1.0)})
    with pytest.raises(ValueError, match=message):
        definition.sample({"p": time_by_region()}, seed=0, **kwargs)


def test_sampled_array_statistics():
    parameter = time_by_region()
    parameter.extensive = True
    definition = fd.UncertaintyDefinition(parameters={"p": fd.NormalDistribution(std=0.5)})
    sampled = definition.sample({"p": parameter}, n_samples=30, seed=3)["p"]
    values = sampled.samples.values
    np.testing.assert_allclose(sampled.mean.values, values.mean(axis=-1))
    np.testing.assert_allclose(sampled.std.values, values.std(axis=-1, ddof=1))
    np.testing.assert_allclose(sampled.quantile(0.9).values, np.quantile(values, 0.9, axis=-1))
    assert sampled.mean.dims.letters == ("t", "r")
    assert sampled.mean.extensive and sampled.std.extensive
    bands = sampled.quantile([0.05, 0.5, 0.95])
    assert bands.dims.letters == ("t", "r", "q")
    assert bands.dims["Quantile"].items == [0.05, 0.5, 0.95]
    assert bands.extensive
    np.testing.assert_allclose(
        bands.values, np.moveaxis(np.quantile(values, [0.05, 0.5, 0.95], axis=-1), 0, -1)
    )
    df = bands[{"r": "north"}].to_df(dim_to_columns="Quantile")
    assert list(df.columns) == [0.05, 0.5, 0.95]
    assert sampled.n_samples == 30


def test_quantile_outside_unit_interval_raises():
    definition = fd.UncertaintyDefinition(parameters={"p": fd.NormalDistribution(std=0.5)})
    sampled = definition.sample({"p": time_by_region()}, n_samples=3, seed=3)["p"]
    for q in (-0.1, 1.5, [0.5, 2.0]):
        with pytest.raises(ValueError, match="between 0 and 1"):
            sampled.quantile(q)


def test_std_of_a_single_sample_is_nan():
    definition = fd.UncertaintyDefinition(parameters={"p": fd.NormalDistribution(std=0.5)})
    sampled = definition.sample({"p": time_by_region()}, n_samples=1, seed=3)["p"]
    assert np.all(np.isnan(sampled.std.values))


def test_sample_dimension_letter_is_free():
    sample_letter = fd.Dimension(name="Size", letter="s", items=["small", "big"])
    parameter = fd.Parameter(name="p", dims=sample_letter, values=np.array([1.0, 2.0]))
    definition = fd.UncertaintyDefinition(parameters={"p": fd.NormalDistribution(std=0.5)})
    sampled = definition.sample({"p": parameter}, n_samples=3, seed=3)["p"]
    assert sampled.samples.dims.letters == ("s", "S")
    assert sampled.sample_dim.name == "Sample"


def test_sampled_array_slicing_and_summing_keep_the_samples():
    parameter = time_by_region()
    parameter.extensive = True
    definition = fd.UncertaintyDefinition(parameters={"p": fd.NormalDistribution(std=0.5)})
    sampled = definition.sample({"p": parameter}, n_samples=4, seed=3)["p"]
    north = sampled[{"r": "north"}]
    assert north.samples.dims.letters == ("t", "s")
    np.testing.assert_array_equal(north.samples.values, sampled.samples.values[:, 0])
    total = sampled.sum_over(("r",))
    assert total.dims.letters == ("t",)
    np.testing.assert_allclose(total.samples.values, sampled.samples.values.sum(axis=1))
    assert total.mean.extensive
    with pytest.raises(ValueError, match="sample dimension cannot be summed"):
        sampled.sum_over(("s",))
    with pytest.raises(ValueError, match="sample dimension cannot be sliced"):
        sampled[{"s": [1, 2]}]
    with pytest.raises(ValueError, match="slices the sample dimension"):
        sampled[1]
    with pytest.raises(ValueError, match="'west' is not an item of Region; items: north, south"):
        sampled[{"r": "west"}]
    with pytest.raises(ValueError, match=r"\['r'\]. .* give a new Dimension with a new letter"):
        sampled[{"r": ["north", "south"]}]
    north_and_south = fd.Dimension(name="Some regions", letter="o", items=["north", "south"])
    assert sampled[{"r": north_and_south}].samples.dims.letters == ("t", "o", "s")
    assert str(sampled) == "SampledArray 'p' with dims (t,r) and shape (4, 3); 4 samples"
