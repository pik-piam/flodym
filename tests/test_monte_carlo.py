"""Tests of run_monte_carlo and MonteCarloResult, with plain model functions.

The model function of an MFA system is tested in test_make_model_function.py.
"""

import logging

import numpy as np
import pytest
import scipy.stats

import flodym as fd
from flodym.example_objects import get_example_mfa

TIME = fd.Dimension(name="Time", letter="t", dtype=int, items=[2000, 2001, 2002, 2003])
REGION = fd.Dimension(name="Region", letter="r", items=["north", "south", "east"])
DIMS = fd.DimensionSet(dim_list=[TIME, REGION])

EXAMPLE_UNCERTAINTY = fd.UncertaintyDefinition(
    parameters={
        "eol machines": fd.LogNormalDistribution(relative_std=0.2),
        "eol buildings": fd.LogNormalDistribution(relative_std=0.2),
        "shredder yield": fd.ParameterUncertainty(
            distribution=fd.NormalDistribution(relative_std=0.05, lower=0.0, upper=1.0),
            independent_over="e",
        ),
        "demolition yield": fd.ParameterUncertainty(
            distribution=fd.UniformDistribution(lower=0.01, upper=0.05), where={"e": "Cu"}
        ),
        "composition eol machines": fd.TriangularDistribution(
            relative_lower=0.8, relative_upper=1.2
        ),
    },
    normalize_over={"composition eol machines": "e"},
)


def linear_parameters() -> dict[str, fd.FlodymArray]:
    return {
        "a": fd.Parameter(name="a", dims=DIMS, values=np.full((4, 3), 10.0)),
        "b": fd.Parameter(name="b", dims=DIMS, values=np.full((4, 3), 5.0)),
        "c": fd.Parameter(name="c", dims=DIMS, values=np.full((4, 3), 1.0)),
    }


LINEAR_UNCERTAINTY = fd.UncertaintyDefinition(
    parameters={"a": fd.NormalDistribution(std=1.0), "b": fd.NormalDistribution(std=0.5)}
)


def linear_model(parameters: dict[str, fd.FlodymArray]) -> dict[str, fd.FlodymArray | float]:
    y = 2 * parameters["a"] + 3 * parameters["b"] + parameters["c"]
    return {"y": y, "a total": float(parameters["a"].sum_values())}


def run_linear(model=linear_model, **kwargs) -> fd.MonteCarloResult:
    kwargs = {"seed": 1, **kwargs}
    if "sample_indices" not in kwargs:
        kwargs.setdefault("n_samples", 3)
    return fd.run_monte_carlo(model, linear_parameters(), LINEAR_UNCERTAINTY, **kwargs)


# --- outputs --------------------------------------------------------------------------------------


def test_outputs_equal_a_loop_over_the_sampled_values():
    parameters = linear_parameters()
    result = fd.run_monte_carlo(linear_model, parameters, LINEAR_UNCERTAINTY, n_samples=20, seed=11)
    inputs = LINEAR_UNCERTAINTY.sample(parameters, n_samples=20, seed=11)
    a = inputs["a"].samples.values
    b = inputs["b"].samples.values
    np.testing.assert_array_equal(result.outputs["y"].samples.values, 2 * a + 3 * b + 1.0)
    np.testing.assert_allclose(result.outputs["a total"].samples.values, a.sum(axis=(0, 1)))
    assert result.outputs["y"].samples.dims.letters == ("t", "r", "s")
    assert result.outputs["a total"].mean.dims.letters == ()
    assert result.sample_indices == list(range(20))
    assert result.sample_dim.items == list(range(20))
    assert result.seed == 11
    assert result.rejection_rate == 0.0
    assert result.outputs.y is result.outputs["y"]
    assert str(result) == (
        "MonteCarloResult with 2 outputs ('y', 'a total'); 20 of 20 samples successful "
        "(rejection rate 0.0%); seed 11"
    )


def test_statistics_of_a_linear_output_match_the_analytical_values():
    n_samples = 1500
    y = run_linear(n_samples=n_samples, seed=2024).outputs["y"]
    # y = 2 a + 3 b + c with independent a ~ N(10, 1) and b ~ N(5, 0.5)
    expected_mean = 2 * 10.0 + 3 * 5.0 + 1.0
    expected_std = np.sqrt((2 * 1.0) ** 2 + (3 * 0.5) ** 2)
    standard_error_mean = expected_std / np.sqrt(n_samples)
    standard_error_std = expected_std / np.sqrt(2 * (n_samples - 1))
    np.testing.assert_allclose(y.mean.values, expected_mean, atol=4 * standard_error_mean)
    np.testing.assert_allclose(y.std.values, expected_std, atol=4 * standard_error_std)
    # standard error of a quantile: sqrt(q (1 - q) / n) / density at the quantile
    density = scipy.stats.norm.pdf(1.959964) / expected_std
    standard_error_quantile = np.sqrt(0.975 * 0.025 / n_samples) / density
    np.testing.assert_allclose(
        y.quantile(0.975).values,
        expected_mean + 1.959964 * expected_std,
        atol=4 * standard_error_quantile,
    )


def test_model_gets_copies_of_all_parameters():
    parameters = linear_parameters()
    nominal_a = parameters["a"].values.copy()
    seen_names = []

    def modifying_model(given: dict[str, fd.FlodymArray]) -> dict[str, fd.FlodymArray]:
        seen_names.append(sorted(given))
        given["a"].values *= 2.0
        given["derived"] = given["a"] + given["c"]
        return {"a": given["a"], "derived": given["derived"]}

    result = fd.run_monte_carlo(
        modifying_model, parameters, LINEAR_UNCERTAINTY, n_samples=5, seed=3
    )
    inputs = LINEAR_UNCERTAINTY.sample(parameters, n_samples=5, seed=3)
    np.testing.assert_array_equal(
        result.outputs["a"].samples.values, 2 * inputs["a"].samples.values
    )
    np.testing.assert_array_equal(parameters["a"].values, nominal_a)
    assert "derived" not in parameters
    assert all(names == ["a", "b", "c"] for names in seen_names)


def test_parameter_attributes_are_accessible_in_the_model():
    result = run_linear(model=lambda given: {"y": given.a * given.c}, n_samples=2)
    assert result.outputs["y"].n_samples == 2


def test_extensivity_of_outputs_is_kept():
    def model(given):
        flow = given["a"].copy()
        flow.extensive = True
        return {"flow": flow, "share": given["a"] / given["b"]}

    result = run_linear(model=model)
    assert result.outputs["flow"].mean.extensive is True
    assert result.outputs["flow"].quantile([0.1, 0.9]).extensive is True
    assert result.outputs["share"].std.extensive is None


def test_nan_outputs():
    def model(given):
        partly_nan = given["a"].copy()
        partly_nan.values[0, 0] = np.nan
        sometimes_nan = float("nan") if given["a"].values[0, 0] > 10.0 else 1.0
        return {"partly nan": partly_nan, "sometimes nan": sometimes_nan}

    # NaN values in the same places pass the determinism check
    result = run_linear(model=model, n_samples=10)
    assert np.all(np.isnan(result.outputs["partly nan"].samples.values[0, 0]))
    assert np.any(np.isnan(result.outputs["sometimes nan"].samples.values))


def test_outputs_which_are_always_nan_do_not_vary(caplog):
    with caplog.at_level(logging.WARNING, logger="flodym"):
        run_linear(model=lambda given: {"nan": float("nan")})
    assert "No output differs between the Monte Carlo samples" in caplog.text


# --- checks ---------------------------------------------------------------------------------------


def test_determinism_check_catches_state_carried_between_runs():
    factor = {"value": 1.0}

    def compounding_model(given):
        factor["value"] *= 1.5
        return {"y": given["a"] * factor["value"], "c": given["c"]}

    with pytest.raises(ValueError, match=r"different values for the outputs 'y' \(largest"):
        run_linear(model=compounding_model)


def test_determinism_check_catches_own_random_numbers():
    generator = np.random.default_rng()

    def noisy_model(given):
        return {"y": given["a"] * generator.random()}

    with pytest.raises(ValueError, match="or it uses random numbers of its own"):
        run_linear(model=noisy_model)


def test_determinism_check_catches_failing_second_run():
    calls = []

    def failing_second_run(given):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("only once")
        return {"y": given["a"]}

    with pytest.raises(ValueError, match="second run of Monte Carlo sample 0") as info:
        run_linear(model=failing_second_run)
    assert isinstance(info.value.__cause__, RuntimeError)


def test_outputs_changing_between_samples_raise():
    calls = []

    def growing_model(given):
        calls.append(1)
        outputs = {"y": given["a"]}
        if len(calls) > 2:
            outputs["z"] = 1.0
        return outputs

    with pytest.raises(ValueError, match=r"sample 1 returned the outputs \['y', 'z'\]"):
        run_linear(model=growing_model)

    calls.clear()

    def slicing_model(given):
        calls.append(1)
        return {"y": given["a"] if len(calls) <= 2 else given["a"][{"r": "north"}]}

    with pytest.raises(ValueError, match="Output 'y' of Monte Carlo sample 1 has the dimensions"):
        run_linear(model=slicing_model)


@pytest.mark.parametrize(
    ("returned", "message"),
    [
        (lambda given: [given["a"]], "must return a mapping"),
        (lambda given: {"y": given["a"].values}, "Output 'y' of the model is a ndarray"),
        (lambda given: {1: given["a"]}, "Output names must be strings"),
    ],
)
def test_invalid_model_outputs_raise(returned, message):
    with pytest.raises(TypeError, match=message):
        run_linear(model=returned)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"model": get_example_mfa()}, r"use flodym.make_model_function\(mfa\)"),
        ({"parameters": get_example_mfa()}, r"Pass the parameters of the MFA system \(mfa"),
        ({"uncertainty": {"a": fd.NormalDistribution(std=1.0)}}, "Use fd.UncertaintyDefinition"),
    ],
)
def test_arguments_of_wrong_type_raise(arguments, message):
    arguments = {
        "model": linear_model,
        "parameters": linear_parameters(),
        "uncertainty": LINEAR_UNCERTAINTY,
        **arguments,
    }
    with pytest.raises(TypeError, match=message):
        fd.run_monte_carlo(**arguments, n_samples=2)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"on_error": "ignore"}, "on_error"), ({"max_memory_gib": 0.0}, "must be positive")],
)
def test_invalid_options_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        run_linear(**kwargs)


def test_memory_error_lists_the_largest_outputs():
    def model(given):
        return {"small": given["a"].sum_to(("t",)), "big": given["a"], "number": 1.0}

    with pytest.raises(
        ValueError, match=r"Largest outputs: 'big' .*, 'small' .*, 'number'"
    ) as info:
        run_linear(model=model, n_samples=1000, max_memory_gib=1e-5)
    assert "restrict track or add derived outputs" in str(info.value)


def test_no_variation_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="flodym"):
        run_linear(model=lambda given: {"c": given["c"]})
    assert "Check that the model uses the uncertain parameters" in caplog.text


def test_no_variation_with_an_empty_definition(caplog):
    empty = fd.UncertaintyDefinition(parameters={})
    with caplog.at_level(logging.WARNING, logger="flodym"):
        fd.run_monte_carlo(linear_model, linear_parameters(), empty, n_samples=2, seed=1)
    assert "has no uncertain parameters, so all Monte Carlo samples are the same" in caplog.text


def test_no_variation_can_be_an_error():
    with (
        fd.config.override(error_behaviors={"monte_carlo_no_variation": "error"}),
        pytest.raises(ValueError, match="monte_carlo_no_variation"),
    ):
        run_linear(model=lambda given: {"c": given["c"]})


# --- failing samples ------------------------------------------------------------------------------


def failing_above_nominal(given):
    if given["a"].values[0, 0] > 10.0:
        raise KeyError("a is too large")
    return {"y": given["a"]}


def test_skip_records_failed_samples_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="flodym"):
        result = run_linear(model=failing_above_nominal, n_samples=40, on_error="skip")
    inputs = LINEAR_UNCERTAINTY.sample(linear_parameters(), n_samples=40, seed=1)
    too_large = [i for i in range(40) if inputs["a"].samples.values[0, 0, i] > 10.0]
    assert sorted(result.failed_samples) == too_large
    assert set(result.failed_samples.values()) == {"KeyError: 'a is too large'"}
    assert result.rejection_rate == len(too_large) / 40
    assert result.sample_dim.items == [i for i in range(40) if i not in too_large]
    assert result.outputs["y"].n_samples == 40 - len(too_large)
    assert "rejection rate" in caplog.text
    assert "differs from the declared one" in caplog.text


def test_rejection_can_be_an_error():
    with (
        fd.config.override(error_behaviors={"monte_carlo_rejection": "error"}),
        pytest.raises(ValueError, match="rejection rate"),
    ):
        run_linear(model=failing_above_nominal, n_samples=40, on_error="skip")


def test_skip_raises_the_first_error_if_all_samples_fail(caplog):
    def always_failing(given):
        raise KeyError("missing")

    with caplog.at_level(logging.ERROR, logger="flodym"), pytest.raises(KeyError, match="missing"):
        run_linear(model=always_failing, n_samples=4, on_error="skip")
    assert "All 4 Monte Carlo samples failed and were skipped (seed=1)" in caplog.text


def test_raise_keeps_the_exception_and_adds_the_sample_index(caplog):
    inputs = LINEAR_UNCERTAINTY.sample(linear_parameters(), n_samples=40, seed=1)
    first_failing = next(i for i in range(40) if inputs["a"].samples.values[0, 0, i] > 10.0)
    with caplog.at_level(logging.ERROR, logger="flodym"), pytest.raises(KeyError):
        run_linear(model=failing_above_nominal, n_samples=40)
    assert f"Monte Carlo sample {first_failing} failed (seed=1)" in caplog.text


def test_seed_is_drawn_and_stored_if_not_given():
    first = fd.run_monte_carlo(linear_model, linear_parameters(), LINEAR_UNCERTAINTY, n_samples=3)
    again = run_linear(seed=first.seed)
    assert isinstance(first.seed, int) and first.seed >= 0
    np.testing.assert_array_equal(
        first.outputs["y"].samples.values, again.outputs["y"].samples.values
    )


# --- sample_indices and concat --------------------------------------------------------------------


def test_parts_joined_by_concat_equal_a_single_run():
    mfa = get_example_mfa()
    model = fd.make_model_function(mfa)
    single = fd.run_monte_carlo(model, mfa.parameters, EXAMPLE_UNCERTAINTY, n_samples=150, seed=42)
    parts = (
        fd.run_monte_carlo(
            model, mfa.parameters, EXAMPLE_UNCERTAINTY, sample_indices=indices, seed=42
        )
        for indices in (range(70, 150), [5, 3, 0, 1, 2, 4], range(6, 70))
    )
    joined = fd.MonteCarloResult.concat(parts)
    assert joined.sample_indices == list(range(150))
    assert joined.sample_dim.items == list(range(150))
    assert list(joined.outputs) == list(single.outputs)
    for name, output in single.outputs.items():
        np.testing.assert_array_equal(joined.outputs[name].samples.values, output.samples.values)
    assert joined.definition_fingerprint == single.definition_fingerprint
    assert joined.parameters_fingerprint == single.parameters_fingerprint


def test_concat_keeps_failed_samples():
    parts = [
        run_linear(model=failing_above_nominal, sample_indices=indices, on_error="skip")
        for indices in (range(20, 40), range(20))
    ]
    single = run_linear(model=failing_above_nominal, n_samples=40, on_error="skip")
    joined = fd.MonteCarloResult.concat(parts)
    assert joined.failed_samples == single.failed_samples
    assert joined.rejection_rate == single.rejection_rate
    assert joined.sample_dim.items == single.sample_dim.items
    np.testing.assert_array_equal(
        joined.outputs["y"].samples.values, single.outputs["y"].samples.values
    )


def test_concat_after_dumping_and_validating_the_results():
    parts = [run_linear(sample_indices=indices) for indices in (range(3), range(3, 6))]
    restored = [fd.MonteCarloResult.model_validate(part.model_dump()) for part in parts]
    joined = fd.MonteCarloResult.concat(restored)
    np.testing.assert_array_equal(
        joined.outputs["y"].samples.values, run_linear(n_samples=6).outputs["y"].samples.values
    )


def test_concat_of_mismatched_runs_raises():
    part = run_linear(sample_indices=range(3))
    with pytest.raises(ValueError, match="Result 1 has the seed 2.*Run all parts with the same"):
        fd.MonteCarloResult.concat([part, run_linear(sample_indices=range(3, 6), seed=2)])
    other = fd.UncertaintyDefinition(parameters={"a": fd.NormalDistribution(std=2.0)})
    other_result = fd.run_monte_carlo(
        linear_model, linear_parameters(), other, sample_indices=range(3, 6), seed=1
    )
    with pytest.raises(ValueError, match="another uncertainty definition than result 0"):
        fd.MonteCarloResult.concat([part, other_result])
    with pytest.raises(ValueError, match=r"Sample indices \[2\] are requested in several results"):
        fd.MonteCarloResult.concat([part, run_linear(sample_indices=range(2, 6))])
    with pytest.raises(ValueError, match="Run all parts with the same model"):
        fd.MonteCarloResult.concat(
            [part, run_linear(model=lambda given: {"y": given["a"]}, sample_indices=range(3, 6))]
        )
    with pytest.raises(ValueError, match="No results"):
        fd.MonteCarloResult.concat([])


def test_concat_with_other_nominal_values_raises():
    parameters = linear_parameters()
    parameters["c"].values[0, 0] = 2.0
    other = fd.run_monte_carlo(
        linear_model, parameters, LINEAR_UNCERTAINTY, sample_indices=[5], seed=1
    )
    with pytest.raises(ValueError, match="other nominal parameter values than result 0"):
        fd.MonteCarloResult.concat([run_linear(sample_indices=range(3)), other])


def test_concat_detects_long_slices_differing_in_one_item():
    time = fd.Dimension(name="Time", letter="t", items=list(range(3000)))
    parameters = {"p": fd.Parameter(name="p", dims=time, values=np.ones(3000))}
    selection = list(range(2000))
    other_selection = selection.copy()
    other_selection[1000] = 2500

    def run(items, indices):
        entry = fd.ParameterUncertainty(
            distribution=fd.NormalDistribution(relative_std=0.1), where={"t": items}
        )
        uncertainty = fd.UncertaintyDefinition(parameters={"p": entry})
        return fd.run_monte_carlo(
            lambda given: {"p": given["p"]}, parameters, uncertainty, sample_indices=indices, seed=1
        )

    with pytest.raises(ValueError, match="another uncertainty definition"):
        fd.MonteCarloResult.concat([run(selection, [0, 1]), run(other_selection, [2, 3])])
