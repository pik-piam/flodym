"""Tests of make_model_function, the model function of an MFA system for run_monte_carlo."""

import logging
import pickle

import numpy as np
import pytest

import flodym as fd
from flodym.example_objects import ExampleMFA, get_example_mfa

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


def copper_content(mfa: fd.MFASystem) -> fd.FlodymArray:
    secondary_steel = mfa.flows["remelting => sysenv"]
    return secondary_steel["Cu"] / secondary_steel.sum_over(("e",))


def example_as(mfa_class: type[ExampleMFA]) -> ExampleMFA:
    """The ExampleMFA as an instance of a subclass."""
    example = get_example_mfa()
    return mfa_class(
        dims=example.dims,
        parameters=example.parameters,
        processes=example.processes,
        flows=example.flows,
        stocks=example.stocks,
    )


def run(mfa: fd.MFASystem, n_samples: int = 5, **kwargs) -> fd.MonteCarloResult:
    model = fd.make_model_function(mfa, **kwargs)
    return fd.run_monte_carlo(
        model, mfa.parameters, EXAMPLE_UNCERTAINTY, n_samples=n_samples, seed=7
    )


def mfa_state(mfa: fd.MFASystem) -> dict:
    arrays = {f"flow {name}": flow for name, flow in mfa.flows.items()}
    for name, stock in mfa.stocks.items():
        for array_name in ("stock", "inflow", "outflow"):
            arrays[f"{name}.{array_name}"] = getattr(stock, array_name)
    state = {key: (array.values.copy(), array.is_set) for key, array in arrays.items()}
    state["parameters"] = {name: p.values.copy() for name, p in mfa.parameters.items()}
    state["parameter objects"] = list(mfa.parameters.values())
    state["parameter values objects"] = [p.values for p in mfa.parameters.values()]
    state["parameters dict"] = [mfa.parameters]
    return state


def assert_same_state(state: dict, other: dict) -> None:
    assert state.keys() == other.keys()
    for key, value in state.items():
        if key in ("parameter objects", "parameter values objects", "parameters dict"):
            assert all(a is b for a, b in zip(value, other[key])), key
        elif key == "parameters":
            for name, values in value.items():
                np.testing.assert_array_equal(values, other[key][name])
                assert values.dtype == other[key][name].dtype
        else:
            np.testing.assert_array_equal(value[0], other[key][0])
            assert value[1] == other[key][1], key


def test_adapter_equals_a_hand_written_loop():
    mfa = get_example_mfa()
    result = run(mfa, n_samples=8)
    inputs = EXAMPLE_UNCERTAINTY.sample(mfa.parameters, n_samples=8, seed=7)
    expected_keys = list(mfa.flows) + [
        f"{name}.{array}" for name in mfa.stocks for array in ("stock", "inflow", "outflow")
    ]
    assert list(result.outputs) == expected_keys
    for sample_index in range(8):
        reference = get_example_mfa()
        for name, sampled in inputs.items():
            reference.parameters[name].values = sampled.samples.values[..., sample_index]
        reference.compute()
        for name, flow in reference.flows.items():
            np.testing.assert_array_equal(
                result.outputs[name].samples.values[..., sample_index], flow.values
            )
        np.testing.assert_array_equal(
            result.outputs["landfills.stock"].samples.values[..., sample_index],
            reference.stocks["landfills"].stock.values,
        )
    assert result.outputs["sysenv => shredder"].mean.extensive is True


def test_adapter_restores_the_system_including_integer_parameters():
    mfa = get_example_mfa()
    # read as integers, e.g. from a file without decimal places
    assert mfa.parameters["eol buildings"].values.dtype.kind == "i"
    before = mfa_state(mfa)
    result = run(mfa)
    assert_same_state(before, mfa_state(mfa))
    # the sampled values were not rounded
    buildings = result.outputs["sysenv => demolition"].samples.values[:, 0]
    assert not np.allclose(buildings, np.round(buildings))


def test_adapter_restores_the_system_after_an_error():
    class FailingMFA(ExampleMFA):
        def compute(self):
            super().compute()
            raise ArithmeticError("compute failed")

    mfa = example_as(FailingMFA)
    before = mfa_state(mfa)
    with pytest.raises(ArithmeticError, match="compute failed"):
        run(mfa)
    assert_same_state(before, mfa_state(mfa))


def test_adapter_on_a_system_computed_before():
    fresh_result = run(get_example_mfa(), n_samples=4)
    mfa = get_example_mfa()
    mfa.compute()
    before = mfa_state(mfa)
    result = run(mfa, n_samples=4)
    for name, output in fresh_result.outputs.items():
        np.testing.assert_array_equal(result.outputs[name].samples.values, output.samples.values)
    assert_same_state(before, mfa_state(mfa))
    assert all(flow.is_set for flow in mfa.flows.values())


class InPlaceMFA(ExampleMFA):
    def compute(self):
        self.parameters["eol machines"].values *= 1.5
        super().compute()


class RebindingEntryMFA(ExampleMFA):
    def compute(self):
        self.parameters["eol machines"] *= 1.5
        super().compute()


class RebindingDictMFA(ExampleMFA):
    def compute(self):
        machines = self.parameters["eol machines"] * 1.5
        self.parameters = fd.AttributeDict({**self.parameters, "eol machines": machines})
        super().compute()


@pytest.mark.parametrize("mfa_class", [InPlaceMFA, RebindingEntryMFA, RebindingDictMFA])
def test_parameter_changes_in_compute_do_not_accumulate(mfa_class):
    mfa = example_as(mfa_class)
    before = mfa_state(mfa)
    result = run(mfa, track=["sysenv => shredder"])
    inputs = EXAMPLE_UNCERTAINTY.sample(mfa.parameters, n_samples=5, seed=7)
    composition = inputs["composition eol machines"].samples.values
    expected = 1.5 * inputs["eol machines"].samples.values[:, None, :] * composition[None, :, :]
    np.testing.assert_allclose(
        result.outputs["sysenv => shredder"].samples.values, expected, rtol=1e-14
    )
    assert_same_state(before, mfa_state(mfa))


def test_determinism_check_catches_state_of_the_system():
    class CountingMFA(ExampleMFA):
        n_runs: int = 0

        def compute(self):
            self.n_runs += 1
            super().compute()
            self.flows["remelting => sysenv"][...] = self.flows["remelting => sysenv"] * self.n_runs

    with pytest.raises(ValueError, match="'remelting => sysenv' \\(largest difference"):
        run(example_as(CountingMFA))


def compute_twice(mfa: fd.MFASystem) -> None:
    mfa.compute()


def total_steel(mfa: fd.MFASystem) -> float:
    return float(mfa.flows["remelting => sysenv"].sum_values())


def test_track_derived_and_compute():
    mfa = get_example_mfa()
    calls = []

    def compute(system: fd.MFASystem) -> None:
        calls.append(1)
        system.compute()

    result = run(
        mfa,
        n_samples=6,
        track=["remelting => sysenv", "landfills", "slag piles.stock"],
        derived={"copper content": copper_content, "total": total_steel},
        compute=compute,
    )
    assert list(result.outputs) == [
        "remelting => sysenv",
        "landfills.stock",
        "landfills.inflow",
        "landfills.outflow",
        "slag piles.stock",
        "copper content",
        "total",
    ]
    assert len(calls) == 7  # the first sample is run twice
    steel = result.outputs["remelting => sysenv"].samples.values
    np.testing.assert_allclose(
        result.outputs["copper content"].samples.values, steel[:, 1] / steel.sum(axis=1)
    )
    np.testing.assert_allclose(result.outputs["total"].samples.values, steel.sum(axis=(0, 1)))
    content = result.outputs["copper content"].quantile([0.05, 0.95])
    assert content.dims.letters == ("t", "q")
    assert np.all(content.values[:, 0] < content.values[:, 1])


def test_empty_track_with_derived_quantities():
    result = run(get_example_mfa(), track=[], derived={"total": total_steel})
    assert list(result.outputs) == ["total"]


def test_adapter_argument_errors():
    mfa = get_example_mfa()
    with pytest.raises(ValueError, match=r"\['nope', 'landfill', 'landfills.mass'\] in track"):
        fd.make_model_function(mfa, track=["nope", "landfills", "landfill", "landfills.mass"])
    with pytest.raises(
        ValueError, match=r"Derived quantities \['landfills.stock'\] have the names"
    ):
        fd.make_model_function(mfa, derived={"landfills.stock": copper_content})
    model = fd.make_model_function(mfa)
    parameters = dict(mfa.parameters)
    parameters["extra"] = parameters.pop("eol machines")
    with pytest.raises(
        ValueError, match=r"Missing parameters: \['eol machines'\]. Unknown .*extra"
    ):
        model(parameters)
    parameters = dict(mfa.parameters)
    parameters["eol machines"] = parameters["shredder yield"]
    with pytest.raises(ValueError, match="'eol machines' given to the model function has other"):
        model(parameters)


def test_read_only_values_are_replaced_and_restored():
    mfa = get_example_mfa()
    # read-only values, as numpy.broadcast_to() or pandas give them
    read_only_parameter = np.broadcast_to(mfa.parameters["shredder yield"].values.copy(), (3,))
    mfa.parameters["shredder yield"].values = read_only_parameter
    flow = mfa.flows["sysenv => shredder"]
    read_only_flow = np.broadcast_to(1.0, flow.shape)
    flow.values = read_only_flow
    flow.mark_set()
    before = mfa_state(mfa)
    result = run(mfa, track=["shredder => remelting"])
    inputs = EXAMPLE_UNCERTAINTY.sample(mfa.parameters, n_samples=5, seed=7)
    expected = (
        inputs["eol machines"].samples.values[:, None, :]
        * inputs["composition eol machines"].samples.values[None, :, :]
        * inputs["shredder yield"].samples.values[None, :, :]
    )
    np.testing.assert_allclose(
        result.outputs["shredder => remelting"].samples.values, expected, rtol=1e-14
    )
    assert_same_state(before, mfa_state(mfa))
    assert mfa.parameters["shredder yield"].values is read_only_parameter
    assert flow.values is read_only_flow


def test_model_function_can_be_pickled():
    mfa = get_example_mfa()
    model = fd.make_model_function(mfa, derived={"total": total_steel}, compute=compute_twice)
    restored = pickle.loads(pickle.dumps(model))
    expected = model(mfa.parameters)
    outputs = restored(restored.mfa.parameters)
    assert outputs["total"] == expected["total"]


class DictReader(fd.DataReader):
    """Reads dimension items and parameter values from dictionaries."""

    def __init__(self, parameter_values: dict[str, np.ndarray]):
        self.parameter_values = parameter_values

    def read_dimension(self, definition: fd.DimensionDefinition) -> fd.Dimension:
        return fd.Dimension(name="Time", letter="t", dtype=int, items=[2000, 2001, 2002, 2003])

    def read_parameter_values(self, parameter_name: str, dims: fd.DimensionSet) -> fd.Parameter:
        values = np.broadcast_to(self.parameter_values[parameter_name], dims.shape).copy()
        return fd.Parameter(dims=dims, values=values, name=parameter_name)


def split_mfa() -> fd.MFASystem:
    """sysenv -> a -> (b, c) -> sysenv; the inflow is prescribed, a -> b has a share."""

    def flow(from_process, to_process, **kwargs):
        return fd.FlowDefinition(
            from_process_name=from_process, to_process_name=to_process, dim_letters=("t",), **kwargs
        )

    definition = fd.MFADefinition(
        dimensions=[fd.DimensionDefinition(name="Time", letter="t", dtype=int)],
        processes=[
            "sysenv",
            fd.ProcessDefinition(name="a", outflow_shares={"b": "share b"}),
            "b",
            "c",
        ],
        flows=[
            flow("sysenv", "a", prescribed="input"),
            flow("a", "b"),
            flow("a", "c"),
            flow("b", "sysenv"),
            flow("c", "sysenv"),
        ],
        stocks=[],
        parameters=[
            fd.ParameterDefinition(name="input", dim_letters=("t",)),
            fd.ParameterDefinition(name="share b", dim_letters=()),
        ],
    )
    reader = DictReader({"input": np.array([1.0, 2.0, 3.0, 4.0]), "share b": 0.3})
    return fd.MFASystem.from_data_reader(definition, reader)


def compute_all_possible(mfa: fd.MFASystem) -> None:
    mfa.compute_all_possible()


def test_adapter_with_compute_all_possible_shares_and_prescribed_flows():
    mfa = split_mfa()
    uncertainty = fd.UncertaintyDefinition(
        parameters={
            "input": fd.LogNormalDistribution(relative_std=0.1),
            "share b": fd.UniformDistribution(lower=0.2, upper=0.4),
        }
    )
    model = fd.make_model_function(mfa, compute=compute_all_possible)
    result = fd.run_monte_carlo(model, mfa.parameters, uncertainty, n_samples=10, seed=3)
    inputs = uncertainty.sample(mfa.parameters, n_samples=10, seed=3)
    total = inputs["input"].samples.values
    share = inputs["share b"].samples.values
    np.testing.assert_allclose(result.outputs["a => b"].samples.values, total * share)
    np.testing.assert_allclose(result.outputs["c => sysenv"].samples.values, total * (1 - share))
    assert not mfa.flows["a => b"].is_set
    np.testing.assert_array_equal(mfa.parameters["share b"].values, 0.3)


@pytest.mark.parametrize("behavior", ["error", "warn", "ignore"])
def test_share_check_follows_the_configured_behavior(behavior, caplog):
    mfa = split_mfa()
    uncertainty = fd.UncertaintyDefinition(
        parameters={"share b": fd.UniformDistribution(lower=0.5, upper=1.5)}
    )
    model = fd.make_model_function(mfa, compute=compute_all_possible)
    with (
        fd.config.override(error_behaviors={"process_shares": behavior}),
        caplog.at_level(logging.WARNING, logger="flodym"),
    ):
        result = fd.run_monte_carlo(
            model, mfa.parameters, uncertainty, n_samples=30, seed=3, on_error="skip"
        )
    share = uncertainty.sample(mfa.parameters, n_samples=30, seed=3)["share b"].samples.values
    above_one = [i for i in range(30) if share[i] > 1.0]
    assert above_one
    if behavior == "error":
        assert sorted(result.failed_samples) == above_one
        assert "between zero and one" in next(iter(result.failed_samples.values()))
    else:
        assert not result.failed_samples
    assert ("not between zero and one" in caplog.text) == (behavior == "warn")


# --- dynamic stocks -------------------------------------------------------------------------------

REGION = fd.Dimension(name="Region", letter="r", items=["north", "south", "east"])
STOCK_TIME = fd.Dimension(name="Time", letter="t", dtype=int, items=list(range(2000, 2031)))
STOCK_DIMS = fd.DimensionSet(dim_list=[STOCK_TIME, REGION])
COHORT = fd.Dimension(name="Cohort", letter="c", dtype=int, items=STOCK_TIME.items)


def stock_parameters() -> dict[str, fd.FlodymArray]:
    inflow = np.outer(np.linspace(1.0, 3.0, 31), [1.0, 2.0, 0.5])
    return {
        "inflow": fd.Parameter(name="inflow", dims=STOCK_DIMS, values=inflow),
        "lifetime": fd.Parameter(name="lifetime", dims=REGION, values=np.array([10.0, 15.0, 5.0])),
    }


LIFETIME_UNCERTAINTY = fd.UncertaintyDefinition(
    parameters={
        "lifetime": fd.ParameterUncertainty(
            distribution=fd.NormalDistribution(relative_std=0.15, lower=1.0),
            independent_over="r",
        )
    }
)


def stock_model(parameters: dict[str, fd.FlodymArray]) -> dict[str, fd.FlodymArray]:
    lifetime = fd.NormalLifetime(
        dims=STOCK_DIMS, mean=parameters["lifetime"], std=0.3 * parameters["lifetime"]
    )
    stock = fd.InflowDrivenDSM(
        dims=STOCK_DIMS, inflow=parameters["inflow"], lifetime_model=lifetime, cohort_dim=COHORT
    )
    stock.compute()
    return {"stock": stock.stock, "outflow": stock.outflow}


def test_stock_model_without_mfa_system():
    result = fd.run_monte_carlo(
        stock_model, stock_parameters(), LIFETIME_UNCERTAINTY, n_samples=6, seed=1
    )
    lifetimes = LIFETIME_UNCERTAINTY.sample(stock_parameters(), n_samples=6, seed=1)["lifetime"]
    for sample_index in range(6):
        parameters = stock_parameters()
        parameters["lifetime"].values = lifetimes.samples.values[:, sample_index]
        expected = stock_model(parameters)
        np.testing.assert_allclose(
            result.outputs["stock"].samples.values[..., sample_index], expected["stock"].values
        )
    # regions are drawn independently
    final_stock = result.outputs["stock"].samples.values[-1]
    relative = final_stock / final_stock.mean(axis=-1, keepdims=True)
    assert not np.allclose(relative[0], relative[1])


class StockMFA(fd.MFASystem):
    assign_lifetime: bool = True

    def compute(self) -> None:
        stock = self.stocks["use"]
        if self.assign_lifetime:
            stock.lifetime_model.mean = self.parameters["lifetime"]
            stock.lifetime_model.std = 0.3 * self.parameters["lifetime"]
        if isinstance(stock, fd.StockDrivenDSM):
            stock.stock[...] = self.parameters["stock"]
            stock.compute()
            self.flows["sysenv => use"][...] = stock.inflow
        else:
            self.flows["sysenv => use"][...] = self.parameters["inflow"]
            stock.inflow[...] = self.flows["sysenv => use"]
            stock.compute()
        self.flows["use => sysenv"][...] = stock.outflow


def make_stock_mfa(stock_class: type[fd.DynamicStockModel], assign_lifetime: bool) -> StockMFA:
    parameters = stock_parameters()
    if stock_class is fd.StockDrivenDSM:
        stock_values = np.outer(np.linspace(10.0, 30.0, 31), [1.0, 2.0, 0.5])
        parameters = {
            "stock": fd.Parameter(name="stock", dims=STOCK_DIMS, values=stock_values),
            "lifetime": parameters["lifetime"],
        }
    processes = fd.make_processes(["sysenv", "use"])
    flows = fd.make_empty_flows(
        processes,
        [
            fd.FlowDefinition(
                from_process_name="sysenv", to_process_name="use", dim_letters=("t", "r")
            ),
            fd.FlowDefinition(
                from_process_name="use", to_process_name="sysenv", dim_letters=("t", "r")
            ),
        ],
        STOCK_DIMS,
    )
    lifetime = fd.NormalLifetime(dims=STOCK_DIMS)
    if not assign_lifetime:
        lifetime.mean = parameters["lifetime"]
        lifetime.std = 0.3 * parameters["lifetime"]
    stock = stock_class(
        dims=STOCK_DIMS,
        name="use",
        process=processes["use"],
        lifetime_model=lifetime,
        cohort_dim=COHORT,
    )
    return StockMFA(
        dims=STOCK_DIMS,
        parameters=parameters,
        processes=processes,
        flows=flows,
        stocks={"use": stock},
        assign_lifetime=assign_lifetime,
    )


def run_stock_mfa(mfa: StockMFA, n_samples: int = 6) -> fd.MonteCarloResult:
    model = fd.make_model_function(mfa)
    return fd.run_monte_carlo(
        model, mfa.parameters, LIFETIME_UNCERTAINTY, n_samples=n_samples, seed=1
    )


def test_inflow_driven_stock_with_uncertain_lifetime(caplog):
    mfa = make_stock_mfa(fd.InflowDrivenDSM, assign_lifetime=True)
    before = mfa_state(mfa)
    with caplog.at_level(logging.WARNING, logger="flodym"):
        result = run_stock_mfa(mfa)
    reference = fd.run_monte_carlo(
        stock_model, stock_parameters(), LIFETIME_UNCERTAINTY, n_samples=6, seed=1
    )
    np.testing.assert_allclose(
        result.outputs["use.stock"].samples.values, reference.outputs["stock"].samples.values
    )
    np.testing.assert_allclose(
        result.outputs["use => sysenv"].samples.values,
        reference.outputs["outflow"].samples.values,
    )
    assert "lifetime model parameters it had before compute()" not in caplog.text
    assert_same_state(before, mfa_state(mfa))
    np.testing.assert_array_equal(mfa.stocks["use"].get_stock_by_cohort().values, 0.0)


def test_stock_driven_stock_with_uncertain_lifetime():
    mfa = make_stock_mfa(fd.StockDrivenDSM, assign_lifetime=True)
    before = mfa_state(mfa)
    result = run_stock_mfa(mfa)
    lifetimes = LIFETIME_UNCERTAINTY.sample(mfa.parameters, n_samples=6, seed=1)["lifetime"]
    for sample_index in (0, 5):
        lifetime = fd.NormalLifetime(dims=STOCK_DIMS)
        lifetime.mean = fd.FlodymArray(
            dims=REGION, values=lifetimes.samples.values[:, sample_index]
        )
        lifetime.std = 0.3 * lifetime.mean
        stock = fd.StockDrivenDSM(
            dims=STOCK_DIMS,
            stock=mfa.parameters["stock"],
            lifetime_model=lifetime,
            cohort_dim=COHORT,
        )
        stock.compute()
        np.testing.assert_allclose(
            result.outputs["use.inflow"].samples.values[..., sample_index], stock.inflow.values
        )
    # the stock is given, the inflow is uncertain
    np.testing.assert_allclose(result.outputs["use.stock"].std.values, 0.0, atol=1e-12)
    assert np.any(result.outputs["sysenv => use"].std.values > 0.01)
    assert_same_state(before, mfa_state(mfa))


def test_fixed_lifetime_as_error_stops_the_run():
    mfa = make_stock_mfa(fd.InflowDrivenDSM, assign_lifetime=False)
    model = fd.make_model_function(mfa)
    with (
        fd.config.override(error_behaviors={"fixed_lifetime": "error"}),
        pytest.raises(ValueError, match="fixed_lifetime"),
    ):
        fd.run_monte_carlo(
            model, mfa.parameters, LIFETIME_UNCERTAINTY, n_samples=4, seed=1, on_error="skip"
        )
    # also when the model function is used again
    with (
        fd.config.override(error_behaviors={"fixed_lifetime": "error"}),
        pytest.raises(ValueError, match="fixed_lifetime"),
    ):
        model(mfa.parameters)


def test_fixed_lifetime_can_be_ignored(caplog):
    mfa = make_stock_mfa(fd.InflowDrivenDSM, assign_lifetime=False)
    with (
        fd.config.override(error_behaviors={"fixed_lifetime": "ignore"}),
        caplog.at_level(logging.WARNING, logger="flodym"),
    ):
        run_stock_mfa(mfa, n_samples=4)
    assert "lifetime model parameters" not in caplog.text


def test_lifetime_assigned_outside_compute_is_reported_once(caplog):
    mfa = make_stock_mfa(fd.InflowDrivenDSM, assign_lifetime=False)
    with caplog.at_level(logging.WARNING, logger="flodym"):
        result = run_stock_mfa(mfa, n_samples=4)
    assert caplog.text.count("Dynamic stock 'use' was computed with the lifetime model") == 1
    # the stock does not follow the sampled lifetimes
    assert not np.any(np.std(result.outputs["use.stock"].samples.values, axis=-1) > 0)
    assert "No output differs" in caplog.text
