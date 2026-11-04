"""Tests for the automatic computation of processes, see Process.compute()."""

import logging

import numpy as np
import pytest

from flodym import (
    DataReader,
    Dimension,
    DimensionDefinition,
    DimensionSet,
    DynamicStockModel,
    FixedLifetime,
    FlowDefinition,
    InflowDrivenDSM,
    MFADefinition,
    MFASystem,
    Parameter,
    ParameterDefinition,
    ProcessDefinition,
    SimpleFlowDrivenStock,
    StockDefinition,
    StockDrivenDSM,
    UnderdeterminedError,
    config,
)

TIME = [2000, 2001, 2002, 2003]
REGIONS = ["north", "south"]
DIM_ITEMS = {"Time": TIME, "Region": REGIONS}
DIM_DEFINITIONS = [
    DimensionDefinition(name="Time", letter="t", dtype=int),
    DimensionDefinition(name="Region", letter="r", dtype=str),
]


class DictReader(DataReader):
    """Reads dimension items and parameter values from dictionaries."""

    def __init__(self, parameter_values: dict[str, np.ndarray]):
        self.parameter_values = parameter_values

    def read_dimension(self, definition: DimensionDefinition) -> Dimension:
        return Dimension(
            name=definition.name, letter=definition.letter, items=DIM_ITEMS[definition.name]
        )

    def read_parameter_values(self, parameter_name: str, dims: DimensionSet) -> Parameter:
        values = np.broadcast_to(self.parameter_values[parameter_name], dims.shape).copy()
        return Parameter(dims=dims, values=values, name=parameter_name)


def make_mfa(
    processes: list,
    flows: list[FlowDefinition],
    parameter_values: dict[str, np.ndarray],
    parameter_dims: dict[str, tuple[str, ...]],
    stocks: list[StockDefinition] = [],
) -> MFASystem:
    definition = MFADefinition(
        dimensions=DIM_DEFINITIONS,
        processes=processes,
        flows=flows,
        stocks=stocks,
        parameters=[
            ParameterDefinition(name=name, dim_letters=dims)
            for name, dims in parameter_dims.items()
        ],
    )
    return MFASystem.from_data_reader(definition, DictReader(parameter_values))


def flow(from_process: str, to_process: str, dims=("t",), **kwargs) -> FlowDefinition:
    return FlowDefinition(
        from_process_name=from_process, to_process_name=to_process, dim_letters=dims, **kwargs
    )


def split_mfa(**process_kwargs) -> MFASystem:
    """sysenv -> a -> (b, c), b -> sysenv, c -> sysenv, with a share of a -> b of 0.3."""
    return make_mfa(
        processes=[
            "sysenv",
            ProcessDefinition(name="a", outflow_shares={"b": "share b"}, **process_kwargs),
            "b",
            "c",
        ],
        flows=[
            flow("sysenv", "a", parameter="input"),
            flow("a", "b"),
            flow("a", "c"),
            flow("b", "sysenv"),
            flow("c", "sysenv"),
        ],
        parameter_values={"input": np.array([1.0, 2.0, 3.0, 4.0]), "share b": 0.3},
        parameter_dims={"input": ("t",), "share b": ()},
    )


def test_forward_computation():
    mfa = split_mfa()
    assert mfa.flows["sysenv => a"].is_set
    assert mfa.processes["a"].compute()
    np.testing.assert_allclose(mfa.flows["a => b"].values, [0.3, 0.6, 0.9, 1.2])
    np.testing.assert_allclose(mfa.flows["a => c"].values, [0.7, 1.4, 2.1, 2.8])
    assert mfa.processes["a"].is_computed


def test_compute_all_possible():
    mfa = split_mfa()
    assert mfa.compute_all_possible() == []
    np.testing.assert_allclose(mfa.flows["c => sysenv"].values, [0.7, 1.4, 2.1, 2.8])
    assert mfa.check_mass_balance()


def test_backward_computation_and_total_from_remainder():
    mfa = split_mfa()
    mfa.flows["sysenv => a"].mark_unset()
    mfa.flows["a => c"][...] = 7.0
    mfa.processes["a"].compute()
    np.testing.assert_allclose(mfa.flows["sysenv => a"].values, 10.0)
    np.testing.assert_allclose(mfa.flows["a => b"].values, 3.0)


def test_total_from_known_flow_with_share():
    mfa = split_mfa()
    mfa.flows["sysenv => a"].mark_unset()
    mfa.flows["a => b"][...] = 3.0
    mfa.processes["a"].compute()
    np.testing.assert_allclose(mfa.flows["sysenv => a"].values, 10.0)
    np.testing.assert_allclose(mfa.flows["a => c"].values, 7.0)


def test_underdetermined():
    mfa = split_mfa()
    mfa.flows["sysenv => a"].mark_unset()
    with pytest.raises(UnderdeterminedError, match="no inflow is known"):
        mfa.processes["a"].compute()
    assert not mfa.processes["a"].compute(on_underdetermined="ignore")
    assert not mfa.flows["a => b"].is_set
    with config.override(error_behaviors={"process_underdetermined": "ignore"}):
        assert not mfa.processes["a"].compute()
    # compute_all_possible() sets unset flows from their parameters first
    assert mfa.compute_all_possible() == []
    mfa.mark_all_unset()
    mfa.flow_parameters = {}
    assert mfa.compute_all_possible() == ["a", "b", "c"]


def test_backwards_compatible_definition_with_names():
    mfa = make_mfa(
        processes=["sysenv", "a"],
        flows=[flow("sysenv", "a"), flow("a", "sysenv")],
        parameter_values={},
        parameter_dims={},
    )
    assert {p.name: p.id for p in mfa.processes.values()} == {"sysenv": 0, "a": 1}
    mfa.flows["sysenv => a"][...] = 2.0
    mfa.processes["a"].compute()
    np.testing.assert_allclose(mfa.flows["a => sysenv"].values, 2.0)


def test_undefined_parameter_in_definition():
    with pytest.raises(ValueError, match="'missing' used in definition of process 'a'"):
        MFADefinition(
            dimensions=DIM_DEFINITIONS,
            processes=["sysenv", ProcessDefinition(name="a", outflow_shares={"b": "missing"})],
            flows=[],
        )


def test_share_keys_and_parallel_flows():
    processes = ["sysenv", "a", "b"]
    flows = [
        flow("sysenv", "a"),
        flow("a", "b", name_override="first"),
        flow("a", "b", name_override="second"),
        flow("b", "sysenv"),
    ]
    mfa = make_mfa(processes, flows, {"s": 0.25}, {"s": ()})
    a = mfa.processes["a"]
    with pytest.raises(ValueError, match="Several outflows"):
        a.set_outflow_share("b", mfa.parameters["s"])
    a.set_outflow_share("first", mfa.parameters["s"])
    mfa.flows["sysenv => a"][...] = 4.0
    a.compute()
    np.testing.assert_allclose(mfa.flows["first"].values, 1.0)
    np.testing.assert_allclose(mfa.flows["second"].values, 3.0)


def test_share_parameter_changes_are_used():
    mfa = split_mfa()
    mfa.parameters["share b"][...] = 0.5
    mfa.compute_all_possible()
    np.testing.assert_allclose(mfa.flows["a => b"].values, mfa.flows["a => c"].values)


def test_invalid_shares():
    mfa = split_mfa()
    mfa.parameters["share b"][...] = 1.5
    with config.override(error_behaviors={"process_shares": "error"}):
        with pytest.raises(ValueError, match="not between zero and one"):
            mfa.processes["a"].compute()


def test_shares_of_all_flows_must_sum_to_one():
    mfa = make_mfa(
        processes=[
            "sysenv",
            ProcessDefinition(name="a", outflow_shares={"b": "s", "c": "s"}),
            "b",
            "c",
        ],
        flows=[flow("sysenv", "a"), flow("a", "b"), flow("a", "c")],
        parameter_values={"s": 0.4},
        parameter_dims={"s": ()},
    )
    mfa.flows["sysenv => a"][...] = 1.0
    with config.override(error_behaviors={"process_shares": "error"}):
        with pytest.raises(ValueError, match="do not sum to one"):
            mfa.processes["a"].compute()


def test_overdetermined_inconsistent_raises():
    mfa = split_mfa()
    mfa.flows["a => b"][...] = 100.0
    assert mfa.processes["a"].is_overdetermined
    with config.override(error_behaviors={"process_shares": "error"}):
        with pytest.raises(ValueError, match="overdetermined"):
            mfa.processes["a"].compute()


def test_dimension_splitter():
    mfa = make_mfa(
        processes=["sysenv", ProcessDefinition(name="a", dimension_splitter="split"), "b"],
        flows=[flow("sysenv", "a", parameter="input"), flow("a", "b", dims=("t", "r"))],
        parameter_values={"input": np.array([1.0, 2.0, 3.0, 4.0]), "split": np.array([0.2, 0.8])},
        parameter_dims={"input": ("t",), "split": ("r",)},
    )
    mfa.processes["a"].compute()
    np.testing.assert_allclose(
        mfa.flows["a => b"].values, np.outer([1.0, 2.0, 3.0, 4.0], [0.2, 0.8])
    )

    # backward: no splitter needed, unused splitter is reported
    mfa.mark_all_unset()
    mfa.flows["a => b"][...] = 1.0
    with config.override(error_behaviors={"unused_dimension_splitter": "error"}):
        with pytest.raises(ValueError, match="not used"):
            mfa.processes["a"].compute()
    np.testing.assert_allclose(mfa.flows["sysenv => a"].values, 2.0)


def test_dimension_splitter_not_summing_to_one(caplog):
    mfa = make_mfa(
        processes=["sysenv", ProcessDefinition(name="a", dimension_splitter="split"), "b"],
        flows=[flow("sysenv", "a", parameter="input"), flow("a", "b", dims=("t", "r"))],
        parameter_values={"input": 1.0, "split": np.array([0.2, 0.7])},
        parameter_dims={"input": ("t",), "split": ("r",)},
    )
    with caplog.at_level(logging.WARNING):
        mfa.processes["a"].compute()
    assert "does not sum to one" in caplog.text


def test_missing_dimension_splitter():
    mfa = make_mfa(
        processes=["sysenv", "a", "b"],
        flows=[flow("sysenv", "a", parameter="input"), flow("a", "b", dims=("t", "r"))],
        parameter_values={"input": 1.0},
        parameter_dims={"input": ("t",)},
    )
    with pytest.raises(ValueError, match="Set a dimension splitter"):
        mfa.processes["a"].compute()


def test_flow_parameters_reset_after_mark_all_unset():
    mfa = split_mfa()
    mfa.compute_all_possible()
    mfa.parameters["input"][...] = 10.0
    mfa.mark_all_unset()
    mfa.compute_all_possible()
    np.testing.assert_allclose(mfa.flows["a => b"].values, 3.0)


def stock_mfa(subclass: type, lifetime: bool = True) -> MFASystem:
    """sysenv -> use (with stock) -> (eol, sysenv), with a share of use -> eol of 0.6."""
    stock_kwargs = {"lifetime_model_class": FixedLifetime} if lifetime else {}
    mfa = make_mfa(
        processes=["sysenv", ProcessDefinition(name="use", outflow_shares={"eol": "share"}), "eol"],
        flows=[
            flow("sysenv", "use"),
            flow("use", "eol"),
            flow("use", "sysenv"),
            flow("eol", "sysenv"),
        ],
        parameter_values={"share": 0.6},
        parameter_dims={"share": ()},
        stocks=[
            StockDefinition(
                name="use", process="use", dim_letters=("t",), subclass=subclass, **stock_kwargs
            )
        ],
    )
    if lifetime:
        mfa.stocks["use"].lifetime_model.mean = 1
    return mfa


def test_simple_stock_process():
    mfa = stock_mfa(SimpleFlowDrivenStock, lifetime=False)
    mfa.flows["sysenv => use"][...] = 2.0
    mfa.flows["use => sysenv"][...] = 0.4
    assert mfa.compute_all_possible() == []
    np.testing.assert_allclose(mfa.flows["use => eol"].values, 0.6)
    np.testing.assert_allclose(mfa.stocks["use"].stock.values, [1.0, 2.0, 3.0, 4.0])
    assert mfa.check_mass_balance()


@pytest.mark.parametrize("subclass", [InflowDrivenDSM, DynamicStockModel])
def test_inflow_driven_stock_process(subclass):
    mfa = stock_mfa(subclass)
    mfa.flows["sysenv => use"][...] = 1.0
    assert mfa.compute_all_possible() == []
    outflow = mfa.stocks["use"].outflow.values
    np.testing.assert_allclose(outflow, [0.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(mfa.flows["use => eol"].values, 0.6 * outflow)
    np.testing.assert_allclose(mfa.flows["use => sysenv"].values, 0.4 * outflow)
    assert mfa.check_mass_balance()


@pytest.mark.parametrize("subclass", [StockDrivenDSM, DynamicStockModel])
def test_stock_driven_stock_process(subclass):
    mfa = stock_mfa(subclass)
    mfa.stocks["use"].stock[...] = 1.0
    assert mfa.compute_all_possible() == []
    np.testing.assert_allclose(mfa.flows["sysenv => use"].values, [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(mfa.flows["use => eol"].values, [0.0, 0.6, 0.6, 0.6])
    assert mfa.check_mass_balance()


def test_stock_driven_without_stock_is_underdetermined():
    mfa = stock_mfa(StockDrivenDSM)
    mfa.flows["sysenv => use"][...] = 1.0
    with pytest.raises(UnderdeterminedError, match="stock-driven"):
        mfa.processes["use"].compute()
    assert not mfa.stocks["use"].inflow.is_set


def test_stock_process_underdetermined_changes_nothing():
    mfa = stock_mfa(InflowDrivenDSM)
    mfa.processes["use"].remove_outflow_share("eol")
    mfa.flows["sysenv => use"][...] = 1.0
    with pytest.raises(UnderdeterminedError, match="neither share nor value"):
        mfa.processes["use"].compute()
    assert not mfa.stocks["use"].inflow.is_set
    assert not mfa.stocks["use"].stock.is_set


def chain_mfa(n_processes: int) -> MFASystem:
    """sysenv -> p0 -> p1 -> ... -> p(n-1) -> sysenv, with processes defined in reverse order."""
    names = [f"p{i}" for i in range(n_processes)]
    links = ["sysenv"] + names + ["sysenv"]
    flows = [flow(a, b) for a, b in zip(links[:-1], links[1:])]
    return make_mfa(["sysenv"] + names[::-1], flows, {}, {})


def test_long_chain_without_recursion_limit():
    mfa = chain_mfa(3000)
    mfa.flows["sysenv => p0"][...] = 1.0
    assert mfa.compute_all_possible() == []
    np.testing.assert_allclose(mfa.flows["p2999 => sysenv"].values, 1.0)


def test_compute_neighbors():
    mfa = chain_mfa(3)
    mfa.flows["sysenv => p0"][...] = 1.0
    assert mfa.processes["p0"].compute()
    assert not mfa.processes["p1"].is_computed
    assert mfa.processes["p1"].compute(compute_neighbors=True)
    assert all(p.is_computed for name, p in mfa.processes.items() if name != "sysenv")
