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
    Stock,
    StockDefinition,
    StockDrivenDSM,
    UnderdeterminedError,
    config,
)

TIME = [2000, 2001, 2002, 2003]
REGIONS = ["north", "south"]
DIM_ITEMS = {"Time": TIME, "Region": REGIONS, "Cohort": TIME}
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
    dimensions: list[DimensionDefinition] = DIM_DEFINITIONS,
) -> MFASystem:
    definition = MFADefinition(
        dimensions=dimensions,
        processes=processes,
        flows=flows,
        stocks=stocks,
        parameters=[
            ParameterDefinition(name=name, dim_letters=dims)
            for name, dims in parameter_dims.items()
        ],
    )
    mfa = MFASystem.from_data_reader(definition, DictReader(parameter_values))
    # fill prescribed flows, so that single processes can be computed without
    # compute_all_possible()
    mfa.set_arrays_from_prescribed()
    return mfa


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
            flow("sysenv", "a", prescribed="input"),
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
    # compute_all_possible() sets prescribed flows from their parameters first
    assert mfa.compute_all_possible() == []
    mfa.mark_all_unset()
    mfa.flows["sysenv => a"]._set_prescribed(None)
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


def splitter_mfa(split_values, **process_kwargs) -> MFASystem:
    """sysenv -> a -> b -> sysenv, with the outflow of a by region and the inflow not."""
    return make_mfa(
        processes=["sysenv", ProcessDefinition(name="a", **process_kwargs), "b"],
        flows=[
            flow("sysenv", "a", prescribed="input"),
            flow("a", "b", dims=("t", "r")),
            flow("b", "sysenv", dims=("t", "r")),
        ],
        parameter_values={"input": np.array([1.0, 2.0, 3.0, 4.0]), "split": split_values},
        parameter_dims={"input": ("t",), "split": ("r",)},
    )


def test_forward_dimension_splitter():
    mfa = splitter_mfa(np.array([0.2, 0.8]), forward_dimension_splitter="split")
    assert mfa.processes["a"].forward_dimension_splitter is mfa.parameters["split"]
    mfa.processes["a"].compute()
    np.testing.assert_allclose(
        mfa.flows["a => b"].values, np.outer([1.0, 2.0, 3.0, 4.0], [0.2, 0.8])
    )

    # backward: the inflow has fewer dimensions than the outflow, so no splitter is needed,
    # and the forward splitter is irrelevant
    mfa.mark_all_unset()
    mfa.flows["a => b"][...] = 1.0
    with config.override(error_behaviors={"unused_dimension_splitter": "error"}):
        mfa.processes["a"].compute()
    np.testing.assert_allclose(mfa.flows["sysenv => a"].values, 2.0)


def test_unused_dimension_splitter():
    mfa = make_mfa(
        processes=["sysenv", ProcessDefinition(name="a", forward_dimension_splitter="split"), "b"],
        flows=[flow("sysenv", "a", dims=("t", "r"), prescribed="input"), flow("a", "b")],
        parameter_values={"input": 1.0, "split": np.array([0.2, 0.8])},
        parameter_dims={"input": ("t", "r"), "split": ("r",)},
    )
    with config.override(error_behaviors={"unused_dimension_splitter": "error"}):
        with pytest.raises(ValueError, match="forward dimension splitter is set, but was not used"):
            mfa.processes["a"].compute()


def test_forward_and_backward_dimension_splitters():
    """Inflows by region, outflows by good: each direction needs its own splitter."""
    dims = {"Time": TIME, "Region": REGIONS, "Good": ["cars", "houses", "tools"]}
    region_by_good = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    forward = region_by_good / region_by_good.sum(axis=1, keepdims=True)  # sums to 1 over goods
    backward = region_by_good / region_by_good.sum(axis=0, keepdims=True)  # ... over regions
    definition = MFADefinition(
        dimensions=DIM_DEFINITIONS + [DimensionDefinition(name="Good", letter="g", dtype=str)],
        processes=[
            "sysenv",
            ProcessDefinition(
                name="a", forward_dimension_splitter="fwd", backward_dimension_splitter="bwd"
            ),
        ],
        flows=[flow("sysenv", "a", dims=("t", "r")), flow("a", "sysenv", dims=("t", "g"))],
        parameters=[
            ParameterDefinition(name="fwd", dim_letters=("r", "g")),
            ParameterDefinition(name="bwd", dim_letters=("r", "g")),
        ],
    )

    class Reader(DictReader):
        def read_dimension(self, definition):
            return Dimension(
                name=definition.name, letter=definition.letter, items=dims[definition.name]
            )

    mfa = MFASystem.from_data_reader(definition, Reader({"fwd": forward, "bwd": backward}))
    a = mfa.processes["a"]
    inflow, outflow = mfa.flows["sysenv => a"], mfa.flows["a => sysenv"]

    inflow[...] = np.array([[1.0, 2.0]] * 4)
    a.compute()
    np.testing.assert_allclose(outflow.values[0], forward.T @ [1.0, 2.0])

    mfa.mark_all_unset()
    outflow[...] = np.array([[1.0, 2.0, 3.0]] * 4)
    a.compute()
    np.testing.assert_allclose(inflow.values[0], backward @ [1.0, 2.0, 3.0])
    assert mfa.check_mass_balance()


def test_dimension_splitter_not_summing_to_one(caplog):
    mfa = splitter_mfa(np.array([0.2, 0.7]), forward_dimension_splitter="split")
    with caplog.at_level(logging.WARNING):
        mfa.processes["a"].compute()
    assert "forward dimension splitter does not sum to one" in caplog.text


def test_missing_dimension_splitter():
    mfa = splitter_mfa(np.array([0.2, 0.8]), backward_dimension_splitter="split")
    with pytest.raises(ValueError, match="Set a forward dimension splitter"):
        mfa.processes["a"].compute()
    with pytest.raises(ValueError, match="'forward' or 'backward'"):
        mfa.processes["a"].set_dimension_splitter("sideways", mfa.parameters["split"])


def test_prescribed_flows_refilled_after_mark_all_unset():
    mfa = split_mfa()
    mfa.compute_all_possible()
    mfa.parameters["input"][...] = 10.0
    mfa.mark_all_unset()
    mfa.compute_all_possible()
    np.testing.assert_allclose(mfa.flows["a => b"].values, 3.0)


def stock_mfa(
    subclass: type,
    lifetime: bool = True,
    parameter_values: dict = {},
    parameter_dims: dict = {},
    **stock_kwargs,
) -> MFASystem:
    """sysenv -> use (with stock) -> (eol, sysenv), with a share of use -> eol of 0.6."""
    if lifetime:
        stock_kwargs["lifetime_model_class"] = FixedLifetime
    mfa = make_mfa(
        processes=["sysenv", ProcessDefinition(name="use", outflow_shares={"eol": "share"}), "eol"],
        flows=[
            flow("sysenv", "use"),
            flow("use", "eol"),
            flow("use", "sysenv"),
            flow("eol", "sysenv"),
        ],
        parameter_values={"share": 0.6, **parameter_values},
        parameter_dims={"share": (), **parameter_dims},
        dimensions=DIM_DEFINITIONS + [DimensionDefinition(name="Cohort", letter="c", dtype=int)],
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


# Regression tests for reviewed edge cases


def test_total_falls_back_to_other_side_if_shares_are_zero():
    mfa = make_mfa(
        processes=[
            "sysenv",
            ProcessDefinition(
                name="a", inflow_shares={"s1": "z1", "s2": "z2"}, outflow_shares={"b": "sb"}
            ),
            "s1",
            "s2",
            "b",
            "c",
        ],
        flows=[
            flow("sysenv", "s1"),
            flow("sysenv", "s2"),
            flow("s1", "a"),
            flow("s2", "a"),
            flow("a", "b"),
            flow("a", "c"),
            flow("b", "sysenv"),
            flow("c", "sysenv"),
        ],
        parameter_values={
            "z1": np.array([0.0, 0.5, 0.5, 0.5]),
            "z2": np.array([1.0, 0.5, 0.5, 0.5]),
            "sb": 0.3,
        },
        parameter_dims={"z1": ("t",), "z2": ("t",), "sb": ()},
    )
    # the share of s1 is zero in the first year, so the total cannot be computed from it there
    mfa.flows["s1 => a"][...] = np.array([0.0, 5.0, 5.0, 5.0])
    mfa.flows["a => c"][...] = 7.0
    mfa.processes["a"].compute()
    np.testing.assert_allclose(mfa.flows["a => b"].values, 3.0)
    np.testing.assert_allclose(mfa.flows["s2 => a"].values, [10.0, 5.0, 5.0, 5.0])


def test_failed_computation_changes_nothing():
    mfa = make_mfa(
        processes=["sysenv", ProcessDefinition(name="a", outflow_shares={"b": "s"}), "b", "c"],
        flows=[
            flow("sysenv", "a", prescribed="input"),
            flow("a", "b", dims=("t", "r")),
            flow("a", "c", dims=("t", "r")),
        ],
        parameter_values={"input": 1.0, "s": np.array([0.2, 0.3])},
        parameter_dims={"input": ("t",), "s": ("r",)},
    )
    with pytest.raises(ValueError, match="forward dimension splitter"):
        mfa.processes["a"].compute()
    assert not mfa.flows["a => b"].is_set and not mfa.flows["a => c"].is_set


def test_failed_stock_computation_changes_nothing():
    mfa = stock_mfa(InflowDrivenDSM)
    mfa.stocks["use"].lifetime_model.mean = None
    mfa.flows["sysenv => use"][...] = 1.0
    with pytest.raises(Exception):
        mfa.processes["use"].compute()
    stock = mfa.stocks["use"]
    assert not any(a.is_set for a in (stock.inflow, stock.outflow, stock.stock))
    # retry with corrected input uses the new values
    stock.lifetime_model.mean = 1
    mfa.flows["sysenv => use"][...] = 5.0
    mfa.processes["use"].compute()
    np.testing.assert_allclose(mfa.flows["use => eol"].values, [0.0, 3.0, 3.0, 3.0])


def test_computed_stock_is_distributed_to_flows():
    mfa = stock_mfa(DynamicStockModel)
    stock = mfa.stocks["use"]
    stock.inflow[...] = 1.0
    stock.compute()
    mfa.processes["use"].compute()
    np.testing.assert_allclose(mfa.flows["sysenv => use"].values, 1.0)
    np.testing.assert_allclose(mfa.flows["use => eol"].values, 0.6 * stock.outflow.values)


def test_inconsistent_set_stock_raises():
    mfa = stock_mfa(SimpleFlowDrivenStock, lifetime=False)
    mfa.flows["sysenv => use"][...] = 2.0
    mfa.flows["use => sysenv"][...] = 0.4
    mfa.stocks["use"].stock[...] = 100.0
    with pytest.raises(ValueError, match="changed the values of the stock arrays"):
        mfa.processes["use"].compute()
    np.testing.assert_allclose(mfa.stocks["use"].stock.values, 100.0)


def test_shares_of_all_flows_are_not_overdetermined():
    mfa = make_mfa(
        processes=[
            "sysenv",
            ProcessDefinition(name="a", outflow_shares={"b": "s", "c": "s"}),
            "b",
            "c",
        ],
        flows=[flow("sysenv", "a"), flow("a", "b"), flow("a", "c")],
        parameter_values={"s": 0.5},
        parameter_dims={"s": ()},
    )
    mfa.flows["sysenv => a"][...] = 1.0
    assert not mfa.processes["a"].is_overdetermined


def test_side_without_flows_has_zero_total():
    mfa = make_mfa(
        processes=["sysenv", "landfill"],
        flows=[flow("sysenv", "landfill")],
        parameter_values={},
        parameter_dims={},
        stocks=[
            StockDefinition(
                name="landfill",
                process="landfill",
                dim_letters=("t",),
                subclass=SimpleFlowDrivenStock,
            )
        ],
    )
    mfa.stocks["landfill"].outflow.values[...] = 0.5  # stale values, not set
    mfa.flows["sysenv => landfill"][...] = 1.0
    mfa.processes["landfill"].compute()
    np.testing.assert_allclose(mfa.stocks["landfill"].outflow.values, 0.0)
    assert mfa.check_mass_balance()


@pytest.mark.parametrize("order", [["sysenv", "p", "q"], ["sysenv", "q", "p"]])
def test_inconsistency_found_regardless_of_order(order):
    processes = [
        ProcessDefinition(name="p", outflow_shares={"sysenv": "share"}) if n == "p" else n
        for n in order
    ]
    mfa = make_mfa(
        processes=processes,
        flows=[
            flow("sysenv", "p", prescribed="a"),
            flow("p", "sysenv"),
            flow("p", "q"),
            flow("q", "sysenv", prescribed="d"),
        ],
        parameter_values={"a": 10.0, "share": 0.3, "d": 5.0},
        parameter_dims={"a": ("t",), "share": (), "d": ("t",)},
    )
    with pytest.raises(ValueError, match="Mass balance check failed"):
        mfa.compute_all_possible()


def test_sysenv_must_have_id_zero():
    with pytest.raises(ValueError, match="must be named 'sysenv'"):
        make_mfa(
            [ProcessDefinition(name="sysenv", id=3), ProcessDefinition(name="a", id=1)], [], {}, {}
        )


def test_unsupported_stock_type_is_skipped():
    class CustomStock(SimpleFlowDrivenStock):
        def _compute_process(self, computation):
            return Stock._compute_process(self, computation)

    mfa = stock_mfa(CustomStock, lifetime=False)
    mfa.flows["sysenv => use"][...] = 1.0
    assert mfa.compute_all_possible() == ["use", "eol"]
    with pytest.raises(UnderdeterminedError, match="does not support"):
        mfa.processes["use"].compute()


def test_prescribed_flow_without_flow_dims():
    with pytest.raises(ValueError, match="does not have its dimensions"):
        make_mfa(
            ["sysenv", "a"],
            [flow("sysenv", "a", dims=("t", "r"), prescribed="p")],
            {"p": 1.0},
            {"p": ("t",)},
        )


def test_stock_inflow_from_parameter():
    mfa = stock_mfa(
        InflowDrivenDSM,
        parameter_values={"inflow": 2.0},
        parameter_dims={"inflow": ("t",)},
        prescribed_inflow="inflow",
    )
    assert mfa.stocks["use"].inflow.is_set
    assert mfa.compute_all_possible() == []
    np.testing.assert_allclose(mfa.flows["sysenv => use"].values, 2.0)
    np.testing.assert_allclose(mfa.flows["use => eol"].values, [0.0, 1.2, 1.2, 1.2])
    # set again after mark_all_unset, with changed parameter values
    mfa.parameters["inflow"][...] = 1.0
    mfa.mark_all_unset()
    assert mfa.compute_all_possible() == []
    np.testing.assert_allclose(mfa.flows["sysenv => use"].values, 1.0)


def test_stock_from_parameter():
    mfa = stock_mfa(
        DynamicStockModel,
        parameter_values={"stock": 1.0},
        parameter_dims={"stock": ("t",)},
        prescribed_stock="stock",
    )
    assert mfa.compute_all_possible() == []
    np.testing.assert_allclose(mfa.flows["sysenv => use"].values, 1.0)
    assert mfa.check_mass_balance()


def test_inflow_by_cohort_from_parameter():
    inflow_by_cohort = np.diag([1.0, 2.0, 3.0, 4.0])
    mfa = stock_mfa(
        DynamicStockModel,
        parameter_values={"ibc": inflow_by_cohort},
        parameter_dims={"ibc": ("t", "c")},
        cohort_letter="c",
        prescribed_inflow_by_cohort="ibc",
    )
    assert mfa.compute_all_possible() == []
    np.testing.assert_allclose(mfa.flows["sysenv => use"].values, [1.0, 2.0, 3.0, 4.0])
    assert mfa.check_mass_balance()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"prescribed_stock": "p", "prescribed_inflow": "p"}, "at most one"),
        ({"prescribed_inflow_by_cohort": "p"}, "requires cohort_letter"),
        ({"subclass": SimpleFlowDrivenStock, "cohort_letter": "c"}, "no cohort dimension"),
    ],
)
def test_invalid_prescribed_stocks(kwargs, match):
    kwargs = {"subclass": DynamicStockModel, "lifetime_model_class": FixedLifetime, **kwargs}
    if kwargs["subclass"] is SimpleFlowDrivenStock:
        del kwargs["lifetime_model_class"]
    with pytest.raises(ValueError, match=match):
        StockDefinition(name="s", process="use", dim_letters=("t",), **kwargs)


def test_undefined_prescribed_stock():
    with pytest.raises(ValueError, match="'missing' used in definition of stock 'use'"):
        stock_mfa(InflowDrivenDSM, prescribed_inflow="missing")


# Linking of parameters ------------------------------------------------------------------------


def linked_split_definition(prescribed: str = "input", **kwargs) -> MFADefinition:
    """Definition of split_mfa(), with the given parameter prescribing the inflow of 'a'."""
    return MFADefinition(
        dimensions=DIM_DEFINITIONS,
        processes=[
            "sysenv",
            ProcessDefinition(name="a", outflow_shares={"b": "share b"}),
            "b",
            "c",
        ],
        flows=[
            flow("sysenv", "a", prescribed=prescribed),
            flow("a", "b"),
            flow("a", "c"),
            flow("b", "sysenv"),
            flow("c", "sysenv"),
        ],
        parameters=[
            ParameterDefinition(name="input", dim_letters=("t",)),
            ParameterDefinition(name="share b", dim_letters=()),
        ],
        **kwargs,
    )


SPLIT_VALUES = {"input": np.array([1.0, 2.0, 3.0, 4.0]), "share b": 0.3}


def test_prescribed_parameter_added_before_linking():
    definition = linked_split_definition(prescribed="input sum", check_parameter_names=False)
    mfa = MFASystem.from_data_reader(definition, DictReader(SPLIT_VALUES), link=False)
    parameters = mfa.parameters
    parameters["input sum"] = parameters["input"] + parameters["input"]
    mfa.link_parameters()
    assert mfa.compute_all_possible() == []
    np.testing.assert_allclose(mfa.flows["sysenv => a"].values, [2.0, 4.0, 6.0, 8.0])
    np.testing.assert_allclose(mfa.flows["a => b"].values, [0.6, 1.2, 1.8, 2.4])


def test_undefined_parameter_is_checked_in_definition_only_if_enabled():
    with pytest.raises(ValueError, match="'missing' used in flow definition"):
        linked_split_definition(prescribed="missing")
    linked_split_definition(prescribed="missing", check_parameter_names=False)


def test_link_parameters_lists_all_missing_parameters():
    definition = linked_split_definition(prescribed="missing", check_parameter_names=False)
    mfa = MFASystem.from_data_reader(definition, DictReader(SPLIT_VALUES), link=False)
    del mfa.parameters["share b"]
    with pytest.raises(ValueError) as error:
        mfa.link_parameters()
    assert "'missing' (used by flow 'sysenv => a')" in str(error.value)
    assert "'share b' (used by process 'a')" in str(error.value)


def test_unlinked_parameters_raise():
    definition = linked_split_definition()
    mfa = MFASystem.from_data_reader(definition, DictReader(SPLIT_VALUES), link=False)
    assert mfa.processes["a"].outflow_shares == {"b": "share b"}
    assert "outflow_shares={'b': 'share b'}" in repr(mfa.processes["a"])
    with pytest.raises(ValueError, match="'input' prescribed for 'sysenv => a' is not linked"):
        mfa.set_arrays_from_prescribed()
    mfa.flows["sysenv => a"][...] = 1.0
    with pytest.raises(ValueError, match="share b of shares or dimension splitters are not"):
        mfa.processes["a"].compute()


def test_link_parameters_resolves_share_keys():
    definition = linked_split_definition()
    mfa = MFASystem.from_data_reader(definition, DictReader(SPLIT_VALUES))
    shares = mfa.processes["a"].outflow_shares
    assert list(shares) == ["a => b"]
    assert shares["a => b"] is mfa.parameters["share b"]
    # linking again changes nothing
    mfa.link_parameters()
    assert mfa.processes["a"].outflow_shares == shares


def test_set_arrays_from_prescribed_overwrites_and_uses_parameter_changes():
    mfa = split_mfa()
    flow = mfa.flows["sysenv => a"]
    flow[...] = 5.0
    mfa.parameters["input"][...] = 7.0
    mfa.set_arrays_from_prescribed()
    np.testing.assert_allclose(flow.values, 7.0)
    assert flow.is_set


def test_prescribed_flow_is_kept_when_relink_renames_it():
    from flodym.flow_naming import process_ids

    mfa = split_mfa()
    mfa.mark_all_unset()
    with config.override(flow_naming=process_ids):
        mfa.relink()
    assert "F0_1" in mfa.flows
    assert list(mfa.processes["a"].outflow_shares) == ["F1_2"]
    assert mfa.compute_all_possible() == []
    np.testing.assert_allclose(mfa.flows["F1_2"].values, [0.3, 0.6, 0.9, 1.2])
