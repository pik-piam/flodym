"""Tests for custom computations of processes and the default MFASystem.compute()."""

import logging

import numpy as np
import pytest

from flodym import (
    CustomComputation,
    Dimension,
    DimensionSet,
    FlowDefinition,
    MFASystem,
    SimpleFlowDrivenStock,
    StockDefinition,
    UnderdeterminedError,
    make_empty_flows,
    make_empty_stocks,
    make_processes,
)

DIMS = DimensionSet(dim_list=[Dimension(name="Time", letter="t", items=[2000, 2001, 2002])])
INPUT = np.array([1.0, 2.0, 3.0])
RECYCLING_SHARE = 0.2


def make_mfa(
    processes: list[str],
    links: list[tuple[str, str]],
    stocks: list[StockDefinition] | None = None,
) -> MFASystem:
    process_dict = make_processes(processes)
    flow_definitions = [
        FlowDefinition(from_process_name=a, to_process_name=b, dim_letters=("t",)) for a, b in links
    ]
    return MFASystem(
        dims=DIMS,
        parameters={},
        processes=process_dict,
        flows=make_empty_flows(process_dict, flow_definitions, DIMS),
        stocks=make_empty_stocks(stocks or [], process_dict, DIMS),
    )


def loop_mfa() -> MFASystem:
    """sysenv -> a -> b -> c -> sysenv, with a recycling flow b -> a.

    Processes a and b form an equation system, which the automatic computation cannot solve.
    """
    return make_mfa(
        ["sysenv", "a", "b", "c"],
        [("sysenv", "a"), ("a", "b"), ("b", "a"), ("b", "c"), ("c", "sysenv")],
    )


def solve_forward(mfa: MFASystem) -> None:
    """Solve a and b from the flow entering a."""
    total_b = mfa.flows["sysenv => a"] / (1 - RECYCLING_SHARE)
    mfa.flows["a => b"][...] = total_b
    mfa.flows["b => a"][...] = total_b * RECYCLING_SHARE
    mfa.flows["b => c"][...] = total_b * (1 - RECYCLING_SHARE)


def solve_backward(mfa: MFASystem) -> None:
    """Solve a and b from the flow leaving b."""
    total_b = mfa.flows["b => c"] / (1 - RECYCLING_SHARE)
    mfa.flows["a => b"][...] = total_b
    mfa.flows["b => a"][...] = total_b * RECYCLING_SHARE
    mfa.flows["sysenv => a"][...] = total_b * (1 - RECYCLING_SHARE)


def check_loop_solution(mfa: MFASystem) -> None:
    assert all(p.is_computed for p in mfa.processes.values() if p.id != 0)
    np.testing.assert_allclose(mfa.flows["a => b"].values, INPUT / (1 - RECYCLING_SHARE))
    np.testing.assert_allclose(mfa.flows["c => sysenv"].values, INPUT)
    mfa.check_mass_balance()


def test_compute_defaults_to_compute_all_possible(caplog):
    mfa = make_mfa(["sysenv", "a", "b"], [("sysenv", "a"), ("a", "b"), ("b", "sysenv")])
    mfa.flows["sysenv => a"][...] = INPUT
    with caplog.at_level(logging.WARNING):
        mfa.compute()
    np.testing.assert_allclose(mfa.flows["b => sysenv"].values, INPUT)
    assert not caplog.records


def test_compute_warns_about_uncomputed_processes(caplog):
    mfa = loop_mfa()
    mfa.flows["sysenv => a"][...] = INPUT
    with caplog.at_level(logging.WARNING):
        mfa.compute()
    assert "Processes a, b, c could not be computed" in caplog.text


def test_coupled_processes_with_default_requires():
    mfa = loop_mfa()
    mfa.add_custom_computation(["a", "b"], solve_forward)
    assert mfa.custom_computations[0].requires is None
    mfa.flows["sysenv => a"][...] = INPUT
    assert mfa.compute_all_possible() == []
    check_loop_solution(mfa)


def test_custom_computation_waits_for_required_arrays():
    # q comes first in the queue, but must wait for p
    mfa = make_mfa(["sysenv", "q", "p"], [("sysenv", "p"), ("p", "q"), ("q", "sysenv")])
    calls = []

    def compute_q(mfa: MFASystem) -> None:
        calls.append(mfa.flows["p => q"].is_set)
        mfa.flows["q => sysenv"][...] = 2 * mfa.flows["p => q"]

    mfa.add_custom_computation("q", compute_q)
    mfa.flows["sysenv => p"][...] = INPUT
    assert mfa.compute_all_possible() == []
    assert calls == [True]
    np.testing.assert_allclose(mfa.flows["q => sysenv"].values, 2 * INPUT)


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_alternative_custom_computations(direction):
    mfa = loop_mfa()
    calls = []

    def tracked(function, name):
        def wrapper(mfa: MFASystem) -> None:
            calls.append(name)
            function(mfa)

        return wrapper

    mfa.add_custom_computation(["a", "b"], tracked(solve_forward, "forward"))
    mfa.add_custom_computation(["a", "b"], tracked(solve_backward, "backward"), requires=["b => c"])
    if direction == "forward":
        mfa.flows["sysenv => a"][...] = INPUT
    else:
        # c is computed automatically first, which sets b => c
        mfa.flows["c => sysenv"][...] = INPUT
    assert mfa.compute_all_possible() == []
    assert calls == [direction]
    check_loop_solution(mfa)


def test_custom_computation_which_leaves_flows_unset_raises_and_restores():
    mfa = make_mfa(
        ["sysenv", "p", "q", "r"],
        [("sysenv", "p"), ("p", "q"), ("q", "sysenv"), ("q", "r"), ("r", "sysenv")],
    )

    def incomplete(mfa: MFASystem) -> None:
        mfa.flows["q => sysenv"][...] = mfa.flows["p => q"]

    mfa.add_custom_computation("q", incomplete)
    mfa.flows["sysenv => p"][...] = INPUT
    with pytest.raises(ValueError, match="did not set all flows"):
        mfa.compute_all_possible()
    assert not mfa.flows["q => sysenv"].is_set
    np.testing.assert_allclose(mfa.flows["q => sysenv"].values, 0.0)


def test_underdetermined_custom_computation_is_skipped():
    mfa = loop_mfa()

    def not_yet(mfa: MFASystem) -> None:
        mfa.flows["a => b"][...] = 1.0
        raise UnderdeterminedError(mfa.processes["a"], "Not enough information.")

    mfa.add_custom_computation(["a", "b"], not_yet)
    mfa.flows["sysenv => a"][...] = INPUT
    assert mfa.compute_all_possible() == ["a", "b", "c"]
    assert not mfa.flows["a => b"].is_set


def test_required_stock_array():
    mfa = make_mfa(
        ["sysenv", "p"],
        [("sysenv", "p"), ("p", "sysenv")],
        stocks=[
            StockDefinition(
                name="s", process="p", dim_letters=("t",), subclass=SimpleFlowDrivenStock
            )
        ],
    )
    mfa.add_custom_computation("p", solve_forward, requires=["s.inflow"])
    with pytest.raises(ValueError, match="neither a flow nor an array of a stock"):
        mfa.add_custom_computation("p", solve_forward, requires=["s.foo"])


@pytest.mark.parametrize(
    "processes, requires, match",
    [
        (["x"], None, "Process 'x' of a custom computation does not exist"),
        (["sysenv"], None, "'sysenv' cannot have a custom computation"),
        ([], None, "at least one process"),
        (["a"], ["x => y"], "neither a flow nor an array of a stock"),
        (["b", "a"], None, "two custom computations of processes"),
    ],
)
def test_invalid_custom_computations(processes, requires, match):
    mfa = loop_mfa()
    mfa.add_custom_computation(["a", "b"], solve_forward)
    with pytest.raises(ValueError, match=match):
        mfa.add_custom_computation(processes, solve_forward, requires=requires)
    assert len(mfa.custom_computations) == 1


def test_custom_computations_are_validated_at_initialization():
    mfa = loop_mfa()
    with pytest.raises(ValueError, match="Process 'x' of a custom computation does not exist"):
        MFASystem(
            dims=mfa.dims,
            parameters={},
            processes=mfa.processes,
            flows=mfa.flows,
            custom_computations=[CustomComputation(processes=["x"], function=solve_forward)],
        )
