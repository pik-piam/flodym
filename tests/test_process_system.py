"""Tests for the simultaneous computation of processes, e.g. in recycling loops.

See ``flodym.compute_processes``. The systems are built with the helpers of
``process_helpers.py``.
"""

import logging

import numpy as np
import pytest
from process_helpers import (
    DIM_DEFINITIONS,
    TIME,
    flow,
    make_mfa,
    split_mfa,
    splitter_mfa,
    stock_mfa,
)

import flodym as fd
import flodym._linear_system as linear_system
import flodym._process_system as process_system
from flodym._process_equations import ProcessEquations, assemble_processes
from flodym.example_objects import get_example_mfa

PRIMARY = np.array([10.0, 20.0, 30.0, 40.0])
COHORT_DIMENSIONS = DIM_DEFINITIONS + [fd.DimensionDefinition(name="Cohort", letter="c", dtype=int)]
GOOD_DIMENSIONS = DIM_DEFINITIONS + [fd.DimensionDefinition(name="Good", letter="g", dtype=str)]
LOOP_PROCESSES = ["production", "use", "eol", "rec"]


def loop_definitions(prefix: str = "", dims: tuple[str, ...] = ("t",), primary: bool = True):
    """Processes and flows of a recycling loop.

    production -> use -> eol -> rec -> production, with the share 'collection' of eol going to
    rec and the share 'yield' of rec going back to production. Losses go to sysenv. The primary
    input into production is prescribed by the parameter 'primary'.
    """
    production, use, eol, rec = (f"{prefix}{name}" for name in LOOP_PROCESSES)
    processes = [
        production,
        use,
        fd.ProcessDefinition(name=eol, outflow_shares={rec: f"{prefix}collection"}),
        fd.ProcessDefinition(name=rec, outflow_shares={production: f"{prefix}yield"}),
    ]
    prescribed = {"prescribed": f"{prefix}primary"} if primary else {}
    flows = [
        flow("sysenv", production, dims, **prescribed),
        flow(production, use, dims),
        flow(use, eol, dims),
        flow(eol, rec, dims),
        flow(eol, "sysenv", dims),
        flow(rec, production, dims),
        flow(rec, "sysenv", dims),
    ]
    return processes, flows


def without_flow(flows: list, from_process: str, to_process: str) -> list:
    return [
        f for f in flows if (f.from_process_name, f.to_process_name) != (from_process, to_process)
    ]


def loop_mfa(
    collection=0.8,
    recycling_yield=0.5,
    primary=PRIMARY,
    dims=("t",),
    parameter_dims=None,
    extra_processes=(),
    extra_flows=(),
    removed_flows=(),
    extra_parameters=None,
    **kwargs,
) -> fd.MFASystem:
    """The recycling loop of :py:func:`loop_definitions`, with more processes, flows and
    parameters if given; ``primary=None`` leaves the primary input unknown."""
    processes, flows = loop_definitions(dims=dims, primary=primary is not None)
    for from_process, to_process in removed_flows:
        flows = without_flow(flows, from_process, to_process)
    parameter_values = {"collection": collection, "yield": recycling_yield}
    parameter_dims = {"collection": (), "yield": (), **(parameter_dims or {})}
    if primary is not None:
        parameter_values["primary"] = primary
        parameter_dims.setdefault("primary", dims)
    for name, (values, letters) in (extra_parameters or {}).items():
        parameter_values[name] = values
        parameter_dims[name] = letters
    return make_mfa(
        ["sysenv"] + processes + list(extra_processes),
        flows + list(extra_flows),
        parameter_values,
        parameter_dims,
        **kwargs,
    )


def two_loops_mfa(**parameter_values) -> fd.MFASystem:
    """Two independent recycling loops, the second with the prefix 'b_'."""
    processes_a, flows_a = loop_definitions()
    processes_b, flows_b = loop_definitions(prefix="b_")
    values = {"collection": 0.8, "yield": 0.5, "b_collection": 0.8, "b_yield": 0.5}
    values |= {"primary": PRIMARY, "b_primary": PRIMARY, **parameter_values}
    return make_mfa(
        ["sysenv"] + processes_a + processes_b,
        flows_a + flows_b,
        values,
        {name: ("t",) if "primary" in name else () for name in values},
    )


def assert_loop_values(mfa: fd.MFASystem, throughput, collection=0.8, recycling_yield=0.5):
    np.testing.assert_allclose(mfa.flows["production => use"].values, throughput)
    np.testing.assert_allclose(mfa.flows["use => eol"].values, throughput)
    np.testing.assert_allclose(mfa.flows["eol => rec"].values, collection * throughput)
    np.testing.assert_allclose(mfa.flows["eol => sysenv"].values, (1 - collection) * throughput)
    returned = collection * recycling_yield * throughput
    np.testing.assert_allclose(mfa.flows["rec => production"].values, returned)
    np.testing.assert_allclose(
        mfa.flows["rec => sysenv"].values, collection * (1 - recycling_yield) * throughput
    )


def array_state(mfa: fd.MFASystem) -> dict:
    """Values and is_set flags of all flows and stock arrays."""
    arrays = dict(mfa.flows)
    for name, stock in mfa.stocks.items():
        arrays.update({f"{name} {a}": getattr(stock, a) for a in ("inflow", "outflow", "stock")})
    return {name: (a.values.copy(), a.is_set) for name, a in arrays.items()}


def assert_state_equal(state: dict, other: dict) -> None:
    assert state.keys() == other.keys()
    for name, (values, is_set) in state.items():
        np.testing.assert_array_equal(values, other[name][0], err_msg=name)
        assert is_set == other[name][1], name


def set_flows(mfa: fd.MFASystem) -> list[str]:
    return [name for name, f in mfa.flows.items() if f.is_set]


def messages(caplog, level: int) -> list[str]:
    """Log messages of flodym at the given level."""
    return [
        r.getMessage() for r in caplog.records if r.levelno == level and r.name.startswith("flodym")
    ]


# Recycling loops ------------------------------------------------------------------------------


def test_recycling_loop_hand_computed(caplog):
    mfa = loop_mfa()
    with caplog.at_level(logging.INFO):
        assert mfa.compute_all_possible() == []
    # T = P + collection * yield * T
    assert_loop_values(mfa, PRIMARY / (1 - 0.8 * 0.5))
    np.testing.assert_allclose(
        mfa.flows["production => use"].values, [16.666667, 33.333333, 50.0, 66.666667]
    )
    assert mfa.check_mass_balance()
    expected = "Computed processes 'production', 'use', 'eol', 'rec' simultaneously."
    assert expected in messages(caplog, logging.INFO)


def test_one_process_at_a_time_leaves_loop_uncomputed():
    mfa = loop_mfa()
    assert mfa.compute_all_possible(simultaneous=False) == LOOP_PROCESSES
    assert set_flows(mfa) == ["sysenv => production"]


def test_loop_with_regions_and_varying_shares():
    collection = np.array([[0.5, 0.9], [0.6, 0.8], [0.7, 0.7], [0.8, 0.6]])
    recycling_yield = np.array([0.4, 0.9])
    primary = np.arange(1.0, 9.0).reshape(4, 2)
    mfa = loop_mfa(
        collection=collection,
        recycling_yield=recycling_yield,
        primary=primary,
        dims=("t", "r"),
        parameter_dims={"collection": ("t", "r"), "yield": ("r",)},
    )
    assert mfa.compute_all_possible() == []
    assert_loop_values(
        mfa, primary / (1 - collection * recycling_yield), collection, recycling_yield
    )
    assert mfa.check_mass_balance()


def test_loop_with_many_items():
    rng = np.random.default_rng(1)
    shape = (50, 100)
    primary = rng.uniform(1.0, 2.0, shape)
    collection = rng.uniform(0.3, 0.9, shape)
    recycling_yield = rng.uniform(0.3, 0.9, shape[1])
    mfa = loop_mfa(
        collection=collection,
        recycling_yield=recycling_yield,
        primary=primary,
        dims=("t", "r"),
        parameter_dims={"collection": ("t", "r"), "yield": ("r",)},
        dim_items={"Time": list(range(shape[0])), "Region": [f"r{i}" for i in range(shape[1])]},
    )
    assert mfa.compute_all_possible() == []
    assert_loop_values(
        mfa, primary / (1 - collection * recycling_yield), collection, recycling_yield
    )


@pytest.mark.parametrize("order", [[0, 1, 2, 3], [3, 1, 0, 2], [2, 3, 1, 0]])
def test_result_does_not_depend_on_process_order(order):
    processes, flows = loop_definitions()
    mfa = make_mfa(
        ["sysenv"] + [processes[i] for i in order],
        flows,
        {"collection": 0.8, "yield": 0.5, "primary": PRIMARY},
        {"collection": (), "yield": (), "primary": ("t",)},
    )
    assert mfa.compute_all_possible() == []
    assert_loop_values(mfa, PRIMARY / 0.6)


def splitter_loop_mfa() -> fd.MFASystem:
    """p -> q -> p, with a forward splitter in p, and the flows from p to q by region."""
    return make_mfa(
        processes=[
            "sysenv",
            fd.ProcessDefinition(name="p", forward_dimension_splitter="split"),
            fd.ProcessDefinition(name="q", outflow_shares={"p": "yield"}),
        ],
        flows=[
            flow("sysenv", "p", prescribed="primary"),
            flow("p", "q", dims=("t", "r")),
            flow("q", "p"),
            flow("q", "sysenv", dims=("t", "r")),
        ],
        parameter_values={
            "primary": 3.0,
            "split": np.array([0.25, 0.75]),
            "yield": np.array([0.4, 0.8]),
        },
        parameter_dims={"primary": ("t",), "split": ("r",), "yield": ("r",)},
    )


def test_dimension_splitter_in_loop():
    mfa = splitter_loop_mfa()
    assert mfa.compute_all_possible() == []
    # q => p = sum_r yield_r * split_r * (3 + q => p) = 0.7 * (3 + q => p)
    np.testing.assert_allclose(mfa.flows["q => p"].values, 7.0)
    np.testing.assert_allclose(mfa.flows["p => q"].values, [[2.5, 7.5]] * 4)
    np.testing.assert_allclose(mfa.flows["q => sysenv"].values, [[1.5, 1.5]] * 4)
    assert mfa.check_mass_balance()


def test_dimension_splitter_in_loop_not_summing_to_one(caplog):
    mfa = splitter_loop_mfa()
    mfa.parameters["split"][...] = np.array([0.25, 0.7])
    with caplog.at_level(logging.WARNING):
        assert mfa.compute_all_possible() == []
    assert "forward dimension splitter does not sum to one" in caplog.text


def test_unused_dimension_splitter_in_loop(caplog):
    mfa = loop_mfa(extra_parameters={"split": (np.array([0.5, 0.5]), ("r",))})
    mfa.processes["use"].set_dimension_splitter("forward", mfa.parameters["split"])
    with caplog.at_level(logging.WARNING):
        assert mfa.compute_all_possible() == []
    assert "In process 'use': The forward dimension splitter is set, but was not" in caplog.text


def test_backward_splitter_in_demand_driven_loop():
    """The final demand is prescribed; use takes the share 0.6 from production, the rest from
    reuse, which returns 0.8 of what leaves use towards eol."""
    demand = np.array([1.0, 2.0, 3.0, 4.0])
    split = np.array([0.3, 0.7])
    mfa = make_mfa(
        processes=[
            "sysenv",
            "production",
            fd.ProcessDefinition(
                name="use",
                inflow_shares={"production": "production share"},
                backward_dimension_splitter="split",
            ),
            fd.ProcessDefinition(name="eol", outflow_shares={"use": "reuse"}),
        ],
        flows=[
            flow("sysenv", "production", dims=("t", "r")),
            flow("production", "use", dims=("t", "r")),
            flow("eol", "use"),
            flow("use", "sysenv", prescribed="demand"),
            flow("use", "eol"),
            flow("eol", "sysenv"),
        ],
        parameter_values={"demand": demand, "split": split, "production share": 0.6, "reuse": 0.8},
        parameter_dims={"demand": ("t",), "split": ("r",), "production share": (), "reuse": ()},
    )
    assert mfa.compute_all_possible(simultaneous=False) == ["production", "use", "eol"]
    assert mfa.compute_all_possible() == []
    # 0.4 * T = 0.8 * (T - demand), so the throughput of use is T = 2 * demand
    np.testing.assert_allclose(mfa.flows["use => eol"].values, demand)
    np.testing.assert_allclose(mfa.flows["eol => use"].values, 0.8 * demand)
    primary = np.outer(1.2 * demand, split)
    np.testing.assert_allclose(mfa.flows["production => use"].values, primary)
    np.testing.assert_allclose(mfa.flows["sysenv => production"].values, primary)
    assert mfa.check_mass_balance()


def test_process_compute_with_neighbors_runs_into_loop():
    mfa = loop_mfa(
        primary=None,
        extra_processes=["feed"],
        extra_flows=[flow("sysenv", "feed", prescribed="primary"), flow("feed", "production")],
        removed_flows=[("sysenv", "production")],
        extra_parameters={"primary": (PRIMARY, ("t",))},
    )
    assert mfa.processes["feed"].compute(compute_neighbors=True)
    assert all(p.is_computed for p in mfa.processes.values() if p.id != 0)
    assert_loop_values(mfa, PRIMARY / 0.6)


def test_compute_processes_subset_and_one_at_a_time():
    mfa = loop_mfa(
        extra_processes=["other"],
        extra_flows=[flow("sysenv", "other", prescribed="primary"), flow("other", "sysenv")],
    )
    fd.compute_processes([mfa.processes["eol"]], simultaneous=False)
    assert not any(p.is_computed for p in mfa.processes.values() if p.id != 0)
    fd.compute_processes([mfa.processes["eol"]])
    assert all(mfa.processes[name].is_computed for name in LOOP_PROCESSES)
    assert not mfa.processes["other"].is_computed
    assert_loop_values(mfa, PRIMARY / 0.6)


# Singular, underdetermined and partly determined blocks ----------------------------------------


@pytest.mark.parametrize(
    "collection, items",
    [
        (1.0, "at Time 2000; Time 2001; Time 2002 and others:"),
        (np.array([0.8, 1.0, 0.8, 0.8]), "at Time 2001:"),
    ],
)
def test_singular_loop(collection, items, caplog):
    def singular_loop_mfa() -> fd.MFASystem:
        return loop_mfa(
            collection=collection, recycling_yield=1.0, parameter_dims={"collection": ("t",)}
        )

    mfa = singular_loop_mfa()
    with caplog.at_level(logging.INFO):
        assert mfa.compute_all_possible() == LOOP_PROCESSES
    warnings = messages(caplog, logging.WARNING)
    assert len(warnings) == 1
    assert "form a loop without exit " + items in warnings[0]
    assert "No item of these processes is computed. Lower a share" in warnings[0]
    assert set_flows(mfa) == ["sysenv => production"]

    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert singular_loop_mfa().compute_all_possible(on_underdetermined="ignore")
    assert not messages(caplog, logging.WARNING) and not messages(caplog, logging.INFO)

    with caplog.at_level(logging.INFO):
        assert singular_loop_mfa().compute_all_possible(on_underdetermined="warn")
    assert len(messages(caplog, logging.WARNING)) == 1

    caplog.clear()
    with (
        caplog.at_level(logging.INFO),
        pytest.raises(fd.UnderdeterminedError, match="loop without exit") as error,
    ):
        singular_loop_mfa().compute_all_possible(on_underdetermined="error")
    assert not messages(caplog, logging.WARNING)
    assert error.value.process_names == LOOP_PROCESSES
    assert error.value.process_name == "production"
    assert "production => use" in error.value.flow_names


def test_independent_loops_singular_and_regular(caplog):
    mfa = two_loops_mfa(collection=1.0, **{"yield": 1.0})
    with caplog.at_level(logging.WARNING):
        assert mfa.compute_all_possible() == LOOP_PROCESSES
    assert "loop without exit" in caplog.text
    np.testing.assert_allclose(mfa.flows["b_production => b_use"].values, PRIMARY / 0.6)


def parallel_flows_mfa() -> fd.MFASystem:
    """Two parallel flows without shares between a and b, with known input and output."""
    return make_mfa(
        ["sysenv", "a", "b"],
        [
            flow("sysenv", "a", prescribed="input"),
            flow("a", "b", name_override="route 1"),
            flow("a", "b", name_override="route 2"),
            flow("b", "sysenv", prescribed="input"),
        ],
        {"input": 10.0},
        {"input": ("t",)},
    )


def diamond_flows() -> list:
    """a splits into b and c without shares, which join in d."""
    return [
        flow("a", "b"),
        flow("a", "c"),
        flow("b", "d"),
        flow("c", "d"),
        flow("d", "sysenv", prescribed="output"),
    ]


def test_parallel_flows_without_shares_are_underdetermined_not_singular(caplog):
    mfa = parallel_flows_mfa()
    with caplog.at_level(logging.INFO):
        assert mfa.compute_all_possible() == ["a", "b"]
    assert not messages(caplog, logging.WARNING)
    assert "only fix the sum of the flows 'route 1', 'route 2'" in caplog.text


def test_diamond_without_shares_is_underdetermined(caplog):
    mfa = make_mfa(
        ["sysenv", "a", "b", "c", "d"],
        [flow("sysenv", "a", prescribed="output")] + diamond_flows(),
        {"output": 10.0},
        {"output": ("t",)},
    )
    with caplog.at_level(logging.INFO):
        assert mfa.compute_all_possible() == ["a", "b", "c", "d"]
    assert not messages(caplog, logging.WARNING)
    assert "only fix the sum of the flows 'a => b', 'a => c', 'b => d', 'c => d'" in caplog.text


def test_loop_next_to_diamond_is_computed(caplog):
    mfa = loop_mfa(
        extra_processes=["a", "b", "c", "d"],
        extra_flows=[flow("eol", "a")] + diamond_flows(),
        removed_flows=[("eol", "sysenv")],
        extra_parameters={"output": (0.2 * PRIMARY / 0.6, ("t",))},
    )
    with caplog.at_level(logging.INFO):
        assert mfa.compute_all_possible() == ["a", "b", "c", "d"]
    assert not messages(caplog, logging.WARNING)
    np.testing.assert_allclose(mfa.flows["production => use"].values, PRIMARY / 0.6)
    np.testing.assert_allclose(mfa.flows["eol => a"].values, 0.2 * PRIMARY / 0.6)


def test_underdetermined_loop_reports_flows_and_hints():
    mfa = loop_mfa()
    mfa.processes["eol"].remove_outflow_share("rec")
    with pytest.raises(fd.UnderdeterminedError) as error:
        mfa.compute_all_possible(on_underdetermined="error")
    message = str(error.value)
    assert message.startswith(
        "Cannot compute processes 'production', 'use', 'eol', 'rec', as they are underdetermined."
        "\nProcesses 'production', 'use', 'eol', 'rec' depend on each other, but are not "
        "determined:\n"
    )
    assert "'eol => rec'" in message and "are not determined" in message
    assert "- Process 'eol': Cannot compute the outflows from the total" in message
    assert set(error.value.flow_names) == set(mfa.flows) - {"sysenv => production"}
    assert error.value.process_names == LOOP_PROCESSES
    assert set_flows(mfa) == ["sysenv => production"]


@pytest.mark.parametrize("behavior, level", [("info", logging.INFO), ("warn", logging.WARNING)])
def test_on_underdetermined_logs_reasons_once(behavior, level, caplog):
    mfa = loop_mfa()
    mfa.processes["eol"].remove_outflow_share("rec")
    with caplog.at_level(logging.INFO):
        mfa.compute_all_possible(on_underdetermined=behavior)
    reports = [r for r in caplog.records if "depend on each other" in r.getMessage()]
    assert len(reports) == 1 and reports[0].levelno == level
    assert "Skip computation" not in caplog.text


def test_determined_loop_next_to_undetermined_processes():
    """eol sends its losses to waste, which splits them without shares."""
    mfa = loop_mfa(
        extra_processes=["waste", "incineration", "landfill"],
        extra_flows=[
            flow("eol", "waste"),
            flow("waste", "incineration"),
            flow("waste", "landfill"),
            flow("incineration", "sysenv"),
            flow("landfill", "sysenv"),
        ],
        removed_flows=[("eol", "sysenv")],
    )
    assert mfa.compute_all_possible() == ["waste", "incineration", "landfill"]
    np.testing.assert_allclose(mfa.flows["eol => waste"].values, 0.2 * PRIMARY / 0.6)
    # the flows between processes which are not computed stay unset
    for name in ("waste => incineration", "waste => landfill", "landfill => sysenv"):
        assert not mfa.flows[name].is_set


def test_determined_flow_between_incomplete_processes_stays_unset():
    """r => p is determined by the shares of p, but neither r nor p is determined completely."""
    mfa = make_mfa(
        processes=[
            "sysenv",
            "r",
            fd.ProcessDefinition(name="p", inflow_shares={"r": "share r", "sysenv": "share s"}),
            "c",
            "d",
        ],
        flows=[
            flow("sysenv", "r"),
            flow("r", "p"),
            flow("r", "sysenv"),
            flow("sysenv", "p", prescribed="input"),
            flow("p", "c"),
            flow("p", "d"),
            flow("c", "sysenv"),
            flow("d", "sysenv"),
        ],
        parameter_values={"share r": 0.4, "share s": 0.6, "input": 6.0},
        parameter_dims={"share r": (), "share s": (), "input": ("t",)},
    )
    before = array_state(mfa)
    assert mfa.compute_all_possible() == ["r", "p", "c", "d"]
    assert_state_equal(array_state(mfa), before)


def test_zero_rest_share_writes_nothing_for_incomplete_process():
    """p sends everything to c, so the rest to sysenv is zero, but the inflows of p are only
    determined in sum."""
    mfa = make_mfa(
        processes=["sysenv", fd.ProcessDefinition(name="p", outflow_shares={"c": "all"}), "c"],
        flows=[
            flow("sysenv", "p", name_override="a"),
            flow("sysenv", "p", name_override="b"),
            flow("p", "c", prescribed="output"),
            flow("p", "sysenv"),
            flow("c", "sysenv"),
        ],
        parameter_values={"all": 1.0, "output": 5.0},
        parameter_dims={"all": (), "output": ("t",)},
    )
    assert mfa.compute_all_possible() == ["p"]
    assert not mfa.flows["p => sysenv"].is_set


def test_shares_summing_to_one_do_not_determine_the_total(caplog):
    """The rest flow e has the remaining share 1 - (0.7 + 0.2 + 0.1), which is zero within the
    tolerance, as in the computation one process at a time."""
    mfa = make_mfa(
        processes=[
            "sysenv",
            fd.ProcessDefinition(name="a", outflow_shares={"b": "s1", "c": "s2", "d": "s3"}),
            "b",
            "c",
            "d",
            "e",
        ],
        flows=[flow("sysenv", "a")]
        + [flow("a", name) for name in "bcd"]
        + [flow("a", "e", prescribed="rest")]
        + [flow(name, "sysenv") for name in "bcde"],
        parameter_values={"s1": 0.7, "s2": 0.2, "s3": 0.1, "rest": 1.0},
        parameter_dims={"s1": (), "s2": (), "s3": (), "rest": ("t",)},
    )
    assert 1.0 - (0.7 + 0.2 + 0.1) != 0.0
    assert mfa.compute_all_possible(simultaneous=False) == ["a", "b", "c", "d"]
    with caplog.at_level(logging.WARNING):
        assert mfa.compute_all_possible() == ["a", "b", "c", "d"]
    # the total is not determined, rather than singular
    assert not messages(caplog, logging.WARNING)
    assert not mfa.flows["sysenv => a"].is_set


def test_share_zero_for_some_items_leaves_process_uncomputed():
    mfa = make_mfa(
        processes=["sysenv", fd.ProcessDefinition(name="a", outflow_shares={"b": "z"}), "b", "c"],
        flows=[
            flow("sysenv", "a"),
            flow("a", "b", prescribed="known"),
            flow("a", "c"),
            flow("b", "sysenv"),
            flow("c", "sysenv"),
        ],
        parameter_values={"z": np.array([0.0, 0.5, 0.5, 0.5]), "known": 1.0},
        parameter_dims={"z": ("t",), "known": ("t",)},
    )
    assert mfa.compute_all_possible() == ["a", "c"]
    assert not mfa.flows["a => c"].is_set and not mfa.flows["sysenv => a"].is_set


# Stocks and errors which are not raised ---------------------------------------------------------


def loop_with_stock_mfa(subclass, stock_values=None, **stock_kwargs) -> fd.MFASystem:
    """Recycling loop with a dynamic stock in use, whose lifetime is not set yet."""
    return loop_mfa(
        dimensions=COHORT_DIMENSIONS,
        stocks=[
            fd.StockDefinition(
                name="use",
                process="use",
                dim_letters=("t",),
                subclass=subclass,
                lifetime_model_class=fd.ExponentialLifetime,
                **stock_kwargs,
            )
        ],
    )


def test_inflow_driven_stock_in_loop_with_lifetime_not_set():
    mfa = loop_with_stock_mfa(fd.InflowDrivenDSM)
    before = array_state(mfa)
    assert mfa.compute_all_possible() == LOOP_PROCESSES
    assert_state_equal(array_state(mfa), before)
    with pytest.raises(fd.UnderdeterminedError, match="left out") as error:
        mfa.compute_all_possible(on_underdetermined="error")
    assert "inflow-driven stock 'use' is not known" in str(error.value)
    assert "Make the stock stock-driven, prescribe its inflow" in str(error.value)


def two_inflow_stock_loop() -> fd.MFASystem:
    """Loop production -> use -> eol -> rec -> use, with a stock-driven stock in use.

    use has two inflows without share, from production and rec, so it is not computed one
    process at a time, although its stock can be computed on its own.
    """
    return make_mfa(
        processes=[
            "sysenv",
            "production",
            "use",
            fd.ProcessDefinition(name="eol", outflow_shares={"rec": "collection"}),
            fd.ProcessDefinition(name="rec", outflow_shares={"use": "yield"}),
        ],
        flows=[
            flow("sysenv", "production"),
            flow("production", "use"),
            flow("use", "eol"),
            flow("eol", "rec"),
            flow("eol", "sysenv"),
            flow("rec", "use"),
            flow("rec", "sysenv"),
        ],
        parameter_values={"collection": 0.8, "yield": 0.5, "stock": np.array([1.0, 2.0, 3.0, 4.0])},
        parameter_dims={"collection": (), "yield": (), "stock": ("t",)},
        dimensions=COHORT_DIMENSIONS,
        stocks=[
            fd.StockDefinition(
                name="use",
                process="use",
                dim_letters=("t",),
                subclass=fd.StockDrivenDSM,
                lifetime_model_class=fd.FixedLifetime,
                prescribed_stock="stock",
            )
        ],
    )


def test_stock_computable_on_its_own_inside_block():
    mfa = two_inflow_stock_loop()
    mfa.stocks["use"].lifetime_model.mean = 1
    assert mfa.compute_all_possible(simultaneous=False) == LOOP_PROCESSES
    mfa.mark_all_unset()
    assert mfa.compute_all_possible() == []
    stock = mfa.stocks["use"]
    np.testing.assert_allclose(mfa.flows["use => eol"].values, stock.outflow.values)
    np.testing.assert_allclose(mfa.flows["rec => use"].values, 0.4 * stock.outflow.values)
    np.testing.assert_allclose(
        mfa.flows["production => use"].values, stock.inflow.values - 0.4 * stock.outflow.values
    )
    assert mfa.check_mass_balance()


def test_stock_computable_on_its_own_raising_is_left_out():
    mfa = two_inflow_stock_loop()
    before = array_state(mfa)
    assert mfa.compute_all_possible() == LOOP_PROCESSES
    assert_state_equal(array_state(mfa), before)
    with pytest.raises(fd.UnderdeterminedError, match="must be set before use"):
        mfa.compute_all_possible(on_underdetermined="error")


def test_simple_stock_receiving_losses_of_loop():
    mfa = loop_mfa(
        extra_processes=["landfill"],
        extra_flows=[flow("eol", "landfill")],
        removed_flows=[("eol", "sysenv")],
        stocks=[
            fd.StockDefinition(
                name="landfill",
                process="landfill",
                dim_letters=("t",),
                subclass=fd.SimpleFlowDrivenStock,
            )
        ],
    )
    assert mfa.compute_all_possible() == []
    losses = 0.2 * PRIMARY / 0.6
    np.testing.assert_allclose(mfa.flows["eol => landfill"].values, losses)
    np.testing.assert_allclose(mfa.stocks["landfill"].stock.values, np.cumsum(losses))
    np.testing.assert_allclose(mfa.stocks["landfill"].outflow.values, 0.0)
    assert mfa.check_mass_balance()


OUTFLOW = np.array([5.0, 10.0, 10.0, 20.0])


def simple_stock_loop(subclass=fd.SimpleFlowDrivenStock, **stock_kwargs) -> fd.MFASystem:
    """Recycling loop with a simple stock in use, by default with a prescribed outflow."""
    stock_kwargs.setdefault("prescribed_outflow", "outflow")
    return loop_mfa(
        stocks=[
            fd.StockDefinition(
                name="use", process="use", dim_letters=("t",), subclass=subclass, **stock_kwargs
            )
        ],
        extra_parameters={"outflow": (OUTFLOW, ("t",))},
    )


def test_simple_stock_inside_block():
    mfa = simple_stock_loop()
    assert mfa.compute_all_possible(simultaneous=False) == LOOP_PROCESSES
    mfa.mark_all_unset()
    assert mfa.compute_all_possible() == []
    inflow = PRIMARY + 0.4 * OUTFLOW
    np.testing.assert_allclose(mfa.flows["use => eol"].values, OUTFLOW)
    np.testing.assert_allclose(mfa.flows["production => use"].values, inflow)
    np.testing.assert_allclose(mfa.stocks["use"].stock.values, np.cumsum(inflow - OUTFLOW))
    assert mfa.check_mass_balance()


def test_simple_stock_without_outflow_in_loop_asks_for_it():
    mfa = simple_stock_loop(prescribed_outflow=None)
    with pytest.raises(fd.UnderdeterminedError, match="Give a value for its outflow"):
        mfa.compute_all_possible(on_underdetermined="error")


def test_stock_with_active_splitter_inside_block():
    """The outflow of use is split by region, while the stock in use is not."""
    split = np.array([0.3, 0.7])
    mfa = make_mfa(
        processes=[
            "sysenv",
            "production",
            fd.ProcessDefinition(name="use", forward_dimension_splitter="split"),
            fd.ProcessDefinition(name="eol", outflow_shares={"rec": "collection"}),
            fd.ProcessDefinition(name="rec", outflow_shares={"production": "yield"}),
        ],
        flows=[
            flow("sysenv", "production", prescribed="primary"),
            flow("production", "use"),
            flow("use", "eol", dims=("t", "r")),
            flow("eol", "rec", dims=("t", "r")),
            flow("eol", "sysenv", dims=("t", "r")),
            flow("rec", "production", dims=("t", "r")),
            flow("rec", "sysenv", dims=("t", "r")),
        ],
        parameter_values={
            "primary": PRIMARY,
            "outflow": OUTFLOW,
            "split": split,
            "collection": 0.8,
            "yield": 0.5,
        },
        parameter_dims={
            "primary": ("t",),
            "outflow": ("t",),
            "split": ("r",),
            "collection": (),
            "yield": (),
        },
        stocks=[
            fd.StockDefinition(
                name="use",
                process="use",
                dim_letters=("t",),
                subclass=fd.SimpleFlowDrivenStock,
                prescribed_outflow="outflow",
            )
        ],
    )
    assert mfa.compute_all_possible() == []
    np.testing.assert_allclose(mfa.flows["use => eol"].values, np.outer(OUTFLOW, split))
    np.testing.assert_allclose(mfa.flows["production => use"].values, PRIMARY + 0.4 * OUTFLOW)
    assert mfa.check_mass_balance()


class StockOverridingCompute(fd.SimpleFlowDrivenStock):
    def compute(self):
        super().compute()


class StockOverridingComputeProcess(fd.SimpleFlowDrivenStock):
    def _compute_process(self, computation):
        return fd.Stock._compute_process(self, computation)


class StockOverridingBoth(StockOverridingComputeProcess):
    """Overrides the computation of its process and the preparation for the simultaneous one."""

    def _prepare_block_computation(self):
        pass


@pytest.mark.parametrize(
    "subclass, method",
    [(StockOverridingCompute, "compute"), (StockOverridingComputeProcess, "_compute_process")],
)
def test_unsupported_stock_subclasses_are_left_out(subclass, method):
    mfa = simple_stock_loop(subclass)
    assert mfa.compute_all_possible() == LOOP_PROCESSES
    with pytest.raises(fd.UnderdeterminedError, match=f"overrides {method}"):
        mfa.compute_all_possible(on_underdetermined="error")


def test_stock_subclass_overriding_the_preparation_takes_part():
    mfa = simple_stock_loop(StockOverridingBoth)
    assert mfa.compute_all_possible() == []
    np.testing.assert_allclose(mfa.flows["use => eol"].values, OUTFLOW)


# Left-out processes -------------------------------------------------------------------------


def test_process_needing_both_splitters_is_left_out_and_computed_afterwards(caplog):
    """sorting receives the losses of the loop by region and splits them into goods."""
    region_by_good = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    forward = region_by_good / region_by_good.sum(axis=1, keepdims=True)
    backward = region_by_good / region_by_good.sum(axis=0, keepdims=True)
    mfa = loop_mfa(
        primary=np.ones((4, 2)),
        dims=("t", "r"),
        dimensions=GOOD_DIMENSIONS,
        extra_processes=[
            fd.ProcessDefinition(
                name="sorting", forward_dimension_splitter="fwd", backward_dimension_splitter="bwd"
            )
        ],
        extra_flows=[
            flow("eol", "sorting", dims=("t", "r")),
            flow("sorting", "sysenv", dims=("t", "g")),
        ],
        removed_flows=[("eol", "sysenv")],
        extra_parameters={"fwd": (forward, ("r", "g")), "bwd": (backward, ("r", "g"))},
    )
    with caplog.at_level(logging.INFO):
        fd.compute_processes(mfa.processes.values())
    assert all(p.is_computed for p in mfa.processes.values() if p.id != 0)
    losses = np.full((4, 2), 0.2 / 0.6)
    np.testing.assert_allclose(mfa.flows["sorting => sysenv"].values, losses @ forward)
    assert mfa.check_mass_balance()


def test_process_needing_both_splitters_is_reported():
    mfa = make_mfa(
        processes=[
            "sysenv",
            fd.ProcessDefinition(
                name="a", forward_dimension_splitter="fwd", backward_dimension_splitter="bwd"
            ),
            fd.ProcessDefinition(name="b", outflow_shares={"a": "back"}),
        ],
        flows=[
            flow("sysenv", "a", prescribed="input", dims=("t", "r")),
            flow("b", "a", dims=("t", "r")),
            flow("a", "b", dims=("t", "g")),
            flow("b", "sysenv", dims=("t", "g")),
        ],
        parameter_values={"input": 1.0, "fwd": 1 / 3, "bwd": 0.5, "back": 0.5},
        parameter_dims={"input": ("t", "r"), "fwd": ("r", "g"), "bwd": ("r", "g"), "back": ()},
        dimensions=GOOD_DIMENSIONS,
    )
    with pytest.raises(fd.UnderdeterminedError, match="Both dimension splitters would be needed"):
        mfa.compute_all_possible(on_underdetermined="error")


def test_share_with_dimensions_the_total_lacks_is_left_out():
    mfa = loop_mfa(recycling_yield=np.array([0.4, 0.5]), parameter_dims={"yield": ("r",)})
    with pytest.raises(fd.UnderdeterminedError) as error:
        mfa.compute_all_possible(on_underdetermined="error")
    assert "Process 'rec' is left out" in str(error.value)
    assert "which the total of the outflows does not have" in str(error.value)


@pytest.mark.parametrize("related", ["parent", "sibling"])
def test_related_dimensions_are_left_out(related):
    regions = fd.Dimension(name="Region", letter="r", items=["EU", "US"])
    countries = fd.Dimension(
        name="Country",
        letter="k",
        items=["DE", "FR", "US"],
        parent=regions,
        parent_items=["EU", "EU", "US"],
    )
    sub_regions = fd.Dimension(
        name="Sub-region",
        letter="s",
        items=["EU-W", "EU-E", "US"],
        parent=regions,
        parent_items=["EU", "EU", "US"],
    )
    time_dim = fd.Dimension(name="Time", letter="t", items=TIME)
    dims = fd.DimensionSet(dim_list=[time_dim, regions, countries, sub_regions])
    coarse = "r" if related == "parent" else "s"
    processes = fd.make_processes(
        ["sysenv", "p", fd.ProcessDefinition(name="q", outflow_shares={"p": "yield"})]
    )
    flow_dims = {
        ("sysenv", "p"): ("t", "k"),
        ("p", "q"): ("t", coarse),
        ("q", "p"): ("t", "k"),
        ("q", "sysenv"): ("t", "k"),
    }
    flow_definitions = [
        fd.FlowDefinition(from_process_name=a, to_process_name=b, dim_letters=letters)
        for (a, b), letters in flow_dims.items()
    ]
    flows = fd.make_empty_flows(processes=processes, flow_definitions=flow_definitions, dims=dims)
    parameters = {"yield": fd.Parameter(name="yield", dims=dims[()], values=np.array(0.5))}
    mfa = fd.MFASystem(dims=dims, processes=processes, flows=flows, parameters=parameters)
    mfa.link_parameters()
    mfa.flows["sysenv => p"][...] = 1.0
    with pytest.raises(fd.UnderdeterminedError, match="are related"):
        mfa.compute_all_possible(on_underdetermined="error")


# Skipped blocks ---------------------------------------------------------------------------------


def count_calls(monkeypatch, module, name: str) -> list:
    """Count the calls of a function of a module, which are recorded in the returned list."""
    calls = []
    original = getattr(module, name)

    def counting(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, counting)
    return calls


def test_block_without_known_values_is_not_solved(monkeypatch):
    solved = count_calls(monkeypatch, process_system, "solve")
    mfa = loop_mfa(primary=None)
    before = array_state(mfa)
    assert mfa.compute_all_possible() == LOOP_PROCESSES
    assert solved == []
    assert_state_equal(array_state(mfa), before)
    with pytest.raises(fd.UnderdeterminedError, match="none of their flows and stock arrays"):
        mfa.compute_all_possible(on_underdetermined="error")


def test_unchanged_block_is_not_solved_twice(monkeypatch):
    """The second loop is underdetermined; it is not solved again after the first loop."""
    solved = count_calls(monkeypatch, process_system, "solve")
    mfa = two_loops_mfa()
    mfa.processes["b_eol"].remove_outflow_share("b_rec")
    assert mfa.compute_all_possible() == ["b_" + name for name in LOOP_PROCESSES]
    assert len(solved) == 2


def test_loop_with_many_items_is_factorised_once(monkeypatch):
    """Guards against a loop over the 2e5 items: one factorisation for all of them."""
    factorisations = count_calls(monkeypatch, linear_system, "splu")
    shape = (200, 1000)
    rng = np.random.default_rng(2)
    primary = rng.uniform(1.0, 2.0, shape)
    collection = rng.uniform(0.3, 0.9, shape)
    mfa = loop_mfa(
        collection=collection,
        primary=primary,
        dims=("t", "r"),
        parameter_dims={"collection": ("t", "r")},
        dim_items={"Time": list(range(shape[0])), "Region": [f"r{i}" for i in range(shape[1])]},
    )
    assert mfa.compute_all_possible() == []
    assert len(factorisations) == 1
    np.testing.assert_allclose(
        mfa.flows["production => use"].values, primary / (1 - 0.5 * collection)
    )


def test_not_enough_memory_is_reported(monkeypatch):
    def solve_without_memory(*args, **kwargs):
        raise MemoryError

    monkeypatch.setattr(process_system, "solve", solve_without_memory)
    mfa = loop_mfa()
    with pytest.raises(fd.UnderdeterminedError, match="Not enough memory to solve the equations"):
        mfa.compute_all_possible(on_underdetermined="error")
    assert set_flows(mfa) == ["sysenv => production"]


@pytest.mark.parametrize("limit", [10, 30])
def test_block_above_size_limit_is_not_solved(limit, monkeypatch):
    # 24 items of unknown flows, 40 unknowns with the totals
    monkeypatch.setattr(process_system, "MAX_BLOCK_UNKNOWNS", limit)
    mfa = loop_mfa()
    with pytest.raises(fd.UnderdeterminedError, match=f"more than the limit of {limit}"):
        mfa.compute_all_possible(on_underdetermined="error")
    assert not mfa.flows["production => use"].is_set


# Inconsistent values and negative results ------------------------------------------------------

# With a collection rate of one in the second year, the remainder of eol is zero there, so that
# eol cannot be computed on its own, although its outflow to sysenv is known.
COLLECTION_WITH_ONE = np.array([0.8, 1.0, 0.8, 0.8])
THROUGHPUT_WITH_ONE = PRIMARY / (1 - 0.5 * COLLECTION_WITH_ONE)


def loop_with_known_losses(losses) -> fd.MFASystem:
    return loop_mfa(
        collection=COLLECTION_WITH_ONE,
        parameter_dims={"collection": ("t",)},
        removed_flows=[("eol", "sysenv")],
        extra_flows=[flow("eol", "sysenv", prescribed="losses")],
        extra_parameters={"losses": (losses, ("t",))},
    )


def test_consistent_additional_value_is_accepted():
    mfa = loop_with_known_losses((1 - COLLECTION_WITH_ONE) * THROUGHPUT_WITH_ONE)
    assert mfa.compute_all_possible(simultaneous=False) == LOOP_PROCESSES
    assert mfa.compute_all_possible() == []
    np.testing.assert_allclose(mfa.flows["production => use"].values, THROUGHPUT_WITH_ONE)
    assert mfa.check_mass_balance()


def test_inconsistent_values_raise_and_restore():
    mfa = loop_with_known_losses(np.array([1.0, 0.0, 1.0, 1.0]))
    mfa.compute_all_possible(simultaneous=False)
    before = array_state(mfa)
    with pytest.raises(ValueError, match="overdetermined") as error:
        mfa.compute_all_possible()
    assert "Not fulfilled: remainder of the outflow 'eol => sysenv' of process 'eol'" in str(
        error.value
    )
    assert "(max error 12.3 at Time 2003)" in str(error.value)
    assert_state_equal(array_state(mfa), before)


def test_inconsistent_values_with_warnings_are_computed(caplog):
    mfa = loop_with_known_losses(np.array([1.0, 0.0, 1.0, 1.0]))
    with (
        # the inconsistency also shows in the mass balance of eol, as for one process at a time
        fd.config.override(
            error_behaviors={"process_overdetermined": "warn", "mass_balance": "warn"}
        ),
        caplog.at_level(logging.WARNING),
    ):
        assert mfa.compute_all_possible() == []
    assert "overdetermined" in caplog.text


def test_negative_results_are_not_checked(caplog):
    """rec also sends a known amount to exports, so that the rest to sysenv is negative in the
    first year. As for one process at a time, check_flows() reports it."""
    mfa = loop_mfa(
        extra_processes=["exports"],
        extra_flows=[flow("rec", "exports", prescribed="exports"), flow("exports", "sysenv")],
        extra_parameters={"exports": (10.0, ("t",))},
    )
    with caplog.at_level(logging.WARNING):
        assert mfa.compute_all_possible() == []
    assert not messages(caplog, logging.WARNING)
    np.testing.assert_allclose(mfa.flows["rec => sysenv"].values, 0.4 * PRIMARY / 0.6 - 10.0)
    assert not mfa.check_flows(error_behavior="info")


# Agreement with the computation one process at a time -----------------------------------------


def unprescribe(mfa: fd.MFASystem, name: str) -> None:
    mfa.flows[name]._set_prescribed(None)
    mfa.flows[name].mark_unset()


def split_backward():
    mfa = split_mfa()
    unprescribe(mfa, "sysenv => a")
    mfa.flows["a => c"][...] = 7.0
    return mfa


def forward_splitter_backward():
    mfa = splitter_mfa(np.array([0.2, 0.8]), forward_dimension_splitter="split")
    unprescribe(mfa, "sysenv => a")
    mfa.flows["a => b"][...] = 1.0
    return mfa


def simple_stock_process():
    mfa = stock_mfa(fd.SimpleFlowDrivenStock, lifetime=False)
    mfa.flows["sysenv => use"][...] = 2.0
    mfa.flows["use => sysenv"][...] = 0.4
    return mfa


def stock_driven_stock_process():
    mfa = stock_mfa(fd.StockDrivenDSM)
    mfa.stocks["use"].stock[...] = 1.0
    return mfa


SEQUENTIAL_SYSTEMS = {
    "split forward": split_mfa,
    "split backward": split_backward,
    "forward splitter": lambda: splitter_mfa(
        np.array([0.2, 0.8]), forward_dimension_splitter="split"
    ),
    "forward splitter used backward": forward_splitter_backward,
    "simple stock": simple_stock_process,
    "stock-driven stock": stock_driven_stock_process,
}


@pytest.mark.parametrize("make", SEQUENTIAL_SYSTEMS.values(), ids=SEQUENTIAL_SYSTEMS.keys())
def test_all_processes_solved_together_equal_one_at_a_time(make):
    """All processes of the system solved as one block, rather than single processes as in the
    tests of test_process_compute.py with the engine fixture."""
    sequential = make()
    fd.compute_processes(sequential.processes.values(), simultaneous=False)
    assert all(p.is_computed for p in sequential.processes.values() if p.id != 0)

    simultaneous = make()
    # solve_block is private; it solves the processes together although they need not be
    process_system.solve_block([p for p in simultaneous.processes.values() if p.id != 0])
    assert all(p.is_computed for p in simultaneous.processes.values() if p.id != 0)
    for name, sequential_flow in sequential.flows.items():
        np.testing.assert_allclose(
            simultaneous.flows[name].values, sequential_flow.values, atol=1e-12, err_msg=name
        )
    for name, stock in sequential.stocks.items():
        for array in ("inflow", "outflow", "stock"):
            np.testing.assert_allclose(
                getattr(simultaneous.stocks[name], array).values, getattr(stock, array).values
            )


def example_mfa_with_shares() -> fd.MFASystem:
    """ExampleMFA with the shares of its processes, as in example 2."""
    mfa = get_example_mfa()
    parameters = mfa.parameters
    mfa.processes["shredder"].set_outflow_share("remelting", parameters["shredder yield"])
    mfa.processes["demolition"].set_outflow_share("remelting", parameters["demolition yield"])
    mfa.processes["remelting"].set_outflow_share("sysenv", parameters["remelting yield"])
    mfa.flows["sysenv => shredder"][...] = (
        parameters["eol machines"] * parameters["composition eol machines"]
    )
    mfa.flows["sysenv => demolition"][...] = (
        parameters["eol buildings"] * parameters["composition eol buildings"]
    )
    return mfa


def test_example_mfa_is_the_same_with_and_without_simultaneous_computation():
    manual = get_example_mfa()
    manual.compute()
    for simultaneous in (True, False):
        mfa = example_mfa_with_shares()
        assert mfa.compute_all_possible(simultaneous=simultaneous) == []
        for name, manual_flow in manual.flows.items():
            np.testing.assert_allclose(mfa.flows[name].values, manual_flow.values, err_msg=name)


def mixed_dims_mfa(splitter: bool) -> fd.MFASystem:
    """Inflows 'fine' (t, r) with share 0.5 and 'coarse' (t) without share, outflow (t, r)."""
    kwargs = {"forward_dimension_splitter": "split"} if splitter else {}
    mfa = make_mfa(
        processes=[
            "sysenv",
            fd.ProcessDefinition(name="a", inflow_shares={"fine": "share"}, **kwargs),
        ],
        flows=[
            flow("sysenv", "a", dims=("t", "r"), name_override="fine"),
            flow("sysenv", "a", name_override="coarse"),
            flow("a", "sysenv", dims=("t", "r"), name_override="out"),
        ],
        parameter_values={"share": 0.5, "split": np.array([0.2, 0.8])},
        parameter_dims={"share": (), "split": ("r",)},
    )
    mfa.flows["fine"][...] = np.array([[1.0, 3.0]] * 4)
    return mfa


@pytest.mark.parametrize("splitter, simultaneous_result", [(False, [2.0, 6.0]), (True, [1.6, 6.4])])
def test_known_limitation_mixed_dimensions_on_one_side(splitter, simultaneous_result):
    """Known limitation: the block takes the dimensions common to the inflows as the total.

    The computation one process at a time takes the total from 'fine' and keeps its region
    dimension, so it needs no splitter. The block equations apply the splitter instead.
    """
    sequential = mixed_dims_mfa(splitter)
    with fd.config.override(error_behaviors={"unused_dimension_splitter": "ignore"}):
        sequential.processes["a"].compute()
    np.testing.assert_allclose(sequential.flows["out"].values[0], [2.0, 6.0])
    simultaneous = mixed_dims_mfa(splitter)
    # solve_block is private; the process is computed one at a time otherwise
    process_system.solve_block([simultaneous.processes["a"]])
    np.testing.assert_allclose(simultaneous.flows["out"].values[0], simultaneous_result)
    np.testing.assert_allclose(simultaneous.flows["coarse"].values, 4.0)


# API ---------------------------------------------------------------------------------------


def test_underdetermined_error_for_single_process():
    process = fd.Process(name="a", id=1)
    error = fd.UnderdeterminedError(process, "Some reason.\n")
    assert error.process_name == "a"
    assert error.process_names == ["a"]
    assert error.flow_names == []
    assert error.reason == "Some reason.\n"
    assert str(error).startswith("Cannot compute process 'a' (ID 1), as it is underdetermined.\n")
    assert "compute_all_possible" not in str(error)


def test_process_compute_points_to_simultaneous_computation_for_flows_only():
    mfa = loop_mfa()
    with pytest.raises(fd.UnderdeterminedError, match="can only be computed together"):
        mfa.processes["eol"].compute()
    mfa = stock_mfa(fd.StockDrivenDSM)
    mfa.flows["sysenv => use"][...] = 1.0
    with pytest.raises(fd.UnderdeterminedError) as error:
        mfa.processes["use"].compute()
    assert "can only be computed together" not in str(error.value)


def test_invalid_on_underdetermined_names_the_valid_values():
    mfa = loop_mfa()
    with pytest.raises(ValueError, match="Valid values are 'error', 'warn', 'info', 'ignore'"):
        mfa.compute_all_possible(on_underdetermined="warning")
    assert not mfa.flows["production => use"].is_set


def test_options_are_keyword_only():
    mfa = loop_mfa()
    with pytest.raises(TypeError):
        mfa.compute_all_possible(False)
    with pytest.raises(TypeError):
        fd.compute_processes(mfa.processes.values(), False)


def test_assemble_processes():
    mfa = loop_mfa()
    processes = [p for p in mfa.processes.values() if p.id != 0]
    system = assemble_processes([ProcessEquations(process) for process in processes])
    # 6 unknown flows and 4 totals, 10 equations, each over the 4 years
    assert system.matrix.shape == (40, 40)


# Values which are not finite and wording of the reports ----------------------------------------


def test_nan_share_in_loop_is_reported_without_error(caplog):
    collection = np.array([0.8, np.nan, 0.8, 0.8])
    mfa = loop_mfa(collection=collection, parameter_dims={"collection": ("t",)})
    with caplog.at_level(logging.INFO):
        assert mfa.compute_all_possible() == LOOP_PROCESSES
    assert "The given values 'collection' are not finite (NaN or infinite)" in caplog.text
    assert "e.g. at Time 2001" in caplog.text
    assert set_flows(mfa) == ["sysenv => production"]


def test_infinite_prescribed_flow_in_loop_is_reported():
    mfa = loop_mfa(primary=np.array([10.0, np.inf, 30.0, 40.0]))
    with pytest.raises(fd.UnderdeterminedError) as error:
        mfa.compute_all_possible(on_underdetermined="error")
    message = str(error.value)
    assert "The given values 'sysenv => production' are not finite" in message
    assert "loop without exit" not in message


def test_nearly_singular_loop_counts_as_singular(caplog):
    mfa = loop_mfa(collection=1.0 - 1e-15, recycling_yield=1.0)
    with caplog.at_level(logging.WARNING):
        assert mfa.compute_all_possible() == LOOP_PROCESSES
    assert "loop without exit" in caplog.text


def test_chain_of_processes_is_not_called_a_loop():
    mfa = make_mfa(
        ["sysenv", "a", "b"],
        [flow("sysenv", "a"), flow("a", "b"), flow("a", "sysenv"), flow("b", "sysenv")],
        {},
        {},
    )
    with pytest.raises(fd.UnderdeterminedError) as error:
        mfa.compute_all_possible(on_underdetermined="error")
    assert "Processes 'a', 'b' are connected by unknown flows" in str(error.value)
    assert "depend on each other" not in str(error.value)


def test_not_finite_values_are_reported_as_warning(caplog):
    mfa = loop_mfa(primary=np.array([10.0, np.nan, 30.0, 40.0]))
    with caplog.at_level(logging.WARNING):
        assert mfa.compute_all_possible() == LOOP_PROCESSES
    warnings = messages(caplog, logging.WARNING)
    assert len(warnings) == 1 and "'sysenv => production' are not finite" in warnings[0]
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        mfa.compute_all_possible(on_underdetermined="ignore")
    assert not messages(caplog, logging.WARNING)


def test_process_compute_hint_only_for_processes_on_a_cycle():
    """sysenv does not close the chain sysenv -> a -> b -> sysenv into a cycle."""
    mfa = make_mfa(
        ["sysenv", "a", "b"],
        [flow("sysenv", "a"), flow("a", "b"), flow("b", "sysenv")],
        {},
        {},
    )
    with pytest.raises(fd.UnderdeterminedError) as error:
        mfa.processes["a"].compute()
    assert "can only be computed together" not in str(error.value)


def test_process_compute_message_ends_with_config_hint():
    mfa = loop_mfa()
    with pytest.raises(fd.UnderdeterminedError) as error:
        mfa.processes["eol"].compute()
    assert "compute_all_possible(). Configure with" in str(error.value)
