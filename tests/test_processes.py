import copy

import pytest

from flodym import (
    Dimension,
    DimensionSet,
    Flow,
    MFASystem,
    Process,
    ProcessDefinition,
    SimpleFlowDrivenStock,
    make_processes,
)
from flodym.example_objects import get_example_mfa


def test_make_processes_from_names():
    processes = make_processes(["sysenv", "a", "b"])
    assert {name: p.id for name, p in processes.items()} == {"sysenv": 0, "a": 1, "b": 2}


def test_make_processes_non_contiguous_ids():
    processes = make_processes(
        [ProcessDefinition(id=0, name="sysenv"), ProcessDefinition(id=10, name="a"), "b", "c"]
    )
    assert {name: p.id for name, p in processes.items()} == {"sysenv": 0, "a": 10, "b": 1, "c": 2}


def test_make_processes_names_skip_explicit_ids():
    processes = make_processes(["sysenv", "a", ProcessDefinition(id=1, name="b")])
    assert {name: p.id for name, p in processes.items()} == {"sysenv": 0, "a": 2, "b": 1}


@pytest.mark.parametrize(
    "definitions, match",
    [
        ([ProcessDefinition(id=0, name="sysenv"), ProcessDefinition(id=0, name="a")], "unique"),
        ([ProcessDefinition(id=-1, name="a"), "sysenv"], "non-negative"),
        (["sysenv", "a", "a"], "more than once"),
        ([ProcessDefinition(id=0, name="a")], "sysenv"),
    ],
)
def test_make_processes_errors(definitions, match):
    with pytest.raises(ValueError, match=match):
        make_processes(definitions)


def test_mfa_system_links_processes():
    mfa = get_example_mfa()
    for process in mfa.processes.values():
        for name, flow in process.inflows.items():
            assert mfa.flows[name] is flow and flow.to_process is process
        for name, flow in process.outflows.items():
            assert mfa.flows[name] is flow and flow.from_process is process
    linked_flows = [f for p in mfa.processes.values() for f in p.inflows.values()]
    assert len(linked_flows) == len(mfa.flows)
    for stock in mfa.stocks.values():
        assert stock.process is not None and stock.process.stock is stock


def test_new_flow_does_not_replace_link():
    mfa = get_example_mfa()
    name = "sysenv => shredder"
    Flow.from_array(mfa.flows[name], name="derived")
    assert mfa.processes["shredder"].inflows[name] is mfa.flows[name]
    assert "derived" not in mfa.processes["shredder"].inflows


def _two_flow_mfa(second_stock: bool = False) -> MFASystem:
    dims = DimensionSet(dim_list=[Dimension(name="time", letter="t", items=[2000, 2001, 2002])])
    processes = make_processes(["sysenv", "a"])
    flows = {
        name: Flow(
            from_process=processes["sysenv"], to_process=processes["a"], dims=dims, name=name
        )
        for name in ("f1", "f2")
    }
    n_stocks = 2 if second_stock else 1
    stocks = {
        f"s{i}": SimpleFlowDrivenStock(dims=dims, process=processes["a"], name=f"s{i}")
        for i in range(n_stocks)
    }
    return MFASystem(dims=dims, parameters={}, processes=processes, flows=flows, stocks=stocks)


def test_parallel_flows_are_linked_separately():
    mfa = _two_flow_mfa()
    assert list(mfa.processes["a"].inflows) == ["f1", "f2"]
    assert list(mfa.processes["sysenv"].outflows) == ["f1", "f2"]


def test_second_stock_of_process_raises():
    with pytest.raises(ValueError, match="cannot have both stocks"):
        _two_flow_mfa(second_stock=True)


def test_flow_with_foreign_process_raises():
    mfa = _two_flow_mfa()
    other = Process(name="a", id=1)
    flows = dict(mfa.flows)
    flows["f3"] = Flow(from_process=mfa.processes["sysenv"], to_process=other, dims=mfa.dims)
    with pytest.raises(ValueError, match="not a process of the MFA system"):
        MFASystem(
            dims=mfa.dims, parameters={}, processes=mfa.processes, flows=flows, stocks=mfa.stocks
        )


def test_process_repr_and_dump():
    mfa = _two_flow_mfa()
    a = mfa.processes["a"]
    assert repr(a) == "Process(name='a', id=1, inflows=['f1', 'f2'], outflows=[], stock='s0')"
    assert "inflows=['f1', 'f2']" in str(a)
    assert a.model_dump() == {"name": "a", "id": 1}
    repr(mfa.stocks["s0"])


def test_links_follow_changes_of_flows():
    mfa = _two_flow_mfa()
    a = mfa.processes["a"]
    del mfa.flows["f1"]
    mfa.flows["f3"] = Flow(from_process=mfa.processes["sysenv"], to_process=a, dims=mfa.dims)
    mfa.check_mass_balance(error_behavior="ignore")
    assert list(a.inflows) == ["f2", "f3"]
    assert list(mfa.processes["sysenv"].outflows) == ["f2", "f3"]
    assert a.inflows["f3"] is mfa.flows["f3"]


def test_links_follow_changes_of_stocks():
    mfa = _two_flow_mfa()
    a = mfa.processes["a"]
    old_stock = mfa.stocks["s0"]
    mfa.stocks["s0"] = SimpleFlowDrivenStock(dims=mfa.dims, process=a, name="s0")
    mfa.compute_all_possible()
    assert a.stock is mfa.stocks["s0"] and a.stock is not old_stock
    del mfa.stocks["s0"]
    mfa.check_mass_balance(error_behavior="ignore")
    assert a.stock is None


def test_links_follow_changed_processes_of_a_flow():
    mfa = _named_flows_mfa()
    mfa.flows["sysenv => a"].to_process = mfa.processes["b"]
    mfa.check_mass_balance(error_behavior="ignore")
    assert list(mfa.flows) == ["sysenv => a", "custom", "b => sysenv"]
    assert list(mfa.processes["b"].inflows) == ["sysenv => a", "custom"]
    assert list(mfa.processes["a"].inflows) == []


def test_link_parameters_follows_changes_of_flows():
    mfa = _two_flow_mfa()
    a = mfa.processes["a"]
    mfa.flows["f3"] = Flow(from_process=mfa.processes["sysenv"], to_process=a, dims=mfa.dims)
    mfa.link_parameters()
    assert list(a.inflows) == ["f1", "f2", "f3"]


def test_process_under_a_second_key_keeps_its_links():
    mfa = _two_flow_mfa()
    a = mfa.processes["a"]
    mfa.processes["alias of a"] = a
    mfa.flows["f1"][...] = 1.0
    assert mfa.check_mass_balance(error_behavior="ignore") is False
    assert list(a.inflows) == ["f1", "f2"]
    assert a.stock is mfa.stocks["s0"]


def test_failed_linking_keeps_links():
    mfa = _two_flow_mfa()
    mfa.flows["f3"] = Flow(
        from_process=mfa.processes["sysenv"], to_process=Process(name="a", id=1), dims=mfa.dims
    )
    with pytest.raises(ValueError, match="not a process of the MFA system"):
        mfa.check_mass_balance()
    assert list(mfa.processes["a"].inflows) == ["f1", "f2"]


def test_flow_replaced_by_plain_array_raises():
    mfa = _two_flow_mfa()
    mfa.flows["f1"] = mfa.flows["f1"] * 3
    with pytest.raises(TypeError, match=r"Entry 'f1' .* is a FlodymArray, not a Flow"):
        mfa.check_mass_balance()
    assert list(mfa.processes["a"].inflows) == ["f1", "f2"]


def test_links_use_keys_of_flows():
    """Flows are linked under their key in the MFA system, also if they have no or equal names."""
    dims = DimensionSet(dim_list=[Dimension(name="time", letter="t", items=[2000, 2001])])
    processes = make_processes(["sysenv", "a"])
    flows = {
        key: Flow(from_process=processes["sysenv"], to_process=processes["a"], dims=dims)
        for key in ("f1", "f2")
    }
    mfa = MFASystem(dims=dims, parameters={}, processes=processes, flows=flows)
    assert [flow.name for flow in mfa.flows.values()] == ["unnamed", "unnamed"]
    assert list(mfa.processes["a"].inflows) == ["f1", "f2"]
    assert list(mfa.processes["sysenv"].outflows) == ["f1", "f2"]


def test_processes_shared_by_two_systems():
    """Each MFA system uses the links to its own flows, also if another one was set up later."""
    dims = DimensionSet(dim_list=[Dimension(name="time", letter="t", items=[2000, 2001])])
    processes = make_processes(["sysenv", "a"])

    def system(outflow_value: float) -> MFASystem:
        flows = {
            "in": Flow(from_process=processes["sysenv"], to_process=processes["a"], dims=dims),
            "out": Flow(from_process=processes["a"], to_process=processes["sysenv"], dims=dims),
        }
        flows["in"][...] = 1.0
        flows["out"][...] = outflow_value
        return MFASystem(dims=dims, parameters={}, processes=processes, flows=flows)

    unbalanced = system(outflow_value=0.5)
    balanced = system(outflow_value=1.0)
    assert unbalanced.check_mass_balance(error_behavior="ignore") is False
    assert processes["a"].outflows["out"] is unbalanced.flows["out"]
    assert balanced.check_mass_balance() is True
    assert processes["a"].outflows["out"] is balanced.flows["out"]


def test_stock_of_sysenv_raises():
    dims = DimensionSet(dim_list=[Dimension(name="time", letter="t", items=[2000, 2001, 2002])])
    processes = make_processes(["sysenv", "a"])
    stocks = {"sink": SimpleFlowDrivenStock(dims=dims, process=processes["sysenv"], name="sink")}
    with pytest.raises(ValueError, match="Stock 'sink' is connected to the process 'sysenv'"):
        MFASystem(dims=dims, parameters={}, processes=processes, flows={}, stocks=stocks)
    stocks["sink"].process = None
    mfa = MFASystem(dims=dims, parameters={}, processes=processes, flows={}, stocks=stocks)
    # also if the stock is connected to 'sysenv' after the setup
    stocks["sink"].process = processes["sysenv"]
    with pytest.raises(ValueError, match="Stock 'sink' is connected to the process 'sysenv'"):
        mfa.check_mass_balance()


def test_stock_replaced_by_stock_array_raises():
    mfa = _two_flow_mfa()
    mfa.stocks["s0"] = mfa.stocks["s0"].stock
    with pytest.raises(TypeError, match=r"Entry 's0' .* is a StockArray, not a Stock"):
        mfa.check_mass_balance()


@pytest.mark.parametrize("deep_copy", [copy.deepcopy, lambda mfa: mfa.model_copy(deep=True)])
def test_deep_copy_keeps_objects_connected(deep_copy):
    mfa = get_example_mfa()
    mfa.compute()
    mfa_copy = deep_copy(mfa)
    copied_processes = {id(process) for process in mfa_copy.processes.values()}
    assert copied_processes.isdisjoint({id(process) for process in mfa.processes.values()})
    for flow in mfa_copy.flows.values():
        assert id(flow.from_process) in copied_processes
        assert id(flow.to_process) in copied_processes
    for process in mfa_copy.processes.values():
        for name, flow in process.inflows.items():
            assert mfa_copy.flows[name] is flow
    assert mfa_copy.check_mass_balance() is True
    mfa_copy.flows["sysenv => shredder"].values[...] *= 2
    assert mfa_copy.check_mass_balance(error_behavior="ignore") is False
    assert mfa.check_mass_balance() is True


def _named_flows_mfa() -> MFASystem:
    from flodym import FlowDefinition, make_empty_flows

    dims = DimensionSet(dim_list=[Dimension(name="time", letter="t", items=[2000, 2001])])
    processes = make_processes(["sysenv", "a", "b"])
    flow_definitions = [
        FlowDefinition(from_process_name="sysenv", to_process_name="a", dim_letters=("t",)),
        FlowDefinition(
            from_process_name="a", to_process_name="b", dim_letters=("t",), name_override="custom"
        ),
        FlowDefinition(from_process_name="b", to_process_name="sysenv", dim_letters=("t",)),
    ]
    flows = make_empty_flows(processes, flow_definitions, dims)
    return MFASystem(dims=dims, parameters={}, processes=processes, flows=flows)


def test_setup_keeps_flow_names():
    import flodym
    from flodym import FlowDefinition, make_empty_flows
    from flodym.flow_naming import process_ids

    dims = DimensionSet(dim_list=[Dimension(name="time", letter="t", items=[2000])])
    processes = make_processes(["sysenv", "a"])
    flow_definitions = [
        FlowDefinition(from_process_name="sysenv", to_process_name="a", dim_letters=("t",))
    ]
    with flodym.config.override(flow_naming=process_ids):
        flows = make_empty_flows(processes, flow_definitions, dims)
    mfa = MFASystem(dims=dims, parameters={}, processes=processes, flows=flows)
    assert list(mfa.flows) == ["F0_1"]
