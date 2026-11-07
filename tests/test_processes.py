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
        assert stock.process.stock is stock


def test_new_flow_does_not_replace_link():
    mfa = get_example_mfa()
    name = "sysenv => shredder"
    Flow.from_array(mfa.flows[name], name="derived")
    assert mfa.processes["shredder"].inflows[name] is mfa.flows[name]
    assert "derived" not in mfa.processes["shredder"].inflows


def _two_flow_mfa(second_stock: bool = False) -> MFASystem:
    dims = DimensionSet(dim_list=[Dimension(name="time", letter="t", items=[2000, 2001])])
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


def test_remove_links():
    mfa = _two_flow_mfa()
    a = mfa.processes["a"]
    a.remove_inflow("f1")
    assert list(a.inflows) == ["f2"]
    with pytest.raises(ValueError, match="No inflow"):
        a.remove_inflow("f1")
    a.remove_stock()
    assert a.stock is None


def test_process_repr_and_dump():
    mfa = _two_flow_mfa()
    a = mfa.processes["a"]
    assert repr(a) == "Process(name='a', id=1, inflows=['f1', 'f2'], outflows=[], stock='s0')"
    assert "inflows=['f1', 'f2']" in str(a)
    assert a.model_dump() == {"name": "a", "id": 1}
    repr(mfa.stocks["s0"])


def test_relink():
    mfa = _two_flow_mfa()
    a = mfa.processes["a"]
    del mfa.flows["f1"]
    mfa.flows["f3"] = Flow(
        from_process=mfa.processes["sysenv"], to_process=a, dims=mfa.dims, name="f3"
    )
    assert list(a.inflows) == ["f1", "f2"]
    mfa.relink()
    assert list(a.inflows) == ["f2", "f3"]
    assert list(mfa.processes["sysenv"].outflows) == ["f2", "f3"]


def test_failed_relink_keeps_links():
    mfa = _two_flow_mfa()
    mfa.flows["f3"] = Flow(
        from_process=mfa.processes["sysenv"], to_process=Process(name="a", id=1), dims=mfa.dims
    )
    with pytest.raises(ValueError, match="not a process of the MFA system"):
        mfa.relink()
    assert list(mfa.processes["a"].inflows) == ["f1", "f2"]


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


def test_relink_updates_names_of_flows_named_after_processes():
    mfa = _named_flows_mfa()
    flows_dict = mfa.flows
    flow = mfa.flows["sysenv => a"]
    flow.to_process = mfa.processes["b"]
    mfa.relink()
    assert list(mfa.flows) == ["sysenv => b", "custom", "b => sysenv"]
    assert mfa.flows is flows_dict
    assert mfa.flows["sysenv => b"] is flow
    assert flow.name == "sysenv => b"
    assert list(mfa.processes["b"].inflows) == ["sysenv => b", "custom"]
    assert list(mfa.processes["a"].inflows) == []


def test_relink_keeps_custom_names_and_avoids_duplicates(caplog):
    mfa = _named_flows_mfa()
    custom = mfa.flows["custom"]
    custom.from_process = mfa.processes["sysenv"]
    # the third flow now connects the same processes as the first one
    mfa.flows["b => sysenv"].from_process = mfa.processes["sysenv"]
    mfa.flows["b => sysenv"].to_process = mfa.processes["a"]
    mfa.relink()
    assert list(mfa.flows) == ["sysenv => a", "custom", "sysenv => a__2"]
    assert custom.name == "custom"
    assert "already exists" in caplog.text


def test_relink_with_config_naming():
    import flodym
    from flodym.flow_naming import process_ids

    mfa = _named_flows_mfa()
    with flodym.config.override(flow_naming=process_ids):
        mfa.relink()
    assert list(mfa.flows) == ["F0_1", "custom", "F2_0"]


def test_relink_without_updating_names():
    mfa = _named_flows_mfa()
    mfa.flows["sysenv => a"].to_process = mfa.processes["b"]
    mfa.relink(update_names=False)
    assert list(mfa.flows) == ["sysenv => a", "custom", "b => sysenv"]
    assert list(mfa.processes["b"].inflows) == ["sysenv => a", "custom"]


def test_relink_renames_share_keys():
    import numpy as np

    from flodym import Parameter

    mfa = _named_flows_mfa()
    a, b = mfa.processes["a"], mfa.processes["b"]
    share = Parameter(dims=mfa.dims, values=np.full(mfa.dims.shape, 0.5), name="share")
    b.set_outflow_share("b => sysenv", share)
    a.set_outflow_share("custom", share)
    flow = mfa.flows["b => sysenv"]
    flow.from_process = mfa.processes["sysenv"]
    flow.to_process = b
    mfa.relink()
    assert list(mfa.flows) == ["sysenv => a", "custom", "sysenv => b"]
    # the share stays with the process it was set for, under the new flow name
    assert list(b.outflow_shares) == ["sysenv => b"]
    assert b.outflow_shares["sysenv => b"] is share
    assert list(a.outflow_shares) == ["custom"]
