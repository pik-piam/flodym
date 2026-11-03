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
    with pytest.raises(ValueError, match="already has the stock"):
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
