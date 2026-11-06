import logging

import pytest

from flodym import Dimension, DimensionSet, FlowDefinition, make_empty_flows, make_processes


@pytest.fixture
def processes():
    return make_processes(["sysenv", "a", "b"])


@pytest.fixture
def dims():
    return DimensionSet(dim_list=[Dimension(name="time", letter="t", items=[2020, 2021])])


def test_make_empty_flows_names_and_links(processes, dims):
    flow_definitions = [
        FlowDefinition(from_process_name="sysenv", to_process_name="a", dim_letters=("t",)),
        FlowDefinition(
            from_process_name="a", to_process_name="b", dim_letters=("t",), name_override="custom"
        ),
    ]
    flows = make_empty_flows(processes, flow_definitions, dims)
    assert list(flows) == ["sysenv => a", "custom"]
    assert flows["custom"].from_process is processes["a"]
    assert flows["custom"].to_process is processes["b"]


def test_make_empty_flows_duplicate_generated_names_get_suffix(processes, dims, caplog):
    flow_definitions = [
        FlowDefinition(from_process_name="a", to_process_name="b", dim_letters=("t",))
        for _ in range(3)
    ]
    with caplog.at_level(logging.WARNING):
        flows = make_empty_flows(processes, flow_definitions, dims)
    assert list(flows) == ["a => b", "a => b__2", "a => b__3"]
    assert all(flow.name == name for name, flow in flows.items())
    assert len({id(flow) for flow in flows.values()}) == 3
    assert "a => b__2" in caplog.text
    assert "a => b__3" in caplog.text


def test_make_empty_flows_suffix_skips_used_names(processes, dims):
    flow_definitions = [
        FlowDefinition(
            from_process_name="a",
            to_process_name="b",
            dim_letters=("t",),
            name_override="a => b__2",
        ),
        FlowDefinition(from_process_name="a", to_process_name="b", dim_letters=("t",)),
        FlowDefinition(from_process_name="a", to_process_name="b", dim_letters=("t",)),
    ]
    flows = make_empty_flows(processes, flow_definitions, dims)
    assert list(flows) == ["a => b__2", "a => b", "a => b__3"]


def test_make_empty_flows_duplicate_name_override_raises(processes, dims):
    flow_definitions = [
        FlowDefinition(
            from_process_name="a", to_process_name="b", dim_letters=("t",), name_override="custom"
        ),
        FlowDefinition(
            from_process_name="b", to_process_name="a", dim_letters=("t",), name_override="custom"
        ),
    ]
    with pytest.raises(ValueError, match="not unique"):
        make_empty_flows(processes, flow_definitions, dims)


@pytest.mark.parametrize("from_name, to_name", [("missing", "a"), ("a", "missing")])
def test_make_empty_flows_missing_process_raises(processes, dims, from_name, to_name):
    flow_definitions = [
        FlowDefinition(from_process_name=from_name, to_process_name=to_name, dim_letters=("t",))
    ]
    with pytest.raises(KeyError, match="missing"):
        make_empty_flows(processes, flow_definitions, dims)


def test_make_empty_flows_uses_config_naming(processes, dims):
    import flodym
    from flodym.flow_naming import process_ids, process_names_with_arrow

    flow_definitions = [
        FlowDefinition(from_process_name="sysenv", to_process_name="a", dim_letters=("t",)),
        FlowDefinition(
            from_process_name="a", to_process_name="b", dim_letters=("t",), name_override="custom"
        ),
    ]
    with flodym.config.override(flow_naming=process_ids):
        flows = make_empty_flows(processes, flow_definitions, dims)
    assert list(flows) == ["F0_1", "custom"]
    assert flodym.config.flow_naming is process_names_with_arrow


def test_make_empty_flows_naming_argument_deprecated(processes, dims, caplog):
    from flodym.flow_naming import process_ids

    flow_definitions = [
        FlowDefinition(from_process_name="sysenv", to_process_name="a", dim_letters=("t",))
    ]
    with caplog.at_level(logging.INFO):
        flows = make_empty_flows(processes, flow_definitions, dims, naming=process_ids)
    assert list(flows) == ["F0_1"]
    assert "deprecated" in caplog.text
    assert "flodym.config.flow_naming" in caplog.text
