from typing import Callable

from .processes import Process
from .flodym_arrays import Flow
from .dimensions import DimensionSet
from .mfa_definition import FlowDefinition
from .flow_naming import process_names_with_arrow


def make_empty_flows(
    processes: dict[str, Process],
    flow_definitions: list[FlowDefinition],
    dims: DimensionSet,
    naming: Callable[[Process, Process], str] = process_names_with_arrow,
) -> dict[str, Flow]:
    """Initialize all defined flows with zero values.

    Args:
        processes: Dictionary of processes, with process names as keys.
        flow_definitions: List of flow definitions.
        dims: DimensionSet object containing all dimensions.
        naming: Function to generate names for flows. Default is `process_names_with_arrow`.

    Returns:
        Dictionary of flows, with flow names as keys.
    """
    flows = {}
    for flow_definition in flow_definitions:
        try:
            from_process = processes[flow_definition.from_process_name]
            to_process = processes[flow_definition.to_process_name]
        except KeyError:
            raise KeyError(f"Missing process required by flow definition {flow_definition}.")
        if flow_definition.name_override is not None:
            name = flow_definition.name_override
        else:
            name = naming(from_process, to_process)
        dim_subset = dims.get_subset(flow_definition.dim_letters)
        flow = Flow(from_process=from_process, to_process=to_process, name=name, dims=dim_subset)
        flows[flow.name] = flow
    return flows
