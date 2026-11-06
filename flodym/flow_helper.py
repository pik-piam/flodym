import logging
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

    If several flows without name_override get the same generated name, e.g. because they connect
    the same pair of processes, a suffix ``_2``, ``_3``, ... is appended to the name of the second,
    third, ... flow, and a warning is logged.

    Args:
        processes: Dictionary of processes, with process names as keys.
        flow_definitions: List of flow definitions.
        dims: DimensionSet object containing all dimensions.
        naming: Function to generate names for flows. Default is `process_names_with_arrow`.

    Returns:
        Dictionary of flows, with flow names as keys.

    Raises:
        KeyError: If a process given in a flow definition is not found in processes.
        ValueError: If a name_override is already used by another flow.
    """
    flows = {}
    for flow_definition in flow_definitions:
        if flow_definition.from_process_name not in processes:
            raise KeyError(
                f"Process name {flow_definition.from_process_name} given in flow definition is not found in processes."
            )
        from_process = processes[flow_definition.from_process_name]
        if flow_definition.to_process_name not in processes:
            raise KeyError(
                f"Process name {flow_definition.to_process_name} given in flow definition is not found in processes."
            )
        to_process = processes[flow_definition.to_process_name]
        if flow_definition.name_override is not None:
            name = flow_definition.name_override
            if name in flows:
                raise ValueError(f"Flow name '{name}' given as name_override is not unique.")
        else:
            name = _make_unique_name(naming(from_process, to_process), flows)
        dim_subset = dims.get_subset(flow_definition.dim_letters)
        flow = Flow(from_process=from_process, to_process=to_process, name=name, dims=dim_subset)
        flows[flow.name] = flow
    return flows


def _make_unique_name(name: str, flows: dict[str, Flow]) -> str:
    """Append a numeric suffix to the name if it is already used in flows."""
    if name not in flows:
        return name
    suffix_number = 2
    while f"{name}_{suffix_number}" in flows:
        suffix_number += 1
    unique_name = f"{name}_{suffix_number}"
    logging.warning(f"Flow name '{name}' already exists. Using '{unique_name}' instead.")
    return unique_name
