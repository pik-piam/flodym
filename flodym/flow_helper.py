import logging
from typing import Callable, Collection, Optional

from .config import config
from .processes import Process
from .flodym_arrays import Flow
from .dimensions import DimensionSet
from .mfa_definition import FlowDefinition


def make_empty_flows(
    processes: dict[str, Process],
    flow_definitions: list[FlowDefinition],
    dims: DimensionSet,
    naming: Optional[Callable[[Process, Process], str]] = None,
) -> dict[str, Flow]:
    """Initialize all defined flows with zero values.

    The names of flows without ``name_override`` are generated from the processes they connect,
    using the function given in the global config option :py:attr:`flodym.config.Config.flow_naming`
    (default: :py:func:`flodym.flow_naming.process_names_with_arrow`). It can be changed, e.g. by
    ``flodym.config.flow_naming = flodym.flow_naming.process_ids``, or temporarily by
    ``with flodym.config.override(flow_naming=...):``.

    If several flows without name_override get the same generated name, e.g. because they connect
    the same pair of processes, a suffix ``_2``, ``_3``, ... is appended to the name of the second,
    third, ... flow, and a warning is logged.

    Args:
        processes: Dictionary of processes, with process names as keys.
        flow_definitions: List of flow definitions.
        dims: DimensionSet object containing all dimensions.
        naming: Deprecated, use the config option :py:attr:`flodym.config.Config.flow_naming`
            instead. If given, it is used instead of the config option.

    Returns:
        Dictionary of flows, with flow names as keys.

    Raises:
        KeyError: If a process given in a flow definition is not found in processes.
        ValueError: If a name_override is already used by another flow.
    """
    if naming is None:
        naming = config.flow_naming
    else:
        logging.info(
            "The naming argument of make_empty_flows() is deprecated since v2.0.0. "
            "It will be removed in v3.0.0. Set flodym.config.flow_naming instead."
        )
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
            name = make_unique_name(naming(from_process, to_process), flows)
        dim_subset = dims.get_subset(flow_definition.dim_letters)
        flow = Flow(from_process=from_process, to_process=to_process, name=name, dims=dim_subset)
        flow._named_after_processes = flow_definition.name_override is None
        flows[flow.name] = flow
    return flows


def make_unique_name(name: str, used_names: Collection[str]) -> str:
    """Append a numeric suffix ``_2``, ``_3``, ... to the name if it is already used.
    In this case, a warning is logged."""
    if name not in used_names:
        return name
    suffix_number = 2
    while f"{name}_{suffix_number}" in used_names:
        suffix_number += 1
    unique_name = f"{name}_{suffix_number}"
    logging.warning(f"Flow name '{name}' already exists. Using '{unique_name}' instead.")
    return unique_name
