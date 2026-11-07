"""Home to helper functions for creating `Flow` objects."""

import logging
from collections.abc import Callable, Collection
from typing import Literal, overload

from ._attribute_dict import AttributeDict
from ._config import config
from .dimensions import DimensionSet
from .flodym_arrays import Flow
from .mfa_definition import FlowDefinition
from .processes import Process

logger = logging.getLogger(__name__)


@overload
def make_empty_flows(
    processes: dict[str, Process],
    flow_definitions: list[FlowDefinition],
    dims: DimensionSet,
    naming: Callable[[Process, Process], str] | None = ...,
    as_attribute_dict: Literal[False] = ...,
) -> dict[str, Flow]: ...
@overload
def make_empty_flows(
    processes: dict[str, Process],
    flow_definitions: list[FlowDefinition],
    dims: DimensionSet,
    naming: Callable[[Process, Process], str] | None = ...,
    *,
    as_attribute_dict: Literal[True],
) -> AttributeDict[Flow]: ...
def make_empty_flows(
    processes: dict[str, Process],
    flow_definitions: list[FlowDefinition],
    dims: DimensionSet,
    naming: Callable[[Process, Process], str] | None = None,
    as_attribute_dict: bool = False,
) -> dict[str, Flow] | AttributeDict[Flow]:
    """Initialize all defined flows with zero values.

    The parameter names given as :py:attr:`flodym.FlowDefinition.prescribed` are stored in the
    flows; the parameters are linked by :py:meth:`flodym.MFASystem.link_parameters`.

    The names of flows without ``name_override`` are generated from the processes they connect,
    using the function given in the global config option :py:attr:`flodym.Config.flow_naming`
    (default: :py:func:`flodym.flow_naming.process_names_with_arrow`). It can be changed, e.g. by
    ``flodym.config.flow_naming = flodym.flow_naming.process_ids``, or temporarily by
    ``with flodym.config.override(flow_naming=...):``.

    If several flows without name_override get the same generated name, e.g. because they connect
    the same pair of processes, a suffix ``__2``, ``__3``, ... is appended to the name of the second,
    third, ... flow, and a warning is logged.

    Args:
        processes: Dictionary of processes, with process names as keys.
        flow_definitions: List of flow definitions.
        dims: DimensionSet object containing all dimensions.
        naming: Deprecated, use the config option :py:attr:`flodym.Config.flow_naming`
            instead. If given, it is used instead of the config option.
        as_attribute_dict: If True, return an :py:class:`flodym.AttributeDict`, whose entries
            can also be accessed as attributes. Default is False, returning a plain dict.

    Returns:
        Dictionary of flows, with flow names as keys.

    Raises:
        KeyError: If a process given in a flow definition is not found in processes.
        ValueError: If a name_override is already used by another flow.
    """
    if naming is None:
        naming = config.flow_naming
    else:
        logger.info(
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
            name = _make_unique_name(naming(from_process, to_process), flows)
        dim_subset = dims.get_subset(flow_definition.dim_letters)
        flow = Flow(from_process=from_process, to_process=to_process, name=name, dims=dim_subset)
        flow._named_after_processes = flow_definition.name_override is None
        flow._set_prescribed(flow_definition.prescribed)
        flows[flow.name] = flow
    if as_attribute_dict:
        return AttributeDict(flows)
    return flows


def _make_unique_name(name: str, used_names: Collection[str]) -> str:
    """Append a numeric suffix ``__2``, ``__3``, ... to the name if it is already used.

    In this case, a warning is logged.

    Args:
        name: Proposed flow name.
        used_names: Names that are already taken.

    Returns:
        ``name`` itself if it is not used, else ``name`` with the lowest free suffix.
    """
    if name not in used_names:
        return name
    suffix_number = 2
    while f"{name}__{suffix_number}" in used_names:
        suffix_number += 1
    unique_name = f"{name}__{suffix_number}"
    logger.warning(f"Flow name '{name}' already exists. Using '{unique_name}' instead.")
    return unique_name
