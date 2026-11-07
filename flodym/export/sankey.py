"""Sankey diagrams of MFA systems."""

from typing import Any

import plotly.graph_objects as go
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, model_validator
from typing_extensions import Self

from ..flodym_arrays import Flow
from ..mfa_system import MFASystem
from ..processes import Process
from ._helper import CustomNameDisplayer


class PlotlySankeyPlotter(CustomNameDisplayer, PydanticBaseModel):
    """Plot the flows of an MFA system as a plotly Sankey diagram.

    Flow values are summed over all dimensions not selected in ``slice_dict``. Flows can be split
    into several colored links along one dimension via ``flow_color_dict``.

    Example:

        >>> from flodym.export import PlotlySankeyPlotter
        >>> fig = PlotlySankeyPlotter(mfa=mfa).plot()
        >>> fig.show()  # doctest: +SKIP
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    mfa: MFASystem
    """MFA system to visualize."""
    slice_dict: dict | None = {}
    """for selection of a subset of the data; all other dimensions are summed over"""
    node_color_dict: dict | None = {"default": "gray"}
    """color of the nodes (processes and stocks)"""
    flow_color_dict: dict | None = {"default": "hsl(230,20,70)"}
    """dictionary of colors for flows.
    Keys are flow names, values are either a single color or a tuple of the dimension names
    to split the flow by, and a color scheme as a list of colors. There must be a "default" key
    to resort to if a flow is not in the dictionary.
    """
    exclude_processes: list[str] | None = ["sysenv"]
    """processes that won't show up in the plot; neither will flows to and from them"""
    exclude_flows: list[str] | None = []
    """flows that won't show up in the plot"""

    @model_validator(mode="after")
    def _check_dims(self) -> Self:
        """Check that all dimensions in ``slice_dict`` are in the MFA system."""
        for dim_letter in self.slice_dict:
            if dim_letter not in self.mfa.dims.letters:
                raise ValueError(f"Dimension {dim_letter} given in slice_dict not in DimensionSet.")
        return self

    @model_validator(mode="after")
    def _check_excluded(self) -> Self:
        """Check that all excluded processes and flows exist in the MFA system."""
        for p in self.exclude_processes:
            if p not in self.mfa.processes:
                raise ValueError(f"Process {p} given in exclude_processes not in MFASystem.")
        for f in self.exclude_flows:
            if f not in self.mfa.flows:
                raise ValueError(f"Flow {f} given in exclude_flows not in MFASystem.")
        return self

    @model_validator(mode="after")
    def _check_flow_colors(self) -> Self:
        """Validate ``flow_color_dict`` and fill in the default color for missing flows."""
        for f in self.shown_flows.values():
            if "default" not in self.flow_color_dict:
                raise ValueError(
                    "flow_color_dict must have a 'default' key to resort to if a flow is not in the dictionary"
                )
            if f.name not in self.flow_color_dict:
                self.flow_color_dict[f.name] = self.flow_color_dict["default"]
                fallback_str = " (not found in dict, using default color)"
            else:
                fallback_str = ""
            self._check_flow_color(f, fallback_str)
        return self

    @model_validator(mode="after")
    def _check_node_colors(self) -> Self:
        """Validate ``node_color_dict`` and fill in the default color for missing processes."""
        for p in self.shown_processes:
            if "default" not in self.node_color_dict:
                raise ValueError(
                    "node_color_dict must have a 'default' key to resort to if a process is not in the dictionary"
                )
            if p.name not in self.node_color_dict:
                self.node_color_dict[p.name] = self.node_color_dict["default"]
                fallback_str = " (not found in dict, using default color)"
            else:
                fallback_str = ""
            if not isinstance(self.node_color_dict[p.name], str):
                raise TypeError(
                    f"Color for process {p.name}{fallback_str} must be a string, not a {type(self.node_color_dict[p.name])}."
                )
        return self

    @property
    def shown_processes(self) -> list[Process]:
        """Processes that are not excluded."""
        return [p for p in self.mfa.processes.values() if p.name not in self.exclude_processes]

    @property
    def _excluded_process_ids(self) -> list[int]:
        """IDs of the excluded processes."""
        return [p.id for p in self.mfa.processes.values() if p.name in self.exclude_processes]

    @property
    def shown_flows(self) -> dict[str, Flow]:
        """Flows that are neither excluded nor connected to an excluded process, by name."""
        return {f.name: f for f in self.mfa.flows.values() if self._flow_is_shown(f)}

    def _flow_is_shown(self, f: Flow) -> bool:
        return not (
            (f.name in self.exclude_flows)
            or (f.from_process_id in self._excluded_process_ids)
            or (f.to_process_id in self._excluded_process_ids)
        )

    @property
    def _ids_in_sankey(self) -> dict[int, int]:
        """Mapping from process IDs to node indices in the Sankey diagram."""
        return {p.id: i for i, p in enumerate(self.shown_processes)}

    def plot(self) -> go.Figure:
        """Create the Sankey diagram.

        Returns:
            The plotly figure.
        """
        links = self._get_links_dict()
        nodes = self._get_nodes_dict()
        return self._get_fig(links, nodes)

    def _get_links_dict(self) -> dict[str, list[Any]]:
        links = _DictOfLists(list_names=["source", "target", "value", "label", "color"])
        for f in self.shown_flows.values():
            self._append_flow(f, links)
        return links.dict

    def _append_flow(self, f: Flow, links: "_DictOfLists") -> None:
        source = self._ids_in_sankey[f.from_process.id]
        target = self._ids_in_sankey[f.to_process.id]
        label = self.display_name(f.name)

        slice_dict = {k: v for k, v in self.slice_dict.items() if k in f.dims.letters}
        f_slice = f[slice_dict]
        color = self.flow_color_dict[f.name]
        if isinstance(color, tuple):
            split_flows_by = color[0]
            colors = color[1]
            labels = self.mfa.dims[split_flows_by].items
            values = f_slice.sum_values_to((self.mfa.dims[split_flows_by].letter,))
            for l, c, v in zip(labels, colors, values):
                links.append(source=source, target=target, label=l, color=c, value=v)
        else:
            links.append(
                source=source,
                target=target,
                label=label,
                color=color,
                value=f_slice.sum_values(),
            )

    def _get_nodes_dict(self) -> dict[str, Any]:
        return {
            "label": [self.display_name(p.name) for p in self.shown_processes],
            "color": [self.node_color_dict[p.name] for p in self.shown_processes],
            "pad": 10,
        }

    def _get_fig(self, links: dict[str, Any], nodes: dict[str, Any]) -> go.Figure:
        fig = go.Figure(
            go.Sankey(
                arrangement="snap",
                link=links,
                node=nodes,
            )
        )
        return fig

    def _check_flow_color(self, f: Flow, fallback_str: str) -> None:
        color = self.flow_color_dict[f.name]
        if isinstance(color, str):
            return
        elif not isinstance(color, tuple):
            raise TypeError(
                f"In flow_color_dict, the value for flow {f.name}{fallback_str} must be either a color string or a tuple of dimension name and color list"
            )

        if len(color) != 2:
            raise ValueError(
                f"In flow_color_dict, color tuple for flow {f.name}{fallback_str} must have length 2."
            )
        if color[0] not in f.dims:
            raise ValueError(
                f"In flow_color_dict, first element of color tuple for flow {f.name}{fallback_str} must be a dimension in flow {f.name}"
            )
        if not isinstance(color[1], list):
            raise TypeError(
                f"In flow_color_dict, second element of color tuple for flow {f.name}{fallback_str} must be a list of colors"
            )
        if len(color[1]) < self.mfa.dims[color[0]].len:
            raise ValueError(
                f"In flow_color_dict, list in second element of color tuple for flow {f.name}{fallback_str} must not be shorter than dimension {color[0]}"
            )


class _DictOfLists:
    """Dictionary of lists, to which values can be appended for several keys at once."""

    def __init__(self, list_names: list[str]) -> None:
        self.dict: dict[str, list[Any]] = {l: [] for l in list_names}

    def append(self, **kwargs: Any) -> None:
        """Append each keyword argument value to the list of the same name."""
        for k, v in kwargs.items():
            self.dict[k].append(v)
