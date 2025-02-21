from pydantic import BaseModel as PydanticBaseModel, model_validator, ConfigDict, computed_field
from typing import Optional, Any
import numpy as np
import plotly.graph_objects as go
import plotly as pl

from ..mfa_system import MFASystem
from ..flodym_arrays import Flow
from .helper import CustomNameDisplayer


class PlotlySankeyPlotter(CustomNameDisplayer, PydanticBaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    mfa: MFASystem
    """MFA system to visualize."""
    slice_dict: Optional[dict] = {}
    """for selection of a subset of the data; all other dimensions are summed over"""
    node_color_dict: Optional[dict] = {"default": "gray"}
    """color of the nodes (processes and stocks)"""
    flow_color_dict: Optional[dict] = {"default": "hsl(230,20,70)"}
    """dictionary of colors for flows.
    Keys are flow names, values are either a single color or a tuple of the dimension names
    to split the flow by, and a color scheme as a list of colors. There must be a "default" key
    to resort to if a flow is not in the dictionary.
    """
    exclude_processes: Optional[list[str]] = ["sysenv"]
    """processes that won't show up in the plot; neither will flows to and from them"""
    exclude_flows: Optional[list[str]] = []
    """flows that won't show up in the plot"""

    @model_validator(mode="after")
    def check_dims(self):
        for dim_letter in self.slice_dict.keys():
            if dim_letter not in self.mfa.dims.letters:
                raise ValueError(f"Dimension {dim_letter} given in slice_dict not in DimensionSet.")
        return self

    @model_validator(mode="after")
    def check_excluded(self):
        for p in self.exclude_processes:
            if p not in self.mfa.processes:
                raise ValueError(f"Process {p} given in exclude_processes not in MFASystem.")
        for f in self.exclude_flows:
            if f not in self.mfa.flows:
                raise ValueError(f"Flow {f} given in exclude_flows not in MFASystem.")
        return self

    @model_validator(mode="after")
    def check_flow_colors(self):
        for f in self.shown_flows.values():
            if "default" not in self.flow_color_dict:
                raise ValueError(
                    f"flow_color_dict must have a 'default' key to resort to if a flow is not in the dictionary"
                )
            if f.name not in self.flow_color_dict:
                self.flow_color_dict[f.name] = self.flow_color_dict["default"]
                fallback_str = " (not found in dict, using default color)"
            else:
                fallback_str = ""
            self._check_flow_color(f, fallback_str)
        return self

    @model_validator(mode="after")
    def check_node_colors(self):
        for p in self.shown_processes:
            if "default" not in self.node_color_dict:
                raise ValueError(
                    f"node_color_dict must have a 'default' key to resort to if a process is not in the dictionary"
                )
            if p.name not in self.node_color_dict:
                self.node_color_dict[p.name] = self.node_color_dict["default"]
                fallback_str = " (not found in dict, using default color)"
            else:
                fallback_str = ""
            if not isinstance(self.node_color_dict[p.name], str):
                raise ValueError(
                    f"Color for process {p.name}{fallback_str} must be a string, not a {type(self.node_color_dict[p.name])}."
                )
        return self

    def _check_flow_color(self, f: Flow, fallback_str):
        color = self.flow_color_dict[f.name]
        if isinstance(color, str):
            return
        elif not isinstance(color, tuple):
            raise ValueError(
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
            raise ValueError(
                f"In flow_color_dict, second element of color tuple for flow {f.name}{fallback_str} must be a list of colors"
            )
        if len(color[1]) < self.mfa.dims[color[0]].len:
            raise ValueError(
                f"In flow_color_dict, list in second element of color tuple for flow {f.name}{fallback_str} must not be shorter than dimension {color[0]}"
            )

    @property
    def shown_processes(self):
        return [p for p in self.mfa.processes.values() if p.name not in self.exclude_processes]

    @property
    def excluded_process_ids(self):
        return [p.id for p in self.mfa.processes.values() if p.name in self.exclude_processes]

    @property
    def shown_flows(self) -> dict[str, Flow]:
        return {f.name: f for f in self.mfa.flows.values() if self._flow_is_shown(f)}

    def _flow_is_shown(self, f: Flow):
        return not (
            (f.name in self.exclude_flows)
            or (f.from_process_id in self.excluded_process_ids)
            or (f.to_process_id in self.excluded_process_ids)
        )

    @property
    def ids_in_sankey(self):
        return {p.id: i for i, p in enumerate(self.shown_processes)}

    def plot(self):
        links = self._get_links_dict()
        nodes = self._get_nodes_dict()
        return self._get_fig(links, nodes)

    def _get_links_dict(self):
        links = DictOfLists(list_names=["source", "target", "value", "label", "color"])
        for f in self.shown_flows.values():
            self._append_flow(f, links)
        return links.dict

    def _append_flow(self, f: Flow, links: "DictOfLists"):
        source = self.ids_in_sankey[f.from_process.id]
        target = self.ids_in_sankey[f.to_process.id]
        label = self.display_name(f.name)

        slice_dict = {k: v for k, v in self.slice_dict.items() if k in f.dims.letters}
        f_slice = f[slice_dict]
        color = self.flow_color_dict[f.name]
        if isinstance(color, tuple):
            split_flows_by = color[0]
            colors = color[1]
            splitting_letter_tuple = (self.mfa.dims[split_flows_by].letter,)
            values = f_slice.sum_values_to(splitting_letter_tuple)
            for v, c in zip(values, colors):
                links.append(label=label, source=source, target=target, color=c, value=v)
        else:
            links.append(
                label=label,
                source=source,
                target=target,
                color=color,
                value=f_slice.sum_values(),
            )

    def _get_nodes_dict(self):
        return {
            "label": [self.display_name(p.name) for p in self.shown_processes],
            "color": [self.node_color_dict[p.name] for p in self.shown_processes],
            "pad": 10,
        }

    def _get_fig(self, links: dict, nodes: dict) -> go.Figure:
        fig = go.Figure(
            go.Sankey(
                arrangement="snap",
                link=links,
                node=nodes,
            )
        )
        return fig


class DictOfLists:

    def __init__(self, list_names: list[str]):
        self.dict = {l: [] for l in list_names}

    def append(self, **kwargs):
        for k, v in kwargs.items():
            self.dict[k].append(v)
