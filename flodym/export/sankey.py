from pydantic import BaseModel as PydanticBaseModel, model_validator, ConfigDict
from typing import Optional, Any
import numpy as np
import plotly.graph_objects as go
import plotly as pl

from ..mfa_system import MFASystem
from ..flodym_arrays import Flow
from .helper import CustomNameDisplayer


class PlotlySankeyPlotter(CustomNameDisplayer, PydanticBaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow", protected_namespaces=())

    mfa: MFASystem
    """MFA system to visualize."""
    slice_dict: Optional[dict] = {}
    """for selection of a subset of the data; all other dimensions are summed over"""
    split_flows_by: Optional[str] = None
    """dimension name to split and color flows by; if None, all flows are colored the same"""
    color_scheme: Optional[str] = "blueish"
    """used if split_flows_by is not None and splitting dimension is in flow."""
    node_color: Optional[str] = "gray"
    """color of the nodes (processes and stocks)"""
    flow_color: Optional[str] = "hsl(230,20,70)"
    """used if split_flows_by is None or splitting dimension not in flow."""
    exclude_processes: Optional[list[str]] = ["sysenv"]
    """processes that won't show up in the plot; neither will flows to and from them"""
    exclude_flows: Optional[list[str]] = []
    """flows that won't show up in the plot"""
    __pydantic_extra__: dict[str, Any]

    @model_validator(mode="after")
    def check_colors(self):
        if self.split_flows_by is None and self.color_scheme != "blueish":
            raise ValueError(
                "If flows are not split, color_scheme is not used. Use flow_color instead."
            )
        return self

    @model_validator(mode="after")
    def check_dims(self):
        for dim_letter in self.slice_dict.keys():
            if dim_letter not in self.mfa.dims.letters:
                raise ValueError(f"Dimension {dim_letter} given in slice_dict not in DimensionSet.")
        if self.split_flows_by is not None and self.split_flows_by not in self.mfa.dims.names:
            raise ValueError(
                f"Dimension {self.split_flows_by} given in split_flows_by not in DimensionSet"
            )
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

    def plot(self):
        self._get_nodes_and_links()
        return self._get_fig()

    def _get_nodes_and_links(self):

        self.processes = [
            p for p in self.mfa.processes.values() if p.name not in self.exclude_processes
        ]
        self.ids_in_sankey = {p.id: i for i, p in enumerate(self.processes)}
        self.exclude_process_ids = [
            p.id for p in self.mfa.processes.values() if p.name in self.exclude_processes
        ]

        self._get_link_list()
        self.links = self.link_list.to_dict()
        self.nodes = self._get_nodes_dict()

    def _get_link_list(self):
        self.link_list = LinkList()
        for f in self.mfa.flows.values():
            if (
                (f.name in self.exclude_flows)
                or (f.from_process_id in self.exclude_process_ids)
                or (f.to_process_id in self.exclude_process_ids)
            ):
                continue
            self._add_flow(f)

    def _add_flow(self, f: Flow):
        source = self.ids_in_sankey[f.from_process.id]
        target = self.ids_in_sankey[f.to_process.id]
        label = self.display_name(f.name)

        slice_dict = {k: v for k, v in self.slice_dict.items() if k in f.dims.letters}
        f_slice = f[slice_dict]

        if self.split_flows_by is not None and self.split_flows_by in f.dims.names:
            splitting_letter_tuple = (self.mfa.dims[self.split_flows_by].letter,)
            values = f_slice.sum_values_to(splitting_letter_tuple)
            for v, c in zip(values, self._colors):
                self.link_list.append(label=label, source=source, target=target, color=c, value=v)
        else:
            self.link_list.append(
                label=label,
                source=source,
                target=target,
                color=self.flow_color,
                value=f_slice.sum_values(),
            )

    def _get_nodes_dict(self):
        return {
            "label": [self.display_name(p.name) for p in self.processes],
            "color": [self.node_color for p in self.processes],  # 'rgb(50, 50, 50)'
            "pad": 10,
        }

    def _get_fig(self):
        fig = go.Figure(
            go.Sankey(
                arrangement="snap",
                link=self.link_list.to_dict(),
                node=self.nodes,
            )
        )
        return fig

    @property
    def _n_colors(self):
        if self.split_flows_by is None:
            return 1
        else:
            return self.mfa.dims[self.split_flows_by].len

    @property
    def _colors(self):
        if self.color_scheme == "blueish":
            n_max = 10

            def colors(n_colors):
                return [f"hsl({240 - 20 * i},70%,50%)" for i in range(n_colors)]

        elif self.color_scheme == "antique":
            n_max = len(pl.colors.qualitative.Antique)

            def colors(n_colors):
                return pl.colors.qualitative.Antique[:n_colors]

        elif self.color_scheme == "viridis":
            n_max = np.inf

            def colors(n_colors):
                return pl.colors.sample_colorscale("Viridis", n_colors + 1, colortype="rgb")

        else:
            raise ValueError("invalid color scheme")

        if self._n_colors > n_max:
            raise ValueError(
                f"Too many colors ({self._n_colors}) requested for color scheme {self.color_scheme}"
            )

        return colors(self._n_colors)


class Link(PydanticBaseModel):

    label: str
    source: int
    target: int
    color: str
    value: float


class LinkList(PydanticBaseModel):

    _links: Optional[list[Link]] = []

    def append(self, label: str, source: int, target: int, color: str, value: float):
        self._links.append(
            Link(label=label, source=source, target=target, color=color, value=value)
        )

    def to_dict(self):
        return {
            "source": [link.source for link in self._links],
            "target": [link.target for link in self._links],
            "value": [link.value for link in self._links],
            "label": [link.label for link in self._links],
            "color": [link.color for link in self._links],
        }
