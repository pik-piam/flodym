"""Visualize the structure of an MFA system.

Visualize the system layout itself, i.e. the directed graph formed
by the processes (nodes), the flows connecting them (directed edges), and the stocks attached
to processes. This is useful to inspect or document the topology of a model definition.

Two backends are provided:

- :py:class:`PlotlyProcessGraphPlotter` renders an interactive graph with plotly
  using a basic layout algorithm.
- :py:class:`GraphvizProcessGraphPlotter` produces a :py:class:`graphviz.Digraph`.
  Usually results in better graphs than plotly but it requires the ``graphviz`` package
  (and, for rendering to an image, the Graphviz system binaries).
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

import plotly.graph_objects as go
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, model_validator

from ..flodym_arrays import Flow
from ..mfa_system import MFASystem
from ..processes import Process
from ..stocks import Stock
from .helper import CustomNameDisplayer

if TYPE_CHECKING:
    import graphviz


class ProcessGraphPlotter(CustomNameDisplayer, ABC, PydanticBaseModel):
    """Abstract base class for plotting the structure of an MFA system as a directed graph.

    Subclasses implement the actual rendering for a specific backend.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow", protected_namespaces=())

    mfa: MFASystem
    """MFA system whose structure is visualized."""
    exclude_processes: list[str] = []
    """Processes that won't show up in the plot; neither will flows to and from them.

    Note that the system boundary process ``sysenv`` is shown by default."""
    exclude_flows: list[str] = []
    """Flows that won't show up in the plot."""
    show_stocks: bool = True
    """Whether to include the stocks (attached to their process) in the graph."""
    show_flow_labels: bool = False
    """Whether to label the edges with the flow names.

    Off by default, as the default flow names encode the connected processes, which
    would duplicate the information in the graph."""
    process_color: str = "#8ecae6"
    """Default fill color of the process nodes."""
    stock_color: str = "#a7c957"
    """Default fill color of the stock nodes."""
    flow_color: str = "#6c757d"
    """Color of the flow edges (and of the connections between processes and stocks)."""
    node_colors: dict[str, str] = {}
    """Optional per-node color overrides, mapping a process or stock name to a color string.

    Nodes not in this dictionary use ``process_color`` or ``stock_color``, respectively."""

    @model_validator(mode="after")
    def check_consistency_with_mfa(self):
        for process in self.exclude_processes:
            if process not in self.mfa.processes:
                raise ValueError(f"Process {process} given in exclude_processes not in MFASystem.")
        for flow in self.exclude_flows:
            if flow not in self.mfa.flows:
                raise ValueError(f"Flow {flow} given in exclude_flows not in MFASystem.")
        for node in self.node_colors.keys():
            if node not in self.mfa.processes and node not in self.mfa.stocks:
                raise ValueError(
                    f"Node {node} given in node_colors not in MFASystem processes or stocks."
                )
        return self

    @property
    def shown_processes(self) -> list[Process]:
        return [
            process
            for process in self.mfa.processes.values()
            if process.name not in self.exclude_processes
        ]

    @property
    def _excluded_process_ids(self) -> list[int]:
        return [
            process.id
            for process in self.mfa.processes.values()
            if process.name in self.exclude_processes
        ]

    @property
    def shown_flows(self) -> list[Flow]:
        def should_show(flow: Flow) -> bool:
            return not (
                (flow.name in self.exclude_flows)
                or (flow.from_process_id in self._excluded_process_ids)
                or (flow.to_process_id in self._excluded_process_ids)
            )

        return [flow for flow in self.mfa.flows.values() if should_show(flow)]

    @property
    def shown_stocks(self) -> list[Stock]:
        if not self.show_stocks:
            return []
        return [
            stock
            for stock in self.mfa.stocks.values()
            if stock.process is not None and stock.process.id not in self._excluded_process_ids
        ]

    def _get_process_color(self, name: str) -> str:
        return self.node_colors.get(name, self.process_color)

    def _get_stock_color(self, name: str) -> str:
        return self.node_colors.get(name, self.stock_color)

    @abstractmethod
    def plot(self):
        """Create the graph figure."""
        ...


class GraphvizProcessGraphPlotter(ProcessGraphPlotter):
    """Render the structure of an MFA system with Graphviz.

    Requires the optional ``graphviz`` python package. Building the returned
    :py:class:`graphviz.Digraph` object only needs the python package; rendering it to an image
    (e.g. via ``.render()`` or displaying it in a notebook) additionally requires the Graphviz
    system binaries (the ``dot`` executable).

    Example:

        >>> from flodym.export import GraphvizProcessGraphPlotter
        >>> dot = GraphvizProcessGraphPlotter(mfa=mfa).plot()
        >>> dot.render("system_structure", format="png")
    """

    rankdir: str = "LR"
    """Layout direction passed to Graphviz: ``'LR'`` (left to right, default), ``'TB'`` (top to
    bottom), ``'RL'`` or ``'BT'``."""
    graph_attr: dict = {}
    """Additional Graphviz graph attributes, see :py:class:`graphviz.Digraph` for details."""

    def plot(self) -> "graphviz.Digraph":
        def get_process_id(process: Process) -> str:
            return f"process_{process.id}"

        try:
            import graphviz
        except ImportError as e:
            raise ImportError(
                "The 'graphviz' package is required for GraphvizProcessGraphPlotter."
            ) from e

        dot = graphviz.Digraph(
            "mfa_system",
            graph_attr={"rankdir": self.rankdir, **self.graph_attr},
        )
        for process in self.shown_processes:
            dot.node(
                get_process_id(process),
                label=self.display_name(process.name),
                shape="box",
                style="rounded,filled",
                fillcolor=self._get_process_color(process.name),
            )
        for flow in self.shown_flows:
            dot.edge(
                get_process_id(flow.from_process),
                get_process_id(flow.to_process),
                label=self.display_name(flow.name) if self.show_flow_labels else "",
                color=self.flow_color,
            )
        for stock in self.shown_stocks:
            stock_id = f"stock_{stock.name}"
            dot.node(
                stock_id,
                label=self.display_name(stock.name),
                shape="cylinder",
                style="filled",
                fillcolor=self._get_stock_color(stock.name),
            )
            dot.edge(
                get_process_id(stock.process),
                stock_id,
                dir="both",
                style="dashed",
                color=self.flow_color,
            )
        return dot


class PlotlyProcessGraphPlotter(ProcessGraphPlotter):
    """Render the structure of an MFA system with plotly.

    Note:
        plotly doesn't have a built-in function for graph plotting, so the layout is
        computed with a simple algorithm that assigns each process to a layer based on
        its longest path from the system boundary, and then distributes the processes
        within each layer horizontally.

    Example:

        >>> from flodym.export import PlotlyProcessGraphPlotter
        >>> fig = PlotlyProcessGraphPlotter(mfa=mfa).plot()
        >>> fig.show()  # doctest: +SKIP
    """

    title: Optional[str] = None
    """Title of the figure, if desired."""
    node_size: float = 26.0
    """Marker size of the process and stock nodes."""

    def plot(self) -> go.Figure:
        process_positions = self._calculate_process_positions()
        stock_positions = self._calculate_stock_positions(process_positions)

        fig = go.Figure()
        self._add_flow_edges(fig, process_positions)
        self._add_stock_edges(fig, process_positions, stock_positions)
        self._add_nodes(
            fig=fig,
            items=self.shown_processes,
            positions=process_positions,
            key_getter=lambda process: process.id,
            color_getter=self._get_process_color,
            marker_symbol="square",
            textposition="top center",
        )
        self._add_nodes(
            fig=fig,
            items=self.shown_stocks,
            positions=stock_positions,
            key_getter=lambda stock: stock.name,
            color_getter=self._get_stock_color,
            marker_symbol="circle",
            textposition="bottom center",
        )

        fig.update_layout(
            title=self.title,
            showlegend=False,
            plot_bgcolor="white",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        )
        return fig

    def _calculate_process_layers(self) -> dict[int, int]:
        """Assign each shown process to a layer via its longest path from the system boundary.

        MFA systems are generally cyclic (e.g., recycling loops), so a plain
        longest-path search would not terminate sensibly.
        The layering is therefore computed by first dropping the back-edges that close
        cycles, and the longest path is then computed over the remaining acyclic graph.
        """
        ids = [process.id for process in self.shown_processes]
        adjacency: dict[int, list[int]] = defaultdict(list)
        for flow in self.shown_flows:
            u, v = flow.from_process_id, flow.to_process_id
            if u != v:
                adjacency[u].append(v)

        forward_edges = self._find_forward_edges(ids, adjacency)

        layer = {i: 0 for i in ids}
        for _ in range(len(ids)):
            changed = False
            for u, v in forward_edges:
                if layer[v] < layer[u] + 1:
                    layer[v] = layer[u] + 1
                    changed = True
            if not changed:
                break
        return layer

    @staticmethod
    def _find_forward_edges(
        ids: list[int], adjacency: dict[int, list[int]]
    ) -> list[tuple[int, int]]:
        """Return the edges of the graph with cycle-closing back-edges removed.

        A depth-first traversal is used. Edges are colored white (unvisited), grey (currently in the stack), or black (fully explored).
        Back-edges are those that point to a grey node, and they are dropped.
        """
        white, grey, black = 0, 1, 2
        color = {i: white for i in ids}
        forward_edges: list[tuple[int, int]] = []
        for start in ids:
            if color[start] != white:
                continue
            color[start] = grey
            stack = [(start, iter(adjacency[start]))]
            while stack:
                node, neighbors = stack[-1]
                descended = False
                for v in neighbors:
                    if color[v] == grey:
                        continue  # back-edge closing a cycle -> drop it
                    forward_edges.append((node, v))
                    if color[v] == white:
                        color[v] = grey
                        stack.append((v, iter(adjacency[v])))
                        descended = True
                        break
                if not descended:
                    color[node] = black
                    stack.pop()
        return forward_edges

    def _calculate_process_positions(self) -> dict[int, tuple[float, float]]:
        layers = self._calculate_process_layers()
        by_layer: dict[int, list[int]] = defaultdict(list)
        for pid in sorted(layers, key=lambda i: (layers[i], i)):
            by_layer[layers[pid]].append(pid)

        positions: dict[int, tuple[float, float]] = {}
        for layer, pids in by_layer.items():
            n = len(pids)
            for k, pid in enumerate(pids):
                y = ((n - 1) / 2 - k) * 1.5
                positions[pid] = (layer, y)
        return positions

    def _calculate_stock_positions(
        self, process_positions: dict[int, tuple[float, float]]
    ) -> dict[str, tuple[float, float]]:
        stocks_by_process: dict[int, list[Stock]] = defaultdict(list)
        stocks_not_attached_to_process = []
        for stock in self.shown_stocks:
            if stock.process is not None:
                stocks_by_process[stock.process.id].append(stock)
            else:
                stocks_not_attached_to_process.append(stock)

        positions: dict[str, tuple[float, float]] = {}
        for pid, stocks in stocks_by_process.items():
            px, py = process_positions[pid]
            n = len(stocks)
            for j, stock in enumerate(stocks):
                x = px + (j - (n - 1) / 2) * 0.5
                positions[stock.name] = (x, py - 0.9)
        for j, stock in enumerate(stocks_not_attached_to_process):
            x = j * 0.5
            positions[stock.name] = (x, -1.5)
        return positions

    def _add_flow_edges(self, fig: go.Figure, positions: dict[int, tuple[float, float]]):
        label_x, label_y, label_text = [], [], []
        for flow in self.shown_flows:
            x0, y0 = positions[flow.from_process_id]
            x1, y1 = positions[flow.to_process_id]
            fig.add_annotation(
                x=x1,
                y=y1,
                ax=x0,
                ay=y0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.2,
                arrowwidth=1.5,
                arrowcolor=self.flow_color,
                standoff=self.node_size / 2 + 4,
                startstandoff=self.node_size / 2 + 4,
            )
            label_x.append((x0 + x1) / 2)
            label_y.append((y0 + y1) / 2)
            label_text.append(self.display_name(flow.name))

        # invisible markers at the edge midpoints, carrying the flow name on hover
        # (and as text if show_flow_labels is set)
        fig.add_trace(
            go.Scatter(
                x=label_x,
                y=label_y,
                mode="markers+text" if self.show_flow_labels else "markers",
                marker=dict(size=1, color="rgba(0,0,0,0)"),
                text=label_text if self.show_flow_labels else None,
                textfont=dict(size=10, color=self.flow_color),
                textposition="middle center",
                hovertext=label_text,
                hoverinfo="text",
            )
        )

    def _add_stock_edges(
        self,
        fig: go.Figure,
        process_positions: dict[int, tuple[float, float]],
        stock_positions: dict[str, tuple[float, float]],
    ):
        for stock in self.shown_stocks:
            px, py = process_positions[stock.process.id]
            sx, sy = stock_positions[stock.name]
            fig.add_shape(
                type="line",
                x0=px,
                y0=py,
                x1=sx,
                y1=sy,
                line=dict(color=self.flow_color, width=1.5, dash="dash"),
            )

    def _add_nodes(
        self,
        fig: go.Figure,
        items,
        positions,
        key_getter,
        color_getter,
        marker_symbol: str,
        textposition: str,
    ):
        if not items:
            return

        x_vals = []
        y_vals = []
        colors = []
        labels = []
        for item in items:
            key = key_getter(item)
            x, y = positions[key]
            x_vals.append(x)
            y_vals.append(y)
            colors.append(color_getter(item.name))
            labels.append(self.display_name(item.name))

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers+text",
                marker=dict(
                    size=self.node_size,
                    symbol=marker_symbol,
                    color=colors,
                    line=dict(color=self.flow_color, width=1),
                ),
                text=labels,
                textposition=textposition,
                hovertext=labels,
                hoverinfo="text",
            )
        )
