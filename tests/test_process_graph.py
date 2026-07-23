import pytest

from flodym import MFASystem
from flodym.example_objects import get_example_mfa
from flodym.export import (
    GraphvizProcessGraphPlotter,
    PlotlyProcessGraphPlotter,
)


@pytest.fixture
def mfa():
    return get_example_mfa()


def test_exclude_process_hides_it_and_its_flows(mfa: MFASystem):
    plotter = PlotlyProcessGraphPlotter(mfa=mfa, exclude_processes=["sysenv"])
    assert "sysenv" not in {p.name for p in plotter.shown_processes}
    for flow in plotter.shown_flows:
        assert flow.from_process.name != "sysenv"
        assert flow.to_process.name != "sysenv"

def test_exclude_flows(mfa: MFASystem):
    flow_name = next(iter(mfa.flows))
    plotter = PlotlyProcessGraphPlotter(mfa=mfa, exclude_flows=[flow_name])
    assert flow_name not in [flow.name for flow in plotter.shown_flows]

def test_display_names_applied(mfa: MFASystem):
    plotter = PlotlyProcessGraphPlotter(mfa=mfa, display_names={"shredder": "Shredder Plant"})
    fig = plotter.plot()
    all_text = [t for trace in fig.data for t in (trace.text or [])]
    assert "Shredder Plant" in all_text
    assert "shredder" not in all_text


def test_layering_by_longest_path(mfa: MFASystem):
    plotter = PlotlyProcessGraphPlotter(mfa=mfa)
    layers = plotter._calculate_process_layers()
    ids = {process.name: process.id for process in mfa.processes.values()}
    # sysenv feeds shredder, which feeds remelting
    assert layers[ids["sysenv"]] < layers[ids["shredder"]] < layers[ids["remelting"]]


def test_graphviz_process_graph_source(mfa: MFASystem):
    graphviz = pytest.importorskip("graphviz")
    dot = GraphvizProcessGraphPlotter(mfa=mfa).plot()
    assert isinstance(dot, graphviz.Digraph)
    source = dot.source
    for process in mfa.processes.values():
        assert process.name in source
    for stock in mfa.stocks.values():
        assert stock.name in source
