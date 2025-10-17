import pytest
import numpy as np

from flodym import Dimension, DimensionSet, FlodymArray
from flodym.export.array_plotter import PyplotArrayPlotter, PlotlyArrayPlotter


@pytest.fixture
def test_array():
    """Create a test array with multiple dimensions."""
    dims = DimensionSet(
        dim_list=[
            Dimension(name="Time", letter="t", items=[2020, 2021, 2022]),
            Dimension(name="Region", letter="r", items=["EU", "US"]),
        ]
    )
    values = np.random.rand(3, 2)
    return FlodymArray(dims=dims, values=values, name="TestArray")


def test_pyplot_plotter_with_names(test_array):
    """Test PyplotArrayPlotter with dimension names."""
    plotter = PyplotArrayPlotter(array=test_array, intra_line_dim="Time", linecolor_dim="Region")
    assert plotter.intra_line_dim == "Time"
    assert plotter.linecolor_dim == "Region"
    plotter.plot()  # Ensure plotting works without error


def test_pyplot_plotter_with_letters(test_array):
    """Test PyplotArrayPlotter with dimension letters."""
    plotter = PyplotArrayPlotter(array=test_array, intra_line_dim="t", linecolor_dim="r")
    assert plotter.intra_line_dim == "t"
    assert plotter.linecolor_dim == "r"
    plotter.plot()  # Ensure plotting works without error


def test_pyplot_plotter_with_mixed(test_array):
    """Test PyplotArrayPlotter with mixed dimension names and letters."""
    plotter = PyplotArrayPlotter(array=test_array, intra_line_dim="Time", linecolor_dim="r")
    assert plotter.intra_line_dim == "Time"
    assert plotter.linecolor_dim == "r"
    plotter.plot()  # Ensure plotting works without error


def test_plotly_plotter_with_names(test_array):
    """Test PlotlyArrayPlotter with dimension names."""
    plotter = PlotlyArrayPlotter(array=test_array, intra_line_dim="Time", linecolor_dim="Region")
    assert plotter.intra_line_dim == "Time"
    assert plotter.linecolor_dim == "Region"
    plotter.plot()  # Ensure plotting works without error


def test_plotly_plotter_with_letters(test_array):
    """Test PlotlyArrayPlotter with dimension letters."""
    plotter = PlotlyArrayPlotter(array=test_array, intra_line_dim="t", linecolor_dim="r")
    assert plotter.intra_line_dim == "t"
    assert plotter.linecolor_dim == "r"
    plotter.plot()  # Ensure plotting works without error


def test_plotly_plotter_with_mixed(test_array):
    """Test PlotlyArrayPlotter with mixed dimension names and letters."""
    plotter = PlotlyArrayPlotter(array=test_array, intra_line_dim="Time", linecolor_dim="r")
    assert plotter.intra_line_dim == "Time"
    assert plotter.linecolor_dim == "r"
    plotter.plot()  # Ensure plotting works without error


def test_plotter_with_subplot_dim_letter(test_array):
    """Test that subplot_dim also accepts letters."""
    # Create array with 3 dimensions
    dims = DimensionSet(
        dim_list=[
            Dimension(name="Time", letter="t", items=[2020, 2021]),
            Dimension(name="Region", letter="r", items=["EU", "US"]),
            Dimension(name="Element", letter="e", items=["C", "Fe"]),
        ]
    )
    values = np.random.rand(2, 2, 2)
    array = FlodymArray(dims=dims, values=values, name="TestArray3D")

    plotter = PyplotArrayPlotter(
        array=array, intra_line_dim="t", linecolor_dim="r", subplot_dim="e"
    )
    assert plotter.subplot_dim == "e"
    plotter.plot()  # Ensure plotting works without error


def test_plotter_invalid_dimension_name(test_array):
    """Test that invalid dimension names/letters raise an error."""
    with pytest.raises(ValueError, match="not in array dimensions"):
        PyplotArrayPlotter(array=test_array, intra_line_dim="invalid", linecolor_dim="r")


def test_plotter_invalid_dimension_letter(test_array):
    """Test that invalid dimension letters raise an error."""
    with pytest.raises(ValueError, match="not in array dimensions"):
        PyplotArrayPlotter(array=test_array, intra_line_dim="t", linecolor_dim="x")
