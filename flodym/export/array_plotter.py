"""Line, scatter and area plots of multi-dimensional `FlodymArray` objects."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, ValuesView
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from plotly import colors as plc
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, PrivateAttr, model_validator
from typing_extensions import Self

from ..dimensions import DimensionSet
from ..flodym_arrays import FlodymArray
from ._helper import CustomNameDisplayer


class ArrayPlotter(CustomNameDisplayer, ABC, PydanticBaseModel):
    """Abstract base class for array plotting classes.

    Subclasses exist for pyplot and plotly.
    Mostly recommended for plotting multi-dimensional arrays, where subplots and multiple lines are
    needed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow", protected_namespaces=())

    array: FlodymArray
    """Values to plot, usually a Flow or Stock; sliced or summed along excess dimensions."""
    intra_line_dim: str
    """Name or letter of the dimension along which lines are plotted.

    If no x_array is given, this is also the x-axis.
    """
    x_array: FlodymArray | None = None
    """Array with x-values for each line.

    Must have the same dimensions as array, or a subset of them. If None, the intra_line_dim is used
    as x-axis.
    """
    subplot_dim: str | None = None
    """Name or letter of the dimension by which to split the array into subplots.

    If None, the array is plotted in a single subplot.
    """
    linecolor_dim: str | None = None
    """Name or letter of the dimension along which to split the array into lines in each subplot.

    If None, only one line is plotted per subplot.
    """
    fig: Any = None
    """Pre-existing figure to plot on. If None, a new figure is created."""
    line_label: str | None = None
    """Custom label for the line.

    If None, the respective item along linecolor_dim is used as label.
    """
    xlabel: str | None = None
    """Custom label for the x-axis. If None, the name of the x_array or intra_line_dim is used."""
    ylabel: str | None = None
    """Custom label for the y-axis. If None, the name of the array is used."""
    title: str | None = None
    """Title of the plot, if desired."""
    color_map: list[str] | None = None
    """List of colors to use for the lines. If None, a default color map is used."""
    chart_type: str = "line"
    """Type of chart to plot. Can be 'line', 'scatter', or 'area'."""
    line_type: str = "solid"
    """Type of line to plot. Can be 'solid', 'dashed', 'dotted', or 'dashdot'."""
    suppress_legend: bool = False
    """If True, the legend is not shown."""

    _nx: int = PrivateAttr(default=1)
    _ny: int = PrivateAttr(default=1)
    _subplot_titles: list[str] | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _check_chart_type(self) -> Self:
        """Check that ``chart_type`` is supported."""
        if self.chart_type not in self._allowed_chart_types:
            raise ValueError("chart_type must be either 'line' or 'scatter'.")
        return self

    @model_validator(mode="after")
    def _check_line_type(self) -> Self:
        """Check ``line_type``, translating between pyplot and plotly names where possible."""
        if self.line_type not in self._allowed_line_types:
            allowed_changes = {"dash": "dashed", "dot": "dotted", "dashed": "dash", "dotted": "dot"}
            if (
                self.line_type in allowed_changes
                and allowed_changes[self.line_type] in self._allowed_line_types
            ):
                self.line_type = allowed_changes[self.line_type]
            else:
                raise ValueError(f"line_type must be one of {self._allowed_line_types}.")
        if self.chart_type != "line" and self.line_type != "solid":
            raise ValueError("line_type is only applicable to chart_type 'line'.")
        return self

    @model_validator(mode="after")
    def _check_colors(self) -> Self:
        """Check that ``line_label`` is not combined with ``linecolor_dim``."""
        if self.linecolor_dim is not None and self.line_label is not None:
            raise ValueError(
                "If linecolor_dim is given, several lines are plotted. In this case, lines are labeled"
                "by items along that dimension, and a line_label must not be given."
            )
        return self

    @model_validator(mode="after")
    def _check_dims(self) -> Self:
        """Check that each array dimension is assigned exactly once and ``x_array`` fits."""
        dim_attributes = [self.linecolor_dim, self.subplot_dim, self.intra_line_dim]
        dim_attribute_names = ["linecolor_dim", "subplot_dim", "intra_line_dim"]
        for attr_name, dim in zip(dim_attribute_names, dim_attributes):
            if dim is not None and dim not in self.array.dims:
                raise ValueError(f"Dimension {dim} given in {attr_name} not in array dimensions.")
        specified_dim_attributes = [d for d in dim_attributes if d is not None]
        # Convert specified dimensions to names for comparison with all dimension names
        specified_dim_names = [self.array.dims[d].name for d in specified_dim_attributes]
        excess_dims = set(self.array.dims.names) - set(specified_dim_names)
        if excess_dims:
            raise ValueError(
                "All dimensions of passed array must be given exactly once. Either as subplot_dim, linecolor_dim, or intra_line_dim."
                + f"Excess dimensions: {', '.join(excess_dims)}; "
                + "Sum or slice array along these dims before passing it to the plotter."
            )
        if self.x_array is not None and any(
            d not in self.array.dims for d in self.x_array.dims.names
        ):
            raise ValueError("x_array must have the same dimensions as array, or a subset of them.")
        return self

    @property
    def _allowed_line_types(self) -> list[str]:
        raise NotImplementedError

    @property
    def _allowed_chart_types(self) -> list[str]:
        return ["line", "area", "scatter"]

    def plot(self, save_path: str | None = None, do_show: bool = False) -> Any:
        """Plot the array.

        Args:
            save_path: If given, the figure is saved to this path.
            do_show: Whether to show the figure.

        Returns:
            The figure, of the type of the respective backend.
        """
        self._fill_fig()
        subplots_array, subplots_x_array = self._prepare_arrays()
        self._plot_all_subplots(subplots_array, subplots_x_array)
        self._plot_legend()
        if self.title is not None:
            self._set_title()
        if save_path is not None:
            self.save(save_path)
        if do_show:
            self.show()
        return self.fig

    def _prepare_arrays(self) -> tuple[ValuesView[FlodymArray], ValuesView[FlodymArray]]:
        self._get_x_array_like_value_array()
        subplots_array = self._dict_of_slices(self.array, self.subplot_dim).values()
        subplots_x_array = self._dict_of_slices(self.x_array, self.subplot_dim).values()
        return subplots_array, subplots_x_array

    @staticmethod
    def _dict_of_slices(
        array: FlodymArray, dim_name_to_slice: str | None
    ) -> dict[Any, FlodymArray]:
        if dim_name_to_slice is not None:
            slice_letter = array.dims[dim_name_to_slice].letter
            return array.split(slice_letter)
        else:
            return {None: array}

    def _plot_all_subplots(
        self,
        subplotlist_array: Iterable[FlodymArray],
        subplotlist_x_array: Iterable[FlodymArray],
    ) -> None:
        for i_subplot, (array_subplot, x_array_subplot) in enumerate(
            zip(subplotlist_array, subplotlist_x_array)
        ):
            self._plot_subplot(i_subplot=i_subplot, array=array_subplot, x_array=x_array_subplot)
            self._label_subplot(i_subplot=i_subplot)

    def _fill_fig(self) -> None:
        if self.fig is not None:  # already filled from input argument
            self._nx, self._ny = self._get_nx_ny()
            self._subplot_titles = None
            return
        if self.subplot_dim is None:
            self._nx, self._ny = 1, 1
            self._subplot_titles = None
        else:
            n_subplots = self.array.dims[self.subplot_dim].len
            self._nx, self._ny = self._get_nx_ny_from_n(n_subplots)
            self._subplot_titles = [
                f"{self.display_name(self.subplot_dim)}={self.display_name(item)}"
                for item in self.array.dims[self.subplot_dim].items
            ]
        self.fig = self._get_fig()

    def _get_x_array_like_value_array(self) -> None:
        if self.x_array is None:
            x_dim_obj = self.array.dims[self.intra_line_dim]
            x_dimset = DimensionSet(dim_list=[x_dim_obj])
            self.x_array = FlodymArray(
                dims=x_dimset, values=np.array(x_dim_obj.items), name=self.intra_line_dim
            )
        self.x_array = self.x_array.cast_to(self.array.dims)

    def _plot_subplot(self, i_subplot: int, array: FlodymArray, x_array: FlodymArray) -> None:
        linedict_array = self._dict_of_slices(array, self.linecolor_dim)
        linedict_x_array = self._dict_of_slices(x_array, self.linecolor_dim)
        prev_y = None
        # Get the dimension name for intra_line_dim (in case a letter was provided)
        intra_line_dim_name = self.array.dims[self.intra_line_dim].name
        for i_line, (array_line, x_array_line, name_line) in enumerate(
            zip(linedict_array.values(), linedict_x_array.values(), linedict_array.keys())
        ):
            label = self.line_label if self.line_label is not None else self.display_name(name_line)
            assert array_line.dims.names == (intra_line_dim_name,), (
                "All dimensions of array must be given exactly once. Either as x_dim / subplot_dim / linecolor_dim, or in "
                "slice_dict or summed_dims."
            )
            self._add_line(i_subplot, x_array_line.values, array_line.values, prev_y, label, i_line)
            prev_y = array_line.values

    def _label_subplot(self, i_subplot: int) -> None:
        if self._subplot_titles is not None:
            self._set_subplot_title(i_subplot, self._subplot_titles[i_subplot])
        xlabel = self.xlabel if self.xlabel is not None else self.display_name(self.x_array.name)
        if xlabel != "unnamed":
            self._set_xlabel(i_subplot, xlabel)
        ylabel = self.ylabel if self.ylabel is not None else self.display_name(self.array.name)
        if ylabel != "unnamed":
            self._set_ylabel(i_subplot, ylabel)

    def _index2d(self, i_subplot: int) -> tuple[int, int]:
        return i_subplot // self._nx, i_subplot % self._nx

    @staticmethod
    def _get_nx_ny_from_n(n: int) -> tuple[int, int]:
        nx = int(np.ceil(np.sqrt(n)))
        ny = int(np.ceil(n / nx))
        return nx, ny

    @abstractmethod
    def save(self, save_path: str) -> None:
        """Save the figure to ``save_path``."""
        raise NotImplementedError

    @abstractmethod
    def show(self) -> None:
        """Show the figure."""
        raise NotImplementedError

    @abstractmethod
    def _get_nx_ny(self) -> tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def _get_fig(self) -> Any:
        """Create a new figure with the required subplots."""
        raise NotImplementedError

    @abstractmethod
    def _set_xlabel(self, i_subplot: int, label: str) -> None:
        """Set the x-axis label of the subplot with the given index."""
        raise NotImplementedError

    @abstractmethod
    def _set_ylabel(self, i_subplot: int, label: str) -> None:
        """Set the y-axis label of the subplot with the given index."""
        raise NotImplementedError

    @abstractmethod
    def _set_subplot_title(self, i_subplot: int, title: str) -> None:
        """Set the title of the subplot with the given index."""
        raise NotImplementedError

    @abstractmethod
    def _add_line(
        self,
        i_subplot: int,
        x: np.ndarray,
        y: np.ndarray,
        prev_y: np.ndarray | None,
        label: str,
        i_line: int,
    ) -> None:
        """Add a line to a subplot.

        Args:
            i_subplot: Index of the subplot.
            x: x-values of the line.
            y: y-values of the line.
            prev_y: y-values of the previous line in the same subplot, used as lower bound for
                area charts. None for the first line.
            label: Label of the line.
            i_line: Index of the line within the subplot, used to select its color.
        """
        raise NotImplementedError

    @abstractmethod
    def _plot_legend(self) -> None:
        """Add a legend to the figure."""
        raise NotImplementedError

    @abstractmethod
    def _set_title(self) -> None:
        """Set the title of the figure."""
        raise NotImplementedError


class PyplotArrayPlotter(ArrayPlotter):
    """Array plotter using matplotlib.pyplot."""

    fig: plt.Figure | None = None
    """A previously created pyplot figure object, for adding lines to an existing figure.
    If None, a new figure is created.
    """

    @property
    def _ax(self) -> list[plt.Axes]:
        """Axes of the figure."""
        return self.fig.axes

    @property
    def _allowed_line_types(self) -> list[str]:
        return ["solid", "dashed", "dotted", "dashdot"]

    def save(self, save_path: str, **kwargs: Any) -> None:
        """Save the figure to ``save_path``, passing ``kwargs`` to ``Figure.savefig``."""
        self.fig.savefig(save_path, **kwargs)

    def show(self) -> None:
        """Show the figure."""
        self.fig.show()

    def _get_fig(self) -> plt.Figure:
        """Create a new pyplot figure with the required subplots."""
        fig, _ = plt.subplots(self._nx, self._ny)
        return fig

    def _get_nx_ny(self) -> tuple[int, int]:
        return self._get_nx_ny_from_n(len(self._ax))

    def _set_xlabel(self, i_subplot: int, label: str) -> None:
        """Set the x-axis label of the subplot with the given index."""
        self._ax[i_subplot].set_xlabel(label)

    def _set_ylabel(self, i_subplot: int, label: str) -> None:
        """Set the y-axis label of the subplot with the given index."""
        self._ax[i_subplot].set_ylabel(label)

    def _set_subplot_title(self, i_subplot: int, title: str) -> None:
        """Set the title of the subplot with the given index."""
        self._ax[i_subplot].set_title(title)

    def _add_line(
        self,
        i_subplot: int,
        x: np.ndarray,
        y: np.ndarray,
        prev_y: np.ndarray | None,
        label: str,
        i_line: int,
    ) -> None:
        """Add a line, scatter or area plot to the subplot with index ``i_subplot``."""
        common_dict: dict[str, Any] = {}
        if not self.suppress_legend:
            common_dict["label"] = label
        if self.color_map is not None:
            common_dict["color"] = self.color_map[i_line]

        if self.chart_type == "line":
            self._ax[i_subplot].plot(x, y, linestyle=self.line_type, **common_dict)
        elif self.chart_type == "scatter":
            self._ax[i_subplot].scatter(x, y, **common_dict)
        elif self.chart_type == "area":
            args = [x, y] if prev_y is None else [x, y, prev_y]
            self._ax[i_subplot].fill_between(*args, **common_dict)

    def _plot_legend(self) -> None:
        """Add a legend below the subplots, using the labels of the first subplot."""
        handles, labels = self._ax[0].get_legend_handles_labels()
        self.fig.legend(handles, labels, loc="lower center")

    def _set_title(self) -> None:
        """Set the title of the figure."""
        self.fig.suptitle(self.title)


class PlotlyArrayPlotter(ArrayPlotter):
    """Array plotter using plotly."""

    fig: go.Figure | None = None
    """A previously created plotly figure object, for adding lines to an existing figure.
    If None, a new figure is created.
    """
    color_map: list[str] | None = plc.qualitative.Dark24
    """List of colors to use for the lines. If None, the plotly default colors are used."""

    _n_previous_lines: int = PrivateAttr(default=0)

    @property
    def _allowed_line_types(self) -> list[str]:
        return ["solid", "dash", "dot", "dashdot"]

    def save(self, save_path: str, **kwargs: Any) -> None:
        """Save the figure to ``save_path``, passing ``kwargs`` to ``Figure.write_image``."""
        self.fig.write_image(save_path, **kwargs)

    def show(self) -> None:
        """Show the figure."""
        self.fig.show()

    def _get_fig(self) -> go.Figure:
        """Create a new plotly figure with the required subplots."""
        fig = make_subplots(self._ny, self._nx, subplot_titles=self._subplot_titles)
        return fig

    def _fill_fig(self) -> None:
        super()._fill_fig()
        self._n_previous_lines = getattr(self.fig, "_n_lines_flodym", 0)
        n_current_lines = (
            len(self.array.dims[self.linecolor_dim].items) if self.linecolor_dim is not None else 1
        )
        self.fig._n_lines_flodym = self._n_previous_lines + n_current_lines

    def _get_nx_ny(self) -> tuple[int, int]:
        grid_ref = self.fig._validate_get_grid_ref()
        nrows = len(grid_ref)
        ncols = len(grid_ref[0])
        return ncols, nrows

    def _row(self, i_subplot: int) -> int:
        """Return the 1-based plotly row of the subplot with index ``i_subplot``."""
        return self._index2d(i_subplot)[0] + 1

    def _col(self, i_subplot: int) -> int:
        """Return the 1-based plotly column of the subplot with index ``i_subplot``."""
        return self._index2d(i_subplot)[1] + 1

    def _set_xlabel(self, i_subplot: int, label: str) -> None:
        """Set the x-axis label of the subplot with the given index."""
        self.fig.update_xaxes(title_text=label, row=self._row(i_subplot), col=self._col(i_subplot))

    def _set_ylabel(self, i_subplot: int, label: str) -> None:
        """Set the y-axis label of the subplot with the given index."""
        self.fig.update_yaxes(title_text=label, row=self._row(i_subplot), col=self._col(i_subplot))

    def _set_subplot_title(self, i_subplot: int, title: str) -> None:
        """Do nothing, as subplot titles are already set when creating the figure."""
        # already set in make_subplots

    def _add_line(
        self,
        i_subplot: int,
        x: np.ndarray,
        y: np.ndarray,
        prev_y: np.ndarray | None,
        label: str,
        i_line: int,
    ) -> None:
        """Add a line, scatter or area trace to the subplot with index ``i_subplot``.

        Raises:
            ValueError: If ``chart_type`` is not supported.
        """
        i_color = i_line + self._n_previous_lines
        color = self.color_map[i_color] if self.color_map is not None else None
        common_dict: dict[str, Any] = {
            "x": x,
            "y": y,
            "name": label,
            "showlegend": i_subplot == 0 and not self.suppress_legend,
        }
        if self.chart_type == "line":
            trace = dict(
                type="scatter",
                **common_dict,
                line={"color": color, "dash": self.line_type},
            )
        elif self.chart_type == "scatter":
            trace = dict(
                type="scatter",
                **common_dict,
                mode="markers",
                marker={"color": color},
            )
        elif self.chart_type == "area":
            trace = dict(
                type="scatter",
                **common_dict,
                fill="tozeroy" if prev_y is None else "tonexty",
                fillcolor=color,
                line={"color": color},
            )
        else:
            raise ValueError("chart_type must be either 'line' or 'scatter'.")
        self.fig.add_trace(
            trace=trace,
            row=self._row(i_subplot),
            col=self._col(i_subplot),
        )

    def _plot_legend(self) -> None:
        """Do nothing, as plotly adds the legend automatically."""

    def _set_title(self) -> None:
        """Set the title of the figure."""
        self.fig.update_layout(title={"text": self.title})
