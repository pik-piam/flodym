from matplotlib import pyplot as plt
from plotly import graph_objects as go, colors as plc
from plotly.subplots import make_subplots
import numpy as np
from pydantic import BaseModel as PydanticBaseModel, model_validator, ConfigDict
from typing import Any, Optional, Union
from abc import ABC, abstractmethod

from ..flodym_arrays import FlodymArray
from ..dimensions import DimensionSet
from .helper import CustomNameDisplayer


class ArrayPlotter(CustomNameDisplayer, ABC, PydanticBaseModel):
    """
    Abstract base class for array plotting classes. Subclasses exist for pyplot and plotly.
    Mostly recommended for plotting multi-dimensional arrays, where subplots and multiple lines are needed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow", protected_namespaces=())

    array: FlodymArray
    """Values to plot, usually a Flow or Stock; sliced or summed along excess dimensions."""
    intra_line_dim: str
    """Name of the dimension along which lines are plotted (if no x_array is given, this is also the x-axis)."""
    x_array: Optional[Union[FlodymArray, None]] = None
    """Array with x-values for each line. Must have the same dimensions as array, or a subset of them. If None, the intra_line_dim is used as x-axis."""
    subplot_dim: str = None
    """Name of the dimension by which to split the array into subplots. If None, the array is plotted in a single subplot."""
    linecolor_dim: str = None
    """Name of the dimension along which to split the array into several lines within each subplot. If None, only one line is plotted per subplot."""
    fig: Any = None
    """Pre-existing figure to plot on. If None, a new figure is created."""
    line_label: str = None
    """Custom label for the line. If None, the respective item along linecolor_dim is used as label."""
    xlabel: str = None
    """Custom label for the x-axis. If None, the name of the x_array or intra_line_dim is used."""
    ylabel: str = None
    """Custom label for the y-axis. If None, the name of the array is used."""
    title: str = None
    """Title of the plot, if desired."""
    __pydantic_extra__: dict[str, Any]

    @model_validator(mode="after")
    def check_colors(self):
        if self.linecolor_dim is not None and self.line_label is not None:
            raise ValueError(
                "If linecolor_dim is given, several lines are plotted. In this case, lines are labeled"
                "by items along that dimension, and a line_label must not be given."
            )
        return self

    @model_validator(mode="after")
    def check_dims(self):
        dim_attributes = [self.linecolor_dim, self.subplot_dim, self.intra_line_dim]
        dim_attribute_names = ["linecolor_dim", "subplot_dim", "intra_line_dim"]
        for attr_name, dim in zip(dim_attribute_names, dim_attributes):
            if dim is not None and dim not in self.array.dims.names:
                raise ValueError(f"Dimension {dim} given in {attr_name} not in array dimensions.")
        specified_dim_attributes = [d for d in dim_attributes if d is not None]
        excess_dims = set(self.array.dims.names) - set(specified_dim_attributes)
        if excess_dims:
            raise ValueError(
                "All dimensions of passed array must be given exactly once. Either as subplot_dim, linecolor_dim, or intra_line_dim."
                + f"Excess dimensions: {', '.join(excess_dims)}; "
                + "Sum or slice array along these dims before passing it to the plotter."
            )
        if self.x_array is not None:
            if any(d not in self.array.dims.names for d in self.x_array.dims.names):
                raise ValueError(
                    "x_array must have the same dimensions as array, or a subset of them."
                )
        return self

    def plot(self, save_path: str = None, do_show: bool = False):
        self._fill_fig()
        subplots_array, subplots_x_array = self._prepare_arrays()
        self._plot_all_subplots(subplots_array, subplots_x_array)
        self.plot_legend()
        if self.title is not None:
            self.set_title()
        if save_path is not None:
            self.save(save_path)
        if do_show:
            self.show()
        return self.fig

    def _prepare_arrays(self) -> tuple[dict[str, FlodymArray], dict[str, FlodymArray]]:
        self._get_x_array_like_value_array()
        subplots_array = self._dict_of_slices(self.array, self.subplot_dim).values()
        subplots_x_array = self._dict_of_slices(self.x_array, self.subplot_dim).values()
        return subplots_array, subplots_x_array

    @staticmethod
    def _dict_of_slices(array: FlodymArray, dim_name_to_slice) -> dict[str, FlodymArray]:
        if dim_name_to_slice is not None:
            slice_letter = array.dims[dim_name_to_slice].letter
            return array.split(slice_letter)
        else:
            return {None: array}

    @property
    def _dims_after_slice(self):
        original_dims = self.array.dims.letters
        dims_removed = [d for d, v in self.slice_dict.items() if not isinstance(v, (list, tuple))]
        return [d for d in original_dims if d not in dims_removed]

    def _plot_all_subplots(self, subplotlist_array, subplotlist_x_array):
        for i_subplot, (array_subplot, x_array_subplot) in enumerate(
            zip(subplotlist_array, subplotlist_x_array)
        ):
            self._plot_subplot(i_subplot=i_subplot, array=array_subplot, x_array=x_array_subplot)
            self._label_subplot(i_subplot=i_subplot)

    def _fill_fig(self):
        if self.fig is not None:  # already filled from input argument
            self.nx, self.ny = self._get_nx_ny()
            self.subplot_titles = None
            return
        if self.subplot_dim is None:
            self.nx, self.ny = 1, 1
            self.subplot_titles = None
        else:
            n_subplots = self.array.dims[self.subplot_dim].len
            self.nx, self.ny = self._get_nx_ny_from_n(n_subplots)
            self.subplot_titles = [
                f"{self.display_name(self.subplot_dim)}={self.display_name(item)}"
                for item in self.array.dims[self.subplot_dim].items
            ]
        self.fig = self.get_fig()

    def _get_x_array_like_value_array(self):
        if self.x_array is None:
            x_dim_obj = self.array.dims[self.intra_line_dim]
            x_dimset = DimensionSet(dim_list=[x_dim_obj])
            self.x_array = FlodymArray(
                dims=x_dimset, values=np.array(x_dim_obj.items), name=self.intra_line_dim
            )
        self.x_array = self.x_array.cast_to(self.array.dims)

    def _plot_subplot(self, i_subplot: int, array: FlodymArray, x_array: FlodymArray):
        linedict_array = self._dict_of_slices(array, self.linecolor_dim)
        linedict_x_array = self._dict_of_slices(x_array, self.linecolor_dim)
        for i_line, (array_line, x_array_line, name_line) in enumerate(
            zip(linedict_array.values(), linedict_x_array.values(), linedict_array.keys())
        ):
            label = self.line_label if self.line_label is not None else self.display_name(name_line)
            assert array_line.dims.names == (self.intra_line_dim,), (
                "All dimensions of array must be given exactly once. Either as x_dim / subplot_dim / linecolor_dim, or in "
                "slice_dict or summed_dims."
            )
            self.add_line(i_subplot, x_array_line.values, array_line.values, label, i_line)

    def _label_subplot(self, i_subplot: int):
        if self.subplot_titles is not None:
            self.set_subplot_title(i_subplot, self.subplot_titles[i_subplot])
        xlabel = self.xlabel if self.xlabel is not None else self.display_name(self.x_array.name)
        if xlabel != "unnamed":
            self.set_xlabel(i_subplot, xlabel)
        ylabel = self.ylabel if self.ylabel is not None else self.display_name(self.array.name)
        if ylabel != "unnamed":
            self.set_ylabel(i_subplot, ylabel)

    def _index2d(self, i_subplot):
        return i_subplot // self.nx, i_subplot % self.nx

    @staticmethod
    def _get_nx_ny_from_n(n):
        nx = int(np.ceil(np.sqrt(n)))
        ny = int(np.ceil(n / nx))
        return nx, ny

    @abstractmethod
    def save(self, save_path: str = None):
        raise NotImplementedError

    @abstractmethod
    def show(self):
        raise NotImplementedError

    @abstractmethod
    def _get_nx_ny(self):
        raise NotImplementedError

    @abstractmethod
    def get_fig(self):
        raise NotImplementedError

    @abstractmethod
    def set_xlabel(self, index, label):
        raise NotImplementedError

    @abstractmethod
    def set_ylabel(self, index, label):
        raise NotImplementedError

    @abstractmethod
    def set_subplot_title(self, index, title):
        raise NotImplementedError

    @abstractmethod
    def add_line(self, index, x, y, label, i_line):
        raise NotImplementedError

    @abstractmethod
    def plot_legend(self):
        raise NotImplementedError

    @abstractmethod
    def set_title(self):
        raise NotImplementedError


class PyplotArrayPlotter(ArrayPlotter):
    fig: plt.Figure = None
    """A previously created pyplot figure object, for adding lines to an existing figure.
    If None, a new figure is created.
    """

    def save(self, save_path: str = None, **kwargs):
        self.fig.savefig(save_path, **kwargs)

    def show(self):
        self.fig.show()

    def get_fig(self):
        fig, _ = plt.subplots(self.nx, self.ny)
        return fig

    def _get_nx_ny(self):
        return self._get_nx_ny_from_n(len(self.ax))

    def set_xlabel(self, i_subplot, label):
        self.ax[i_subplot].set_xlabel(label)

    def set_ylabel(self, i_subplot, label):
        self.ax[i_subplot].set_ylabel(label)

    def set_subplot_title(self, i_subplot, title):
        self.ax[i_subplot].set_title(title)

    def add_line(self, i_subplot, x, y, label, i_line):
        self.ax[i_subplot].plot(x, y, label=label)

    def plot_legend(self):
        handles, labels = self.ax[0].get_legend_handles_labels()
        self.fig.legend(handles, labels, loc="lower center")

    def set_title(self):
        self.fig.suptitle(self.title)

    @property
    def ax(self) -> plt.Axes:
        return self.fig.axes


class PlotlyArrayPlotter(ArrayPlotter):

    fig: go.Figure = None
    """A previously created plotly figure object, for adding lines to an existing figure.
    If None, a new figure is created.
    """

    def save(self, save_path: str = None, **kwargs):
        self.fig.write_image(save_path, **kwargs)

    def show(self):
        self.fig.show()

    def get_fig(self):
        fig = make_subplots(self.ny, self.nx, subplot_titles=self.subplot_titles)
        return fig

    def _fill_fig(self):
        super()._fill_fig()
        self.n_previous_lines = getattr(self.fig, "_n_lines_flodym", 0)
        n_current_lines = (
            len(self.array.dims[self.linecolor_dim].items) if self.linecolor_dim is not None else 1
        )
        self.fig._n_lines_flodym = self.n_previous_lines + n_current_lines

    def _get_nx_ny(self):
        grid_ref = self.fig._validate_get_grid_ref()
        nrows = len(grid_ref)
        ncols = len(grid_ref[0])
        return ncols, nrows

    def row(self, i_subplot):
        return self._index2d(i_subplot)[0] + 1

    def col(self, i_subplot):
        return self._index2d(i_subplot)[1] + 1

    def set_xlabel(self, i_subplot, label):
        self.fig.update_xaxes(title_text=label, row=self.row(i_subplot), col=self.col(i_subplot))

    def set_ylabel(self, i_subplot, label):
        self.fig.update_yaxes(title_text=label, row=self.row(i_subplot), col=self.col(i_subplot))

    def set_subplot_title(self, index, title):
        pass  # already set in make_subplots

    def add_line(self, i_subplot, x, y, label, i_line):
        i_color = i_line + self.n_previous_lines
        color = plc.DEFAULT_PLOTLY_COLORS[i_color]
        self.fig.add_trace(
            go.Scatter(x=x, y=y, name=label, line=dict(color=color), showlegend=i_subplot == 0),
            row=self.row(i_subplot),
            col=self.col(i_subplot),
        )

    def plot_legend(self):
        pass

    def set_title(self):
        self.fig.update_layout(title={"text": self.title})
