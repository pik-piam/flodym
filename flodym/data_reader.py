"""Home to some data readers."""

import warnings
from abc import ABC, abstractmethod
from os import PathLike
from typing import Any

import pandas as pd

from .dimensions import Dimension, DimensionSet
from .flodym_arrays import Parameter
from .mfa_definition import DimensionDefinition, ParameterDefinition, _dimension_dependency_order


class DataReader(ABC):
    """Template for creating a data reader.

    Shows the required methods and data formats needed for use in the MFASystem model.
    """

    def read_dimensions(self, dimension_definitions: list[DimensionDefinition]) -> DimensionSet:
        """Method to read data for multiple dimensions, by looping over `read_dimension`.

        Sub-dimensions (see :py:attr:`flodym.DimensionDefinition.parent`) are linked to their
        parent. Dimensions with :py:attr:`flodym.DimensionDefinition.splitting` are not read, but
        created from their parent with :py:meth:`flodym.Dimension.split`.

        Args:
            dimension_definitions: Definitions of the dimensions to read.

        Returns:
            The dimensions, in the order of the definitions.
        """
        dimensions: dict[str, Dimension] = {}
        for definition in _dimension_dependency_order(dimension_definitions):
            if definition.splitting is not None:
                dimensions[definition.letter] = _split_dimension(definition, dimensions)
                continue
            dimension = self.read_dimension(definition)
            if definition.parent is not None:
                dimension = _link_parent(dimension, definition, dimensions[definition.parent])
            dimensions[definition.letter] = dimension
        return DimensionSet(dim_list=[dimensions[d.letter] for d in dimension_definitions])

    @abstractmethod
    def read_dimension(self, definition: DimensionDefinition) -> Dimension:
        """Required method to read data for a single dimension, given its definition."""

    @abstractmethod
    def read_parameter_values(self, parameter_name: str, dims: DimensionSet) -> Parameter:
        """Required method to read data for a particular parameter."""

    def read_parameters(
        self, parameter_definitions: list[ParameterDefinition], dims: DimensionSet
    ) -> dict[str, Parameter]:
        """Method to read data for a list of parameters, by looping over `read_parameter_values`.

        Args:
            parameter_definitions: Definitions of the parameters to read.
            dims: Dimensions of the MFA system; each parameter receives the subset given by the
                ``dim_letters`` of its definition.

        Returns:
            The parameters, keyed by parameter name.
        """
        parameters = {}
        for parameter_definition in parameter_definitions:
            dim_subset = dims.get_subset(parameter_definition.dim_letters)
            parameters[parameter_definition.name] = self.read_parameter_values(
                parameter_name=parameter_definition.name,
                dims=dim_subset,
            )
        return parameters


class DimensionReader(ABC):
    """Template for creating a dimension reader.

    Shows the required methods and data formats needed.
    """

    read_dimensions = DataReader.read_dimensions

    @abstractmethod
    def read_dimension(self, definition: DimensionDefinition) -> Dimension:
        """Required method to read data for a single dimension, given its definition."""


class CSVDimensionReader(DimensionReader):
    """Read dimensions from a CSV file.

    Expects a single row or single column csv file with no header, containing the dimension items.
    For a dimension with a :py:attr:`flodym.DimensionDefinition.parent`, the file has two columns:
    the items, and the parent item of each item.

    Args:
        dimension_files: {dimension_name: file_path, ...}
        read_csv_kwargs: Additional keyword arguments passed to pandas.read_csv.
            The default is {"header": None}. Not encouraged to use, since it may not lead to the
            intended DataFrame format. Sticking to recommended csv file format is preferred.
    """

    def __init__(
        self,
        dimension_files: dict[str, str | PathLike[str]],
        **read_csv_kwargs: Any,
    ) -> None:
        self.dimension_files = dimension_files
        self.read_csv_kwargs = read_csv_kwargs

    def read_dimension(self, definition: DimensionDefinition) -> Dimension:
        """Read the items of a dimension from the CSV file given for the dimension name."""
        path = self.dimension_files[definition.name]
        if "header" not in self.read_csv_kwargs:
            self.read_csv_kwargs["header"] = None
        df = pd.read_csv(path, **self.read_csv_kwargs)
        return Dimension.from_df(df, definition)


class ExcelDimensionReader(DimensionReader):
    """Read dimensions from Excel file(s).

    Expects a single row or single column excel sheet with no header, containing the dimension
    items. For a dimension with a :py:attr:`flodym.DimensionDefinition.parent`, the sheet has two
    columns: the items, and the parent item of each item.

    Args:
        dimension_files: {dimension_name: file_path, ...}
        dimension_sheets: {dimension_name: sheet_name, ...}. If None, the first sheet is used.
        read_excel_kwargs: Additional keyword arguments passed to pandas.read_excel.
            The default is {"header": None}. Not encouraged to use, since it may not lead to the
            intended DataFrame format. Sticking to recommended excel file format is preferred.
    """

    def __init__(
        self,
        dimension_files: dict[str, str | PathLike[str]],
        dimension_sheets: dict[str, str] | None = None,
        **read_excel_kwargs: Any,
    ) -> None:
        self.dimension_files = dimension_files
        self.dimension_sheets = dimension_sheets
        self.read_excel_kwargs = read_excel_kwargs

    def read_dimension(self, definition: DimensionDefinition) -> Dimension:
        """Read the items of a dimension from the Excel file and sheet given for its name."""
        path = self.dimension_files[definition.name]
        # load data from excel
        if self.dimension_sheets is None:
            sheet_name = 0  # first sheet; None would make pandas read all sheets into a dict
        else:
            sheet_name = self.dimension_sheets[definition.name]
        # default for header is None
        if "header" not in self.read_excel_kwargs:
            self.read_excel_kwargs["header"] = None
        df = pd.read_excel(path, sheet_name=sheet_name, **self.read_excel_kwargs)
        return Dimension.from_df(df, definition)


class ParameterReader(ABC):
    """Template for creating a parameter reader.

    Shows the required methods and data formats needed.
    """

    @abstractmethod
    def read_parameter_values(self, parameter_name: str, dims: DimensionSet) -> Parameter:
        """Required method to read data for a particular parameter."""

    read_parameters = DataReader.read_parameters


class CSVParameterReader(ParameterReader):
    """Reads a csv file to a pandas data frame and calls :py:meth:`flodym.Parameter.from_df`.

    Expects comma separation and no header, apart from optional column names.
    For further detail on expected format, see :py:meth:`flodym.FlodymArray.from_df`.

    Args:
        parameter_files: Mapping of parameter names to file paths.
            Format: {parameter_name: file_path, ...}
        allow_missing_values: Whether to allow missing values in the DataFrame.
            This includes both missing rows, and NaN values in the value column.
            Defaults to False.
        allow_extra_values: Whether to allow extra rows in the DataFrame,
            i.e. rows with index items not present in the FlodymArray dimension items.
            Defaults to False.
        read_csv_kwargs: Additional keyword arguments passed to pandas.read_csv.
            Not encouraged to use, since it may not lead to the intended DataFrame format.
            Sticking to recommended csv file format is preferred.
    """

    def __init__(
        self,
        parameter_files: dict[str, str | PathLike[str]],
        allow_missing_values: bool = False,
        allow_extra_values: bool = False,
        **read_csv_kwargs: Any,
    ) -> None:
        self.parameter_files = parameter_files
        self.allow_missing_values = allow_missing_values
        self.allow_extra_values = allow_extra_values
        self.read_csv_kwargs = read_csv_kwargs

    @property
    def parameter_filenames(self) -> dict[str, str | PathLike[str]]:
        """Deprecated. Use :py:attr:`parameter_files` instead."""
        warnings.warn(
            "parameter_filenames is deprecated since v2.0.0. It will be removed in v3.0.0. "
            "Use parameter_files instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.parameter_files

    @parameter_filenames.setter
    def parameter_filenames(self, value: dict[str, str | PathLike[str]]) -> None:
        warnings.warn(
            "parameter_filenames is deprecated since v2.0.0. It will be removed in v3.0.0. "
            "Use parameter_files instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.parameter_files = value

    def read_parameter_values(self, parameter_name: str, dims: DimensionSet) -> Parameter:
        """Read the values of a parameter from the CSV file given for the parameter name.

        Raises:
            ValueError: If no parameter files were specified.
        """
        if not self.parameter_files:
            raise ValueError("No parameter files specified.")
        datasets_path = self.parameter_files[parameter_name]
        data = pd.read_csv(datasets_path, **self.read_csv_kwargs)
        return Parameter.from_df(
            dims=dims,
            name=parameter_name,
            df=data,
            allow_missing_values=self.allow_missing_values,
            allow_extra_values=self.allow_extra_values,
        )


class ExcelParameterReader(ParameterReader):
    """Reads an excel file to a pandas data frame and calls :py:meth:`flodym.Parameter.from_df`.

    Expects contiguous data starting in the upper left cell A1.
    For further detail on expected format, see :py:meth:`flodym.FlodymArray.from_df`.

    Args:
        parameter_files: Mapping of parameter names to file paths.
            Can be the same file for multiple parameters if the sheets are different.
            Format: {parameter_name: file_path, ...}
        parameter_sheets: Mapping of parameter names to sheet names in the excel file.
            If None, the first sheet is used.
            Format: {parameter_name: sheet_name, ...}
            Defaults to None.
        allow_missing_values: Whether to allow missing values in the DataFrame.
            This includes both missing rows, and NaN values in the value column.
            Defaults to False.
        allow_extra_values: Whether to allow extra rows in the DataFrame,
            i.e. rows with index items not present in the FlodymArray dimension items.
            Defaults to False.
        read_excel_kwargs: Additional keyword arguments passed to pandas.read_excel.
            Not encouraged to use, since it may not lead to the intended DataFrame format.
            Sticking to recommended excel file format is preferred
    """

    def __init__(
        self,
        parameter_files: dict[str, str | PathLike[str]],
        parameter_sheets: dict[str, str] | None = None,
        allow_missing_values: bool = False,
        allow_extra_values: bool = False,
        **read_excel_kwargs: Any,
    ) -> None:
        self.parameter_files = parameter_files
        self.parameter_sheets = parameter_sheets
        self.allow_missing_values = allow_missing_values
        self.allow_extra_values = allow_extra_values
        self.read_excel_kwargs = read_excel_kwargs

    def read_parameter_values(self, parameter_name: str, dims: DimensionSet) -> Parameter:
        """Read the values of a parameter from the Excel file and sheet given for its name."""
        datasets_path = self.parameter_files[parameter_name]
        if self.parameter_sheets is None:
            sheet_name = 0  # first sheet; None would make pandas read all sheets into a dict
        else:
            sheet_name = self.parameter_sheets[parameter_name]
        data = pd.read_excel(datasets_path, sheet_name=sheet_name, **self.read_excel_kwargs)
        return Parameter.from_df(
            dims=dims,
            name=parameter_name,
            df=data,
            allow_missing_values=self.allow_missing_values,
            allow_extra_values=self.allow_extra_values,
        )


class CompoundDataReader(DataReader):
    """Combines a DimensionReader and a ParameterReader to create a DataReader.

    The resulting reader reads both dimensions and parameters.

    Args:
        dimension_reader: Reader used for all dimensions.
        parameter_reader: Reader used for all parameters.
    """

    def __init__(
        self,
        dimension_reader: DimensionReader,
        parameter_reader: ParameterReader,
    ) -> None:
        self.dimension_reader = dimension_reader
        self.parameter_reader = parameter_reader

    def read_dimension(self, definition: DimensionDefinition) -> Dimension:
        """Read a dimension using the dimension reader."""
        return self.dimension_reader.read_dimension(definition)

    def read_parameter_values(self, parameter_name: str, dims: DimensionSet) -> Parameter:
        """Read parameter values using the parameter reader."""
        return self.parameter_reader.read_parameter_values(parameter_name, dims)


def _link_parent(
    dimension: Dimension, definition: DimensionDefinition, parent: Dimension
) -> Dimension:
    """Return a copy of a read dimension, linked to its parent.

    The parent items are converted to the datatype of the parent, if it has one.

    Raises:
        ValueError: If the read dimension has no parent items.
    """
    if dimension.parent_items is None:
        raise ValueError(
            f"Dimension '{definition.name}' has parent '{definition.parent}' in its definition, "
            "but no parent items were read. Give the parent item of each item, e.g. as a second "
            "column in the dimension file."
        )
    parent_items = dimension.parent_items
    if parent.dtype is not None:
        parent_items = [parent.dtype(item) for item in parent_items]
    return Dimension(
        name=dimension.name,
        letter=dimension.letter,
        items=dimension.items,
        dtype=dimension.dtype,
        parent=parent,
        parent_items=parent_items,
    )


def _split_dimension(
    definition: DimensionDefinition, dimensions: dict[str, Dimension]
) -> Dimension:
    """Create a dimension by splitting items of its parent, as given in the definition.

    Args:
        definition: Definition with parent and splitting.
        dimensions: Already created dimensions by letter, including those the definition
            depends on.
    """
    if definition.parent is None or definition.splitting is None:
        raise ValueError(f"Dimension definition '{definition.name}' has no parent or splitting.")
    parent = dimensions[definition.parent]
    splitting = {
        (parent.dtype(item) if parent.dtype is not None else item): dimensions[letter]
        for item, letter in definition.splitting.items()
    }
    split = parent.split(name=definition.name, letter=definition.letter, splitting=splitting)
    return Dimension.model_validate({**dict(split), "dtype": definition.dtype})
