"""Home to some data readers."""

from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict

from .flodym_arrays import Parameter
from .mfa_definition import DimensionDefinition, ParameterDefinition
from .dimensions import DimensionSet, Dimension


class DataReader:
    """Template for creating a data reader, showing required methods and data formats needed for
    use in the MFASystem model.
    """

    def read_dimensions(self, dimension_definitions: List[DimensionDefinition]) -> DimensionSet:
        """Method to read data for multiple dimensions, by looping over `read_dimension`."""
        dimensions = [self.read_dimension(definition) for definition in dimension_definitions]
        return DimensionSet(dim_list=dimensions)

    @abstractmethod
    def read_dimension(self, dimension_definition: DimensionDefinition) -> Dimension:
        """Required method to read data for a single dimension,
        corresponding to the dimension definition."""
        pass

    @abstractmethod
    def read_parameter_values(self, parameter_name: str, dims: DimensionSet) -> Parameter:
        """Required method to read data for a particular parameter."""
        pass

    def read_parameters(
        self, parameter_definitions: List[ParameterDefinition], dims: DimensionSet
    ) -> Dict[str, Parameter]:
        """Method to read data for a list of parameters, by looping over `read_parameter_values`."""
        parameters = {}
        for parameter_definition in parameter_definitions:
            dim_subset = dims.get_subset(parameter_definition.dim_letters)
            parameters[parameter_definition.name] = self.read_parameter_values(
                parameter_name=parameter_definition.name,
                dims=dim_subset,
            )
        return parameters


class DimensionReader(ABC):
    """Template for creating a dimension reader, showing required methods and data formats needed"""

    read_dimensions = DataReader.read_dimensions

    @abstractmethod
    def read_dimension(self, dimension_definition: DimensionDefinition) -> Dimension:
        pass


class CSVDimensionReader(DimensionReader):
    """Read dimensions from a CSV file. Expects a single row or single columns csv file with no header containing the dimension items.

    Args:
        dimension_files (dict): {dimension_name: file_path, ...}
        read_csv_kwargs: Additional keyword arguments passed to pandas.read_csv.
            The default is {"header": None}. Not encouraged to use, since it may not lead to the
            intended DataFrame format. Sticking to recommended csv file format is preferred.
    """

    def __init__(
        self,
        dimension_files,
        **read_csv_kwargs,
    ):
        self.dimension_files = dimension_files
        self.read_csv_kwargs = read_csv_kwargs

    def read_dimension(self, definition: DimensionDefinition):
        path = self.dimension_files[definition.name]
        if "header" not in self.read_csv_kwargs:
            self.read_csv_kwargs["header"] = None
        df = pd.read_csv(path, **self.read_csv_kwargs)
        return Dimension.from_df(df, definition)


class ExcelDimensionReader(DimensionReader):
    """Read dimensions from Excel file(s). Expects a single row or single columns excel sheet with no header containing the dimension items.

    Args:
        dimension_files (dict): {dimension_name: file_path, ...}
        dimension_sheets (dict): {dimension_name: sheet_name, ...}
        ead_excel_kwargs: Additional keyword arguments passed to pandas.read_excel.
            The default is {"header": None}. Not encouraged to use, since it may not lead to the
            intended DataFrame format. Sticking to recommended excel file format is preferred.
    """

    def __init__(
        self,
        dimension_files: dict,
        dimension_sheets: dict = None,
        **read_excel_kwargs,
    ):
        self.dimension_files = dimension_files
        self.dimension_sheets = dimension_sheets
        self.read_excel_kwargs = read_excel_kwargs

    def read_dimension(self, definition: DimensionDefinition):
        path = self.dimension_files[definition.name]
        # load data from excel
        if self.dimension_sheets is None:
            sheet_name = None
        else:
            sheet_name = self.dimension_sheets[definition.name]
        # default for header is None
        if "header" not in self.read_excel_kwargs:
            self.read_excel_kwargs["header"] = None
        df = pd.read_excel(path, sheet_name=sheet_name, **self.read_excel_kwargs)
        return Dimension.from_df(df, definition)


class ParameterReader(ABC):
    """Template for creating a parameter reader, showing required methods and data formats needed"""

    @abstractmethod
    def read_parameter_values(self, parameter_name: str, dims: DimensionSet) -> Parameter:
        pass

    read_parameters = DataReader.read_parameters


class CSVParameterReader(ParameterReader):
    """Reads a csv file to a pandas data frame and calls :py:meth:`flodym.Parameter.from_df`.
    Expects comma separation and no header, apart from optional column names.
    For further detail on expected format, see :py:meth:`flodym.FlodymArray.from_df`.

    Args:
        parameter_files (dict): Mapping of parameter names to file paths.
            Format: {parameter_name: file_path, ...}
        allow_missing_values (bool, optional): Whether to allow missing values in the DataFrame.
            This includes both missing rows, and NaN values in the value column.
            Defaults to False.
        allow_extra_values (bool, optional): Whether to allow extra rows in the DataFrame,
            i.e. tows with index items not present in the FlodymArray dimension items.
            Defaults to False.
        read_csv_kwargs: Additional keyword arguments passed to pandas.read_csv.
            Not encouraged to use, since it may not lead to the intended DataFrame format.
            Sticking to recommended csv file format is preferred.
    """

    def __init__(
        self,
        parameter_files: dict = None,
        allow_missing_values: bool = False,
        allow_extra_values: bool = False,
        **read_csv_kwargs,
    ):
        self.parameter_filenames = parameter_files  # {parameter_name: file_path, ...}
        self.allow_missing_values = allow_missing_values
        self.allow_extra_values = allow_extra_values
        self.read_csv_kwargs = read_csv_kwargs

    def read_parameter_values(self, parameter_name: str, dims):
        if self.parameter_filenames is None:
            raise ValueError("No parameter files specified.")
        datasets_path = self.parameter_filenames[parameter_name]
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
        parameter_files (dict): Mapping of parameter names to file paths.
            Can be the same file for multiple parameters if the sheets are different.
            Format: {parameter_name: file_path, ...}
        parameter_sheets (dict): Mapping of parameter names to sheet names in the excel file.
            If None, the first sheet is used.
            Format: {parameter_name: sheet_name, ...}
            Defaults to None.
        allow_missing_values (bool, optional): Whether to allow missing values in the DataFrame.
            This includes both missing rows, and NaN values in the value column.
            Defaults to False.
        allow_extra_values (bool, optional): Whether to allow extra rows in the DataFrame,
            i.e. tows with index items not present in the FlodymArray dimension items.
            Defaults to False.
        read_excel_kwargs: Additional keyword arguments passed to pandas.read_excel.
            Not encouraged to use, since it may not lead to the intended DataFrame format.
            Sticking to recommended excel file format is preferred
    """

    def __init__(
        self,
        parameter_files: dict,
        parameter_sheets: dict = None,
        allow_missing_values: bool = False,
        allow_extra_values: bool = False,
        **read_excel_kwargs,
    ):
        self.parameter_files = parameter_files
        self.parameter_sheets = parameter_sheets
        self.allow_missing_values = allow_missing_values
        self.allow_extra_values = allow_extra_values
        self.read_excel_kwargs = read_excel_kwargs

    def read_parameter_values(self, parameter_name: str, dims):
        datasets_path = self.parameter_files[parameter_name]
        if self.parameter_sheets is None:
            sheet_name = None
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
    """Combines a DimensionReader and a ParameterReader to create a DataReader,
    reading both dimensions and parameters.
    """

    def __init__(
        self,
        dimension_reader: DimensionReader,
        parameter_reader: ParameterReader,
    ):
        self.dimension_reader = dimension_reader
        self.parameter_reader = parameter_reader

    def read_dimension(self, dimension_definition: DimensionDefinition) -> Dimension:
        return self.dimension_reader.read_dimension(dimension_definition)

    def read_parameter_values(self, parameter_name: str, dims: DimensionSet) -> Parameter:
        return self.parameter_reader.read_parameter_values(parameter_name, dims)
