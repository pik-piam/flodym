from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Dict
import yaml

from .named_dim_arrays import Parameter
from .mfa_definition import DimensionDefinition, ParameterDefinition
from .dimensions import DimensionSet, Dimension


class DataReader(ABC):
    """Template for creating a data reader, showing required methods and data formats needed for
    use in the MFASystem model.
    """

    def read_dimensions(self, dimension_definitions: List[DimensionDefinition]) -> DimensionSet:
        dimensions = [self.read_dimension(definition) for definition in dimension_definitions]
        return DimensionSet(dim_list=dimensions)

    @abstractmethod
    def read_dimension(self, dimension_definition: DimensionDefinition) -> Dimension:
        pass

    def read_scalar_data(self, parameters: List[str]) -> dict:
        """Optional addition method if additional scalar parameters are required."""
        pass

    @abstractmethod
    def read_parameter_values(self, parameter: str, dims: DimensionSet) -> Parameter:
        pass

    def read_parameters(
        self, parameter_definitions: List[ParameterDefinition], dims: DimensionSet
    ) -> Dict[str, Parameter]:
        parameters = {}
        for parameter in parameter_definitions:
            dim_subset = dims.get_subset(parameter.dim_letters)
            parameters[parameter.name] = self.read_parameter_values(
                parameter=parameter.name,
                dims=dim_subset,
            )
        return parameters


class ExampleDataReader(DataReader):
    """Example data reader, that reads .csv files for parameters and dimensions, and a yaml file
    for scalar parameter data. File locations need to be specified on initialization.

    **Example**

        >>> from sodym import ExampleDataReader, DimensionDefinition
        >>> time_definition = DimensionDefinition(name='time', letter='t', dtype=int)
        >>> dimension_datasets = {
        >>>     'time': 'path/to/file_with_list_of_times.csv',
        >>>     'place': 'path/to/file_with_list_of_places.csv',
        >>>     ...
        >>> }
        >>> data_reader = ExampleDataReader(dimension_datasets=dimension_datasets, ...)
        >>> time_dimension = data_reader.read_dimension(definition=time_definition)

    """

    def __init__(self, scalar_data_yaml: str, parameter_datasets: dict, dimension_datasets: dict):
        self.scalar_data_yaml = scalar_data_yaml  # file_path
        self.parameter_datasets = parameter_datasets  # {parameter_name: file_path, ...}
        self.dimension_datasets = dimension_datasets  # {dimension_name: file_path, ...}

    def read_scalar_data(self, parameters: List[str]):
        with open(self.scalar_data_yaml, "r") as stream:
            data = yaml.safe_load(stream)
        return {name: data[name] for name in data if name in parameters}

    def read_parameter_values(self, parameter: str, dims):
        datasets_path = self.parameter_datasets[parameter]
        data = pd.read_csv(datasets_path)
        values = self.get_np_from_df(data, dims.names)
        return Parameter(dims=dims, values=values)

    def read_dimension(self, definition: DimensionDefinition):
        path = self.dimension_datasets[definition.name]
        data = np.loadtxt(path, dtype=definition.dtype, delimiter=";").tolist()
        # catch size one lists, which are transformed to scalar by np.ndarray.tolist()
        data = data if isinstance(data, list) else [data]
        return Dimension(name=definition.name, letter=definition.letter, items=data)

    @staticmethod
    def get_np_from_df(df_in: pd.DataFrame, dims: tuple):
        df = df_in.copy()
        dim_columns = [d for d in dims if d in df.columns]
        value_cols = np.setdiff1d(df.columns, dim_columns)
        df.set_index(dim_columns, inplace=True)
        df = df.sort_values(by=dim_columns)

        # check for sparsity
        if df.index.has_duplicates:
            raise Exception("Double entry in df!")
        shape_out = df.index.shape if len(dim_columns) == 1 else df.index.levshape
        if np.prod(shape_out) != df.index.size:
            raise Exception("Dataframe is missing values!")

        if np.any(value_cols != "value"):
            out = {vc: df[vc].values.reshape(shape_out) for vc in value_cols}
        else:
            out = df["value"].values.reshape(shape_out)
        return out
