import sys
import logging
import numpy as np
import pandas as pd
from typing import Literal, Optional, TYPE_CHECKING, Iterable
from pydantic import BaseModel as PydanticBaseModel
import itertools

from .dimensions import Dimension

if TYPE_CHECKING:
    from .flodym_arrays import FlodymArray


class FlodymDataFormat(PydanticBaseModel):

    type: Literal["long", "wide"]
    value_column: str = "value"
    columns_dim: Optional[str] = None


class DataFrameToFlodymDataConverter:
    """Converts a panda DataFrame with various possible formats to a numpy array that can be used
    as values of a FlodymArray.

    Usually not called by the user, but from within the FlodymArray from_df and
    set_values_from_df methods.

    In case of errors, turning on debug logging might help to understand the process.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        flodym_array: "FlodymArray",
        allow_missing_values: bool = False,
        allow_extra_values: bool = False,
    ):
        self.df = df.copy()
        self.flodym_array = flodym_array
        self.allow_missing_values = allow_missing_values
        self.allow_extra_values = allow_extra_values
        try:
            self.target_values = self.get_target_values()
        except Exception as e:
            # add error context to all errors to ease debugging
            raise type(e)(f"{self.error_context} {str(e)}").with_traceback(sys.exc_info()[2])

    def get_target_values(self) -> np.ndarray:
        logging.debug(
            f"Start setting values for FlodymArray {self.flodym_array.name} with dimensions {self.flodym_array.dims.names} from dataframe."
        )
        self._reset_non_default_index()
        self._determine_format()
        self._df_to_long_format()
        self._check_missing_dim_columns()
        self._convert_type()
        self._sort_columns()
        values = self._check_data_complete()
        return values

    def _reset_non_default_index(self):
        if isinstance(self.df.index, pd.MultiIndex):
            self.df.reset_index(inplace=True)
        elif self.df.index.name is not None:
            self.df.reset_index(inplace=True)
        elif self.df.index.dtype != np.int64:
            self.df.reset_index(inplace=True)
        elif self.df.index.min() >= 1700 and self.df.index.max() <= 2300:
            self.df.reset_index(inplace=True)

    def _determine_format(self):
        self._get_dim_columns_by_name()
        self._check_if_first_row_are_items()
        self._check_for_dim_columns_by_items()
        self._check_value_columns()

    def _get_dim_columns_by_name(self):
        self.dim_columns = [c for c in self.df.columns if c in self.flodym_array.dims.names]
        logging.debug(f"Recognized index columns by name: {self.dim_columns}")

    def _check_if_first_row_are_items(self):
        """If data without columns names was read, but the first row was assumed to be column names,
        the first row of the data frame might erroneously end up as column names.
        This method checks if that is the case, and if so, prepends a row based on column names.
        """
        column_name = self.df.columns[0]
        col_items = self.df[column_name].unique()
        extended_col_items = [column_name] + col_items.tolist()
        for dim in self.flodym_array.dims:
            if self.same_items(extended_col_items, dim):
                self._add_column_names_as_row(column_name, dim)

    def _add_column_names_as_row(self, column_name: str, dim: Dimension):
        if len(self.dim_columns) > 0:
            raise ValueError(
                f"Ambiguouity detected: column with first item {column_name} could be "
                f"dimension {dim.name} if the first row counts as an item, but columns "
                f"{self.dim_columns} are already recognized as dimensions first row as name."
                f" Please change the item names of the affected dimension, or use a"
                f" different method to read data."
            )
        # prepend a row to df with all the column names
        self.df = pd.concat([pd.DataFrame([self.df.columns], columns=self.df.columns), self.df])
        # rename columns to range from 0 to n, save original name
        column_name = f"column {self.df.columns.get_loc(column_name)}"
        self.df.columns = [f"column {i}" for i in range(len(self.df.columns))]

    def _check_for_dim_columns_by_items(self):
        for cn in self.df.columns:
            if cn in self.dim_columns:
                continue
            found = self._check_if_dim_column_by_items(cn)
            if not found:
                logging.debug(
                    f"Could not find dimension with same items as column {cn}. "
                    "Assuming this is the first value column; Won't look further."
                )
                return

    def _check_if_dim_column_by_items(self, column_name: str) -> bool:
        logging.debug(f"Checking if {column_name} is a dimension by comparing items with dim items")
        col_items = self.df[column_name].unique()
        for dim in self.flodym_array.dims:
            if self.same_items(col_items, dim):
                logging.debug(f"{column_name} is dimension {dim.name}.")
                self.df.rename(columns={column_name: dim.name}, inplace=True)
                self.dim_columns.append(dim.name)
                return True
        return False

    def _check_value_columns(self):
        value_cols = np.setdiff1d(list(self.df.columns), self.dim_columns)
        logging.debug(f"Assumed value columns: {value_cols}")
        value_cols_are_dim_items = self._check_if_value_columns_match_dim_items(value_cols)
        if not value_cols_are_dim_items:
            self._check_if_valid_long_format(value_cols)

    def _check_if_value_columns_match_dim_items(self, value_cols: list[str]) -> bool:
        logging.debug("Trying to match set of value column names with items of dimension.")
        for dim in self.flodym_array.dims:
            if self.same_items(value_cols, dim):
                logging.debug(f"Value columns match dimension items of {dim.name}.")
                self.format = FlodymDataFormat(type="wide", columns_dim=dim.name)
                if dim.dtype is not None:
                    for c in value_cols:
                        self.df.rename(columns={c: dim.dtype(c)}, inplace=True)
                return True
        return False

    def _check_if_valid_long_format(self, value_cols: list[str]):
        logging.debug(
            "Could not find dimension with same item set as value column names. Assuming long format, i.e. one value column."
        )
        if len(value_cols) == 1:
            self.format = FlodymDataFormat(type="long", value_column=value_cols[0])
            logging.debug(f"Value column name is {value_cols[0]}.")
        else:
            raise ValueError(
                f"More than one value columns. Could not find a dimension the items of which match the set of value column names. "
                f"Value columns: {value_cols}. Please check input data for format, typos, data types and missing items."
            )

    def _df_to_long_format(self):
        if self.format.type != "wide":
            return
        logging.debug("Converting wide format to long format.")
        value_cols = self.flodym_array.dims[self.format.columns_dim].items
        self.df = self.df.melt(
            id_vars=[c for c in self.df.columns if c not in value_cols],
            value_vars=value_cols,
            var_name=self.format.columns_dim,
            value_name=self.format.value_column,
        )
        self.dim_columns.append(self.format.columns_dim)
        self.format = FlodymDataFormat(type="long", value_column=self.format.value_column)

    def _check_missing_dim_columns(self):
        missing_dim_columns = np.setdiff1d(list(self.flodym_array.dims.names), self.dim_columns)
        for c in missing_dim_columns:
            if len(self.flodym_array.dims[c].items) == 1:
                self.df[c] = self.flodym_array.dims[c].items[0]
                self.dim_columns.append(c)
            else:
                raise ValueError(
                    f"Dimension {c} from array has more than one item, but is not found in df. Please specify column in dataframe."
                )

    def _convert_type(self):
        for dim in self.flodym_array.dims:
            if dim.dtype is not None:
                self.df[dim.name] = self.df[dim.name].map(dim.dtype)
        self.df[self.format.value_column] = self.df[self.format.value_column].astype(np.float64)

    def _sort_columns(self):
        """Sort the columns of the data frame according to the order of the dimensions in the
        FlodymArray.
        """
        self.df = self.df[list(self.flodym_array.dims.names) + [self.format.value_column]]

    def _check_data_complete(self):
        # check for double entries in the index columns
        indices = self.df[list(self.flodym_array.dims.names)]
        if indices.duplicated().any():
            raise ValueError(
                f"The following index combinations occur more than once in the data: ",
                indices[indices.duplicated()],
            )

        # remove rows with extra values or throw error
        if self.allow_extra_values:
            for dim in self.flodym_array.dims:
                self.df = self.df[self.df[dim.name].isin(dim.items)]
        else:
            for dim in self.flodym_array.dims:
                unique_items = set(self.df[dim.name].unique())
                extra_items = unique_items - set(dim.items)
                if extra_items:
                    raise ValueError(
                        f"Dimension column '{dim.name}' contains items that are not in the dimension: {extra_items}"
                    )

        if self.allow_missing_values:
            # fill na values with zeros
            self.df[self.format.value_column].fillna(0, inplace=True)
        else:
            if len(self.df) != self.flodym_array.size:
                # print warning first, as compiling expected index tuples may take long
                logging.warning(
                    f"{self.error_context} Detected missing values in the data, but "
                    f"allow_missing_values is set to False. Expected {self.flodym_array.size} "
                    f"rows, but only got {len(self.df)}. Computing missing values...."
                )
                expected_index_tuples = set(
                    itertools.product(*[dim.items for dim in self.flodym_array.dims])
                )
                indices = self.df[list(self.flodym_array.dims.names)]
                actual_index_tuples = set(indices.itertuples(index=False, name=None))
                unexpected_items = actual_index_tuples - expected_index_tuples
                raise ValueError(
                    f"Detected missing values in the data, but allow_missing_values is set to False. "
                    f"Missing values for index combinations: {unexpected_items}."
                )
            if any(self.df[self.format.value_column].isna()):
                raise ValueError("Empty cells/NaN values in value column!")

        # for performance and memory reasons, we use numpy operations to convert the data to the numpy array
        # replace non-value items in df with their index in dims.items
        for dim in self.flodym_array.dims:
            self.df[dim.name] = self.df[dim.name].map({item: i for i, item in enumerate(dim.items)})
        # convert df to numpy index array: Take all non-value columns
        fill_indices = self.df.values[:, :-1].T.astype(np.int16)
        fill_values = self.df[self.format.value_column].values
        # this is what ends up in the parameter; initialize with zeros
        values = np.zeros(self.flodym_array.dims.shape())
        values[tuple(fill_indices)] = fill_values
        return values

    @staticmethod
    def same_items(arr: Iterable, dim: Dimension) -> bool:
        if dim.dtype is not None:
            try:
                arr = [dim.dtype(a) for a in arr]
            except ValueError:
                return False
        return len(set(arr).symmetric_difference(set(dim.items))) == 0

    @property
    def error_context(self) -> str:
        return f"While setting values of {self.flodym_array.__class__.__name__} '{self.flodym_array.name}' from DataFrame:"
