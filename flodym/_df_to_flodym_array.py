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

    def __init__(self, df: pd.DataFrame, flodym_array: "FlodymArray"):
        self.df = df.copy()
        self.flodym_array = flodym_array
        self.target_values = self.get_target_values()

    def get_target_values(self) -> np.ndarray:
        logging.debug(
            f"Start setting values for FlodymArray {self.flodym_array.name} with dimensions {self.flodym_array.dims.names} from dataframe."
        )
        self._reset_non_default_index()
        self._determine_format()
        self._df_to_long_format()
        self._check_missing_dim_columns()
        self._convert_type()
        self._sort_df()
        self._check_data_complete()
        return self.df[self.format.value_column].values.reshape(self.flodym_array.shape)

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
                "More than one value columns. Could not find a dimension the items of which match the set of value column names. "
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

    def _sort_df(self):
        """Sort the columns of the data frame according to the order of the dimensions in the
        FlodymArray.
        Sort the rows of the data frame according to the order of the dimension items in the
        FlodymArray.
        """
        # sort columns
        self.df = self.df[list(self.flodym_array.dims.names) + [self.format.value_column]]
        # sort rows
        self.df = self.df.sort_values(
            by=list(self.flodym_array.dims.names),
            key=lambda x: x.map(lambda y: self.flodym_array.dims[x.name].items.index(y)),
        )

    def _check_data_complete(self):
        # Generate expected index tuples from FlodymArray dimensions
        if self.flodym_array.dims.ndim == 0:
            expected_index_tuples = set()
        else:
            expected_index_tuples = set(
                itertools.product(*(dim.items for dim in self.flodym_array.dims))
            )

        # Generate actual index tuples from DataFrame columns
        actual_index_tuples = set(
            self.df.drop(columns=self.format.value_column).itertuples(index=False, name=None)
        )

        # Compare the two sets
        if expected_index_tuples != actual_index_tuples:
            missing_items = expected_index_tuples - actual_index_tuples
            unexpected_items = actual_index_tuples - expected_index_tuples
            raise ValueError(
                f"Dataframe index mismatch! Missing items: {missing_items}, Unexpected items: {unexpected_items}"
            )
        if any(self.df[self.format.value_column].isna()):
            raise ValueError("Empty cells/NaN values in value column!")

    @staticmethod
    def same_items(arr: Iterable, dim: Dimension) -> bool:
        if dim.dtype is not None:
            try:
                arr = [dim.dtype(a) for a in arr]
            except ValueError:
                return False
        return len(set(arr).symmetric_difference(set(dim.items))) == 0
