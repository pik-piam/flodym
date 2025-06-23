"""Home to a base class for MFA systems.

Specific MFA models can be built that inherit from this class.
"""

import logging
from typing import Dict, Optional

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, ConfigDict

from .mfa_definition import MFADefinition
from .dimensions import DimensionSet
from .flodym_arrays import Flow, Parameter, FlodymArray
from .stocks import Stock
from .processes import Process, make_processes
from .stock_helper import make_empty_stocks
from .flow_helper import make_empty_flows
from .data_reader import (
    DataReader,
    CompoundDataReader,
    CSVDimensionReader,
    CSVParameterReader,
    ExcelDimensionReader,
    ExcelParameterReader,
)


class MFASystem(PydanticBaseModel):
    """An MFASystem class handles the calculation of a Material Flow Analysis system, which
    consists of a set of processes, flows, stocks defined over a set of dimensions.
    For the concrete definition of the system, a subclass of MFASystem must be implemented.

    **Example**
    Define your MFA System:

        >>> from flodym import MFASystem
        >>> class CustomMFA(MFASystem):
        >>>     def compute(self):
        >>>         # do some computations on the CustomMFA attributes: stocks and flows

    MFA flows, stocks and parameters are defined as instances of subclasses of :py:class:`flodym.FlodymArray`.
    Dimensions are managed with the :py:class:`flodym.Dimension` and :py:class:`flodym.DimensionSet`.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    dims: DimensionSet
    """All dimensions that appear in the MFA system."""
    parameters: Dict[str, Parameter]
    """The parameters of the MFA system,
    as a dictionary mapping the names of the MFA system parameters to the parameters themselves.
    """
    processes: Dict[str, Process]
    """The processes of the MFA system, i.e. the nodes of the MFA system graph,
    as a dictionary mapping the names of the MFA system processes to the processes themselves.
    """
    flows: Dict[str, Flow]
    """The flows of the MFA system, i.e. the edges of the MFA system graph,
    as a dictionary mapping the names of the MFA system flows to the flows themselves.
    """
    stocks: Optional[Dict[str, Stock]] = {}
    """The stocks of the MFA system,
    as a dictionary mapping the names of the MFA system stocks to the stocks themselves.
    """

    @classmethod
    def from_data_reader(cls, definition: MFADefinition, data_reader: DataReader) -> "MFASystem":
        """Define and set up the MFA system and load all required data.
        Initialises stocks and flows with all zero values."""
        dims = data_reader.read_dimensions(definition.dimensions)
        parameters = data_reader.read_parameters(definition.parameters, dims=dims)
        processes = make_processes(definition.processes)
        flows = make_empty_flows(processes=processes, flow_definitions=definition.flows, dims=dims)
        stocks = make_empty_stocks(
            processes=processes, stock_definitions=definition.stocks, dims=dims
        )
        return cls(
            dims=dims,
            parameters=parameters,
            processes=processes,
            flows=flows,
            stocks=stocks,
        )

    @classmethod
    def from_csv(
        cls,
        definition: MFADefinition,
        dimension_files: dict,
        parameter_files: dict,
        allow_missing_parameter_values: bool = False,
        allow_extra_parameter_values: bool = False,
    ):
        """Define and set up the MFA system and load all required data from CSV files.
        Initialises stocks and flows with all zero values.

        See :py:class:`flodym.CSVDimensionReader` and
        :py:class:`flodym.CSVParameterReader`, and :py:meth:`flodym.FlodymArray.from_df` for expected
        format.

        :param definition: The MFA definition object
        :param dimension_files: A dictionary mapping dimension names to CSV files
        :param parameter_files: A dictionary mapping parameter names to CSV files
        :param allow_missing_parameter_values: Whether to allow missing values in the parameter data (missing rows or empty value cells)
        :param allow_extra_parameter_values: Whether to allow extra values in the parameter data
        """

        dimension_reader = CSVDimensionReader(
            dimension_files=dimension_files,
        )
        parameter_reader = CSVParameterReader(
            parameter_files=parameter_files,
            allow_missing_values=allow_missing_parameter_values,
            allow_extra_values=allow_extra_parameter_values,
        )
        data_reader = CompoundDataReader(
            dimension_reader=dimension_reader,
            parameter_reader=parameter_reader,
        )
        return cls.from_data_reader(definition, data_reader)

    @classmethod
    def from_excel(
        cls,
        definition: MFADefinition,
        dimension_files: dict,
        parameter_files: dict,
        dimension_sheets: dict = None,
        parameter_sheets: dict = None,
        allow_missing_parameter_values: bool = False,
        allow_extra_parameter_values: bool = False,
    ):
        """Define and set up the MFA system and load all required data from Excel files.
        Initialises stocks and flows with all zero values.
        Builds a CompoundDataReader from Excel readers, and calls the from_data_reader class method.

        See :py:class:`flodym.ExcelDimensionReader`,
        :py:class:`flodym.ExcelParameterReader`, and
        :py:meth:`flodym.FlodymArray.from_df` for expected format.

        :param definition: The MFA definition object
        :param dimension_files: A dictionary mapping dimension names to Excel files
        :param parameter_files: A dictionary mapping parameter names to Excel files
        :param dimension_sheets: A dictionary mapping dimension names to sheet names in the Excel files
        :param parameter_sheets: A dictionary mapping parameter names to sheet names in the Excel files
        :param allow_missing_parameter_values: Whether to allow missing values in the parameter data (missing rows or empty value cells)
        :param allow_extra_parameter_values: Whether to allow extra values in the parameter data
        """
        dimension_reader = ExcelDimensionReader(
            dimension_files=dimension_files,
            dimension_sheets=dimension_sheets,
        )
        parameter_reader = ExcelParameterReader(
            parameter_files=parameter_files,
            parameter_sheets=parameter_sheets,
            allow_missing_values=allow_missing_parameter_values,
            allow_extra_values=allow_extra_parameter_values,
        )
        data_reader = CompoundDataReader(
            dimension_reader=dimension_reader,
            parameter_reader=parameter_reader,
        )
        return cls.from_data_reader(definition, data_reader)

    def compute(self):
        """Perform all computations for the MFA system.
        This method must be implemented in a subclass of MFASystem.
        """
        raise NotImplementedError(
            "The compute method must be implemented in a subclass of MFASystem if it is to be used."
        )

    def get_new_array(self, dim_letters: tuple = None, **kwargs) -> FlodymArray:
        """get a new FlodymArray object.

        :param dim_letters: tuple of dimension letters to include in the new FlodymArray. If None, all dimensions are included.
        :param kwargs: keyword arguments to pass to the FlodymArray constructor.
        """
        dims = self.dims.get_subset(dim_letters)
        return FlodymArray(dims=dims, **kwargs)

    def _get_mass_balance(self) -> Dict[str, FlodymArray]:
        """Calculate the mass balance for each process, by summing the contributions.
        - all flows entering (positive)
        - all flows leaving (negative)
        - the stock change of the process (negative)

        Returns:
            A dictionary mapping process names to their mass balance contributions.
            Each contribution is a :py:class:`flodym.FlodymArray` with dimensions common to all contributions.
        """
        contributions = {p: [] for p in self.processes.keys()}

        # Add flows to mass balance
        for flow in self.flows.values():
            contributions[flow.from_process.name].append(-flow)  # Subtract flow from start process
            contributions[flow.to_process.name].append(flow)  # Add flow to end process

        # Add stock changes to the mass balance
        for stock in self.stocks.values():
            if stock.process is None:  # not connected to a process
                continue
            stock_change = stock.inflow - stock.outflow
            #    sum(flows_to) - sum(flows_from) = stock_change
            # => sum(flows_to) - sum(flows_from) - stock_change = 0
            # => stock_change is subtracted from the process
            contributions[stock.process.name].append(-stock_change)
            # system_mass_change = sum(stock_changes),
            # where the sysenv process mass balance is the negative system_mass_change:
            # system_mass_change = flows_into_system - flows_out_of_system
            #                    = sum(flows_from_sysenv) - sum(flows_to_sysenv)
            # So stock change is accounted to sysenv process with opposite sign as to other
            # processes => added instead of subtracted.
            contributions["sysenv"].append(stock_change)

        return {p_name: sum(parts) for p_name, parts in contributions.items()}

    @property
    def _absolute_float_precision(self) -> float:
        """The numpy float precision, multiplied by the maximum absolute flow or stock value."""
        max_flow_value = max([np.max(np.abs(f.values)) for f in self.flows.values()])
        max_stock_value = max([np.max(np.abs(s.stock.values)) for s in self.stocks.values()])
        epsilon = np.finfo(next(iter(self.flows.values())).values.dtype).eps
        return epsilon * max(max_flow_value, max_stock_value)

    def check_mass_balance(self, tolerance=None, raise_error: bool = True):
        """Compute mass balance, and check whether it is within a certain tolerance.
        Throw an error if it isn't.

        Args:
            tolerance (float, optional): The tolerance for the mass balance check.
                If None, defaults to 100 times the numpy float precision,
                multiplied by the maximum absolute flow or stock value.
            raise_error (bool): If True, raises an error if the mass balance check fails.
                Else, logs a warning and continues execution.
        """

        logging.info(f"Checking mass balance of {self.__class__.__name__} object...")

        if tolerance is None:
            tolerance = 100 * self._absolute_float_precision

        # returns array with dim [t, process, e]
        balances = self._get_mass_balance()
        max_errors = {p_name: np.max(np.abs(b.values)) for p_name, b in balances.items()}
        failed = {p_name: e for p_name, e in max_errors.items() if e > tolerance}
        if failed:
            info = ", ".join(f"{p_name} (max error: {e})" for p_name, e in failed.items())
            message = "Mass balance check failed for the following processes: " + info
            self._error_or_warning(message, raise_error)
        else:
            logging.info(f"Success - Mass balance is consistent!")

    def check_flows(
        self, exceptions: list[str] = [], raise_error: bool = False, verbose: bool = False
    ):
        """Check if all flows are non-negative.

        Args:
            exceptions (list[str]): A list of strings representing flow names to be excluded from the check.
            raise_error (bool): If True, raises an error instead of logging warnings.
            verbose (bool): If True, logs detailed information about negative flows.

        Raises:
            ValueError: If a negative flow is found and `raise_error` is True.

        Logs:
            Warning: If a negative flow is found and `raise_error` is False.
            Info: If no negative flows are found.
        """
        logging.info("Checking flows for NaN and negative values...")

        flows = [f for f in self.flows.values() if f.name not in exceptions]
        flows = [
            f
            for f in flows
            if f.from_process.name not in exceptions and f.to_process.name not in exceptions
        ]

        all_good = True
        # Check for NaN values
        for flow in flows:
            if np.any(np.isnan(flow.values)):
                message = f"NaN values found in flow {flow.name}!"
                self._error_or_warning(message, raise_error)
                all_good = False

        # Check for negative values
        tolerance = 100 * self._absolute_float_precision
        for flow in flows:
            if np.any(flow.values < -tolerance):
                message = f"Negative value in flow {flow.name}!"
                if verbose:
                    message += f"\n Items:"
                    indices = flow.items_where(lambda x: x < -tolerance)
                    for index in indices:
                        message += "\n  " + ", ".join(index)
                self._error_or_warning(message, raise_error)
                all_good = False

        if all_good:
            logging.info(f"Success - No negative flows or NaN values in {self.__class__.__name__}")

    @staticmethod
    def _error_or_warning(message: str, raise_error: bool) -> bool:
        if raise_error:
            raise ValueError(message)
        else:
            logging.warning(message)
