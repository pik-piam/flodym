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

    def _get_mass_contributions(self):
        """List all contributions to the mass balance of each process:
        - all flows entering are positive
        - all flows leaving are negative
        - the stock change of the process
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
            # add/subtract stock changes to processes
            contributions[stock.process.name].append(-stock.inflow)
            contributions[stock.process.name].append(stock.outflow)
            # add/subtract stock changes in system boundary for mass balance of whole system
            contributions["sysenv"].append(stock.inflow)
            contributions["sysenv"].append(-stock.outflow)

        return contributions

    def _get_mass_balance(self, contributions: dict = {}):
        """Calculate the mass balance for each process, by summing the contributions.
        The sum returns a :py:class:`flodym.FlodymArray`,
        with the dimensions common to all contributions.
        """
        if not contributions:
            contributions = self._get_mass_contributions()
        return {p_name: sum(parts) for p_name, parts in contributions.items()}

    def _get_mass_totals(self, contributions: dict = {}):
        """Calculate the total mass of a process by summing the absolute values of all
        the contributions.
        """
        if not contributions:
            contributions = self._get_mass_contributions()
        return {
            p_name: sum([abs(part) for part in parts]) for p_name, parts in contributions.items()
        }

    def _get_relative_mass_balance(self, epsilon=1e-9):
        """Determines a relative mass balance for each process of the MFA system,
        by dividing the mass balances by the mass totals.
        """
        mass_contributions = self._get_mass_contributions()
        balances = self._get_mass_balance(contributions=mass_contributions)
        totals = self._get_mass_totals(contributions=mass_contributions)

        relative_balance = {
            p_name: (balances[p_name] / (totals[p_name] + epsilon)).values
            for p_name in self.processes
        }
        return relative_balance

    def check_mass_balance(self, tolerance=1e-4):
        """Compute mass balance, and check whether it is within a certain tolerance.
        Throw an error if it isn't."""

        # returns array with dim [t, process, e]
        relative_balance = self._get_relative_mass_balance()  # assume no error if total sum is 0
        id_failed = {p_name: np.any(rb > tolerance) for p_name, rb in relative_balance.items()}
        messages_failed = [
            f"{p_name} ({np.max(relative_balance[p_name])*100:.2f}% error)"
            for p_name in self.processes.keys()
            if id_failed[p_name]
        ]
        if any(id_failed.values()):
            raise RuntimeError(
                f"Error, Mass Balance fails for processes {', '.join(messages_failed)}"
            )
        else:
            logging.info(
                f"Success - Mass balance of {self.__class__.__name__} object is consistent!"
            )
        return
