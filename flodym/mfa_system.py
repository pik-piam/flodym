"""Home to a base class for MFA systems.

Specific MFA models can be built that inherit from this class.
"""

import logging
import warnings
from os import PathLike
from typing import Dict, Optional, SupportsFloat, Union

import numpy as np
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, model_validator
from typing_extensions import Self

from .data_reader import (
    CompoundDataReader,
    CSVDimensionReader,
    CSVParameterReader,
    DataReader,
    ExcelDimensionReader,
    ExcelParameterReader,
)
from .dimensions import DimensionSet
from .flodym_arrays import FlodymArray, Flow, Parameter
from .flow_helper import make_empty_flows
from .mfa_definition import MFADefinition
from .processes import Process, compute_processes, make_processes, set_process_parameters
from .stock_helper import make_empty_stocks
from .stocks import Stock
from ._checks import config_hint, report_balance_failures
from .config import ErrorBehavior, config, handle_error


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
    stocks: Dict[str, Stock] = {}
    """The stocks of the MFA system,
    as a dictionary mapping the names of the MFA system stocks to the stocks themselves.
    """
    flow_parameters: Dict[str, str] = {}
    """Names of parameters whose values flows are set to, keyed by flow name,
    see :py:meth:`set_flows_from_parameters`. Taken from
    :py:attr:`flodym.FlowDefinition.parameter` when set up from a definition."""

    @model_validator(mode="after")
    def link_processes(self):
        """Link the processes to the flows and stocks at setup, see :py:meth:`relink`."""
        self.relink()
        return self

    def relink(self) -> None:
        """Link each process to its inflows, outflows and stock, as given by the flows and stocks
        of the MFA system (see :py:attr:`flodym.Process.inflows`).

        This is done automatically when the MFA system is set up. Call it again after adding,
        replacing or removing entries of :py:attr:`flows` or :py:attr:`stocks`.
        Existing links of the processes are replaced.

        Raises:
            ValueError: If a flow or stock refers to a process object which is not in
                :py:attr:`processes`, or if several stocks refer to the same process.
                In this case, the existing links are left unchanged.
        """
        linked_stocks = {}
        for flow in self.flows.values():
            for process in (flow.from_process, flow.to_process):
                self._check_is_own_process(process, f"flow '{flow.name}'")
        for stock in self.stocks.values():
            if stock.process is None:
                continue
            self._check_is_own_process(stock.process, f"stock '{stock.name}'")
            if stock.process.name in linked_stocks:
                raise ValueError(
                    f"Process '{stock.process.name}' cannot have both stocks "
                    f"'{linked_stocks[stock.process.name]}' and '{stock.name}'."
                )
            linked_stocks[stock.process.name] = stock.name

        for process in self.processes.values():
            process.clear_links()
        for flow in self.flows.values():
            flow.from_process.add_outflow(flow)
            flow.to_process.add_inflow(flow)
        for stock in self.stocks.values():
            if stock.process is not None:
                stock.process.set_stock(stock)

    def _check_is_own_process(self, process: Process, used_by: str) -> None:
        if self.processes.get(process.name) is not process:
            raise ValueError(
                f"Process '{process.name}' of {used_by} is not a process of the MFA system."
            )

    @classmethod
    def from_data_reader(cls, definition: MFADefinition, data_reader: DataReader) -> Self:
        """Define and set up the MFA system and load all required data.
        Initialises stocks and flows with all zero values, except for flows with a parameter
        (see :py:attr:`flodym.FlowDefinition.parameter`), which are set to its values.
        Shares and dimension splitters of process definitions are set to the processes.
        """
        dims = data_reader.read_dimensions(definition.dimensions)
        parameters = data_reader.read_parameters(definition.parameters, dims=dims)
        processes = make_processes(definition.processes)
        flows = make_empty_flows(processes=processes, flow_definitions=definition.flows, dims=dims)
        stocks = make_empty_stocks(
            processes=processes, stock_definitions=definition.stocks, dims=dims
        )
        if len(flows) != len(definition.flows):
            raise ValueError("Flow definitions result in duplicate flow names.")
        flow_parameters = {}
        for flow_definition, flow_name in zip(definition.flows, flows):
            if flow_definition.parameter is not None:
                flow_parameters[flow_name] = flow_definition.parameter
        mfa = cls(
            dims=dims,
            parameters=parameters,
            processes=processes,
            flows=flows,
            stocks=stocks,
            flow_parameters=flow_parameters,
        )
        set_process_parameters(mfa.processes, definition.processes, mfa.parameters)
        mfa.set_flows_from_parameters()
        return mfa

    @classmethod
    def from_csv(
        cls,
        definition: MFADefinition,
        dimension_files: dict[str, str | PathLike[str]],
        parameter_files: dict[str, str | PathLike[str]],
        allow_missing_parameter_values: bool = False,
        allow_extra_parameter_values: bool = False,
    ) -> Self:
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
        dimension_files: dict[str, str | PathLike[str]],
        parameter_files: dict[str, str | PathLike[str]],
        dimension_sheets: Optional[dict[str, str]] = None,
        parameter_sheets: Optional[dict[str, str]] = None,
        allow_missing_parameter_values: bool = False,
        allow_extra_parameter_values: bool = False,
    ) -> Self:
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

    def get_new_array(self, dim_letters: Optional[tuple[str, ...]] = None, **kwargs) -> FlodymArray:
        """Get a new FlodymArray object.

        :param dim_letters: tuple of dimension letters to include in the new FlodymArray. If None, all dimensions are included.
        :param kwargs: keyword arguments to pass to the FlodymArray constructor.
        """
        dims = self.dims.get_subset(dim_letters)
        return FlodymArray(dims=dims, **kwargs)

    @property
    def total_stock_change(self) -> Union[FlodymArray, float]:
        """Sum of the net inflows (per year) of all stocks connected to a process, i.e. the mass
        change of the whole system, reduced to the dimensions common to all stocks."""
        return sum((s.net_inflow for s in self.stocks.values() if s.process is not None), 0.0)

    @property
    def _absolute_float_precision(self) -> float:
        """Machine epsilon times the largest absolute value of all flows and stocks."""
        precisions = [f._absolute_float_precision for f in self.flows.values()]
        precisions += [s._absolute_float_precision for s in self.stocks.values()]
        return max(precisions, default=0.0)

    @property
    def tolerance(self) -> float:
        """Absolute tolerance based on all flows and stocks of the MFA system, see
        :py:attr:`flodym.config.Config.relative_tolerance`. Used by :py:meth:`check_flows`."""
        return config.get_tolerance(self._absolute_float_precision)

    def check_mass_balance(
        self,
        tolerance: Optional[SupportsFloat] = None,
        error_behavior: Optional[ErrorBehavior | str] = None,
        raise_error: Optional[bool] = None,
    ) -> bool:
        """Check the mass balance of all processes and stocks of the MFA system.

        - For each process, the sum of inflows minus the sum of outflows must equal the net inflow
          of its stock, or zero if it has no stock (see :py:meth:`flodym.Process.check_mass_balance`).
          For the 'sysenv' process, it must equal the negative mass change of the system
          (see :py:attr:`total_stock_change`).
        - For each stock, inflow, outflow and stock must be consistent
          (see :py:meth:`flodym.Stock.check_mass_balance`).

        Args:
            tolerance: Absolute tolerance for all checks. If None, each process and stock is
                checked with its own tolerance (see :py:attr:`flodym.Process.tolerance` and
                :py:attr:`flodym.Stock.tolerance`), so that small processes and stocks are checked
                as strictly as large ones. The tolerance of 'sysenv' includes the in- and outflows
                of all stocks.
            error_behavior: What to do if the check fails. If None,
                ``flodym.config.error_behaviors.mass_balance`` is used (default: raise an error).
            raise_error: Deprecated, use ``error_behavior`` instead. True corresponds to
                ``"error"``, False to ``"warn"``.

        Returns:
            Whether all checks passed.
        """
        logging.info(f"Checking mass balance of {self.__class__.__name__} object...")

        if tolerance is not None:
            tolerance = float(tolerance)
        error_behavior = _resolve_error_behavior(
            error_behavior, raise_error, config.error_behaviors.mass_balance, "check_mass_balance"
        )

        failures = []
        for process in self.processes.values():
            if process.id == 0:
                failures.append(
                    process._mass_balance_failure(
                        tolerance=self._sysenv_tolerance(process)
                        if tolerance is None
                        else tolerance,
                        mass_change=-self.total_stock_change,
                    )
                )
            else:
                failures.append(process._mass_balance_failure(tolerance=tolerance))
        failures += [
            stock._mass_balance_failure(tolerance=tolerance) for stock in self.stocks.values()
        ]

        return report_balance_failures(
            failures,
            error_behavior,
            hint=config_hint("mass_balance", tolerance=True),
            success_message="Success - Mass balance is consistent!",
        )

    def _sysenv_tolerance(self, sysenv: Process) -> float:
        """Tolerance for the mass balance of the 'sysenv' process, which includes the in- and
        outflows of all stocks connected to a process."""
        precisions = [sysenv._absolute_float_precision]
        for stock in self.stocks.values():
            if stock.process is not None:
                precisions += [
                    stock.inflow._absolute_float_precision,
                    stock.outflow._absolute_float_precision,
                ]
        return config.get_tolerance(max(precisions))

    def check_flows(
        self,
        exceptions: list[str] = [],
        error_behavior: Optional[ErrorBehavior | str] = None,
        verbose: bool = False,
        raise_error: Optional[bool] = None,
    ) -> bool:
        """Check that no flow has NaN values or negative values (beyond the tolerance of the
        MFA system, see :py:attr:`tolerance`).

        Args:
            exceptions (list[str]): Names of flows to be excluded from the check. Flows from or to
                processes with names in this list are excluded as well.
            error_behavior: What to do for each flow that fails the check. If None,
                ``flodym.config.error_behaviors.check_flows`` is used (default: log a warning).
            verbose (bool): If True, the message lists the items with negative values.
            raise_error: Deprecated, use ``error_behavior`` instead. True corresponds to
                ``"error"``, False to ``"warn"``.

        Returns:
            Whether all checked flows passed.
        """
        logging.info("Checking flows for NaN and negative values...")
        error_behavior = _resolve_error_behavior(
            error_behavior, raise_error, config.error_behaviors.check_flows, "check_flows"
        )

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
                message = f"NaN values found in flow {flow.name}!" + config_hint("check_flows")
                handle_error(behavior=error_behavior, message=message)
                all_good = False

        # Check for negative values
        tolerance = self.tolerance
        for flow in flows:
            if np.any(flow.values < -tolerance):
                message = f"Negative value in flow {flow.name}!" + config_hint(
                    "check_flows", tolerance=True
                )
                if verbose:
                    message += "\n Items:"
                    indices = flow.items_where(lambda x: x < -tolerance)
                    for index in indices:
                        message += "\n  " + ", ".join(index)
                handle_error(behavior=error_behavior, message=message)
                all_good = False

        if all_good:
            logging.info(f"Success - No negative flows or NaN values in {self.__class__.__name__}")
        return all_good

    def mark_all_set(self):
        """Mark all stocks and flows as set."""
        for flow in self.flows.values():
            flow.mark_set()
        for stock in self.stocks.values():
            stock.mark_all_set()

    def mark_all_unset(self):
        """Mark all stocks and flows as unset."""
        for flow in self.flows.values():
            flow.mark_unset()
        for stock in self.stocks.values():
            stock.mark_all_unset()

    def set_flows_from_parameters(self, overwrite: bool = False):
        """Set the flows in :py:attr:`flow_parameters` to the values of their parameters.

        Args:
            overwrite: If False, only flows which are not set (see
                :py:attr:`flodym.FlodymArray.is_set`) are set.
        """
        for flow_name, parameter_name in self.flow_parameters.items():
            if flow_name not in self.flows:
                raise KeyError(
                    f"Flow '{flow_name}' of flow_parameters is not a flow of the system."
                )
            if parameter_name not in self.parameters:
                raise KeyError(
                    f"Parameter '{parameter_name}' given for flow '{flow_name}' is not a parameter "
                    "of the system."
                )
            flow = self.flows[flow_name]
            if overwrite or not flow.is_set:
                flow[...] = self.parameters[parameter_name]

    def compute_all_possible(self) -> list[str]:
        """Compute all processes which can be computed from the known flows, shares and stock
        values, see :py:meth:`flodym.Process.compute`.

        Unset flows with a parameter (see :py:attr:`flow_parameters`) are set first. Then, all
        processes are tried, and each computed process triggers a new try of its neighbors, until
        no further process can be computed (see :py:func:`flodym.compute_processes`).
        Underdetermined processes are skipped.

        Returns:
            The names of the processes (except 'sysenv') which could not be computed.
        """
        self.set_flows_from_parameters()
        compute_processes(self.processes.values())
        not_computed = [p.name for p in self.processes.values() if p.id != 0 and not p.is_computed]
        if not_computed:
            logging.info(
                f"compute_all_possible(): Could not compute processes {', '.join(not_computed)}."
            )
        return not_computed


def _resolve_error_behavior(
    error_behavior: Optional[ErrorBehavior | str | bool],
    raise_error: Optional[bool],
    default: ErrorBehavior,
    method_name: str,
) -> ErrorBehavior:
    """Get the error behavior of a check, supporting the deprecated ``raise_error`` argument.

    A bool given as ``error_behavior`` stems from the deprecated positional use of
    ``raise_error`` and is treated as such.
    """
    if isinstance(error_behavior, bool):
        raise_error, error_behavior = error_behavior, None
    if raise_error is not None:
        warnings.warn(
            f"The raise_error argument of MFASystem.{method_name}() is deprecated since v1.2.0. "
            "It will be removed in v2.0.0. Use error_behavior='error' or 'warn' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        if error_behavior is not None:
            raise ValueError("Only one of error_behavior and raise_error may be given.")
        return ErrorBehavior.ERROR if raise_error else ErrorBehavior.WARN
    if error_behavior is None:
        return default
    return ErrorBehavior(error_behavior)
