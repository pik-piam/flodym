from typing import List, Dict, Optional
from pydantic import BaseModel as PydanticBaseModel, model_validator
import logging
from functools import reduce
import numpy as np

from .flodym_arrays import Flow, FlodymArray, Parameter
from .stocks import Stock
from .mfa_definition import ProcessDefinition
from .dimensions import DimensionSet
from .config import config, handle_error, ErrorBehavior


class Process(PydanticBaseModel, arbitrary_types_allowed=True):
    """Processes serve as nodes for the MFA system layout definition.
    Flows are defined between two processes. Stocks are connected to a process.
    Processes do not contain values themselves.

    Processes get an ID by the order they are defined in the :py:attribute::`MFASystem.definition`.
    The process with ID 0 necessarily contains everything outside the system boundary.
    It has to be named 'sysenv'.
    """

    name: str
    """Name of the process."""
    id: int
    """ID of the process."""
    _inflows: Dict[str, Flow] = {}
    """Inflows to the process, keyed by linked process name."""
    _outflows: Dict[str, Flow] = {}
    """Outflows from the process, keyed by linked process name."""
    stock: Optional[Stock] = None

    @model_validator(mode="after")
    def check_id0(self):
        """Ensure that the process with ID 0 is named 'sysenv'."""
        if self.id == 0 and self.name != "sysenv":
            raise ValueError(
                "The process with ID 0 must be named 'sysenv', as it contains everything outside the system boundary."
            )
        return self

    def __repr__(self):
        return ""  # TODO

    @property
    def inflows(self) -> Dict[str, Flow]:
        """Inflows to the process."""
        return self._inflows

    @property
    def outflows(self) -> Dict[str, Flow]:
        """Outflows from the process."""
        return self._outflows

    def add_inflow(self, flow: Flow) -> None:
        """Add an inflow flow to the process."""
        self._inflows[flow.from_process.name] = flow

    def add_outflow(self, flow: Flow) -> None:
        """Add an outflow flow from the process."""
        self._outflows[flow.to_process.name] = flow

    def remove_inflow(self, from_process: str) -> None:
        """Remove an inflow from the process."""
        if from_process in self._inflows:
            del self._inflows[from_process]
        else:
            raise ValueError(
                f"In process {self.name}: No inflow from process '{from_process}' found."
            )

    def remove_outflow(self, to_process: str) -> None:
        """Remove an outflow from the process."""
        if to_process in self._outflows:
            del self._outflows[to_process]
        else:
            raise ValueError(f"In process {self.name}: No outflow to process '{to_process}' found.")

    @property
    def is_computed(self) -> bool:
        """Check if the process has been computed."""
        if self.unknown_flows("in") or self.unknown_flows("out"):
            return False
        if self.stock is not None and not self.stock.is_computed:
            return False
        return True

    def flows(self, direction: str):
        if direction == "in":
            return self._inflows
        elif direction == "out":
            return self._outflows
        else:
            raise ValueError("Direction must be 'in' or 'out'.")

    def known_flows(self, side: str) -> Dict[str, Flow]:
        return {name: flow for name, flow in self.flows(side).items() if flow.is_set}

    def unknown_flows(self, side: str) -> Dict[str, Flow]:
        return {name: flow for name, flow in self.flows(side).items() if not flow.is_set}

    def get_sum_inflows(self) -> FlodymArray:
        """Sum of all inflows to the process."""
        if not self.is_computed:
            raise ValueError(
                f"Cannot compute sum_inflows for process {self.name} before it is computed."
            )
        if not self.inflows:
            return 0.0
        return sum(self.inflows.values())

    def get_sum_outflows(self) -> FlodymArray:
        """Sum of all outflows from the process."""
        if not self.is_computed:
            raise ValueError(
                f"Cannot compute sum_outflows for process {self.name} before it is computed."
            )
        if not self.outflows:
            return 0.0
        return sum(self.outflows.values())

    def get_net_inflow(self):
        return self.get_sum_inflows() - self.get_sum_outflows()

    @property
    def _absolute_float_precision(self) -> float:
        """The numpy float precision, multiplied by the maximum absolute flow or stock value."""
        if self.inflows:
            max_inflow = max([f._absolute_float_precision for f in self.inflows.values()])
        else:
            max_inflow = 0.0
        if self.outflows:
            max_outflow = max([f._absolute_float_precision for f in self.outflows.values()])
        else:
            max_outflow = 0.0
        return max(max_inflow, max_outflow)

    @property
    def tolerance(self) -> float:
        """The tolerance for checks like mass balance"""
        if config.absolute_tolerance is not None:
            return config.absolute_tolerance
        return config.relative_tolerance * self._absolute_float_precision

    def check_mass_balance(
        self,
        tolerance: float = None,
        error_behavior: ErrorBehavior = None,
        mass_change_target: FlodymArray = None,
    ):
        """Compute mass balance, and check whether it is within a certain tolerance.
        Throw an error if it isn't.

        Args:
            tolerance (float, optional): A tolerance override for the mass balance check.
                If None, takes the  processes's own tolerance
            error_behavior (ErrorBehavior): What to do if the mass balance check fails.
                If None, takes the global config setting, which defaults to raising an error
        """

        if self.id == 0 and mass_change_target is None:
            logging.info(
                f"mass_change_target not given. Skip mass balance check for system environment."
            )
            return

        logging.info(f"Checking mass balance of {self.__class__.__name__} object...")

        if mass_change_target is None:
            mass_change_target = 0.0 if self.stock is None else self.stock.net_inflow
        # net_inflow = mass_change, so the difference should be zero
        mass_balance = self.get_net_inflow() - mass_change_target
        max_error = np.max(np.abs(mass_balance.values))
        if tolerance is None:
            tolerance = self.tolerance
        if max_error > tolerance:
            message = f"In process {self.name}: Mass balance check failed (error = {max_error})"
            if error_behavior is None:
                error_behavior = config.error_behaviors.mass_balance
            handle_error(behavior=error_behavior, message=message)
        else:
            logging.info(f"In process {self.name}: Success - Mass balance is consistent!")


def make_processes(definitions: List[str | ProcessDefinition]) -> dict[str, Process]:
    """Create a dictionary of processes from a list of process names."""
    processes = {}
    for definition in definitions:
        if isinstance(definition, str):
            name = definition
            id = len(processes)
        elif isinstance(definition, ProcessDefinition):
            if definition.id != len(processes):
                raise ValueError(
                    f"Processes must be defined with consecutive IDs starting from 0, but found {definition.id} in {len(processes)}'th definition."
                )
            name = definition.name
            id = definition.id
        processes[name] = Process(name=name, id=id)
    return processes


def set_process_parameters(
    processes: Dict[str, Process],
    definitions: List[ProcessDefinition],
    parameters: Dict[str, Parameter],
):
    for definition in definitions:
        if isinstance(definition, str):
            continue

        if definition.name not in processes:
            raise ValueError(
                f"Process {definition.name} is not defined in the processes dictionary."
            )
        process = processes[definition.name]

        if definition.inflow_shares:
            for from_process, parameter_name in definition.inflow_shares.items():
                if parameter_name not in parameters:
                    raise ValueError(
                        f"Parameter {parameter_name} given in definition of process {definition.name} is not defined."
                    )
                process.add_inflow_share(from_process, parameters[parameter_name])

        if definition.outflow_shares:
            for to_process, parameter_name in definition.outflow_shares.items():
                if parameter_name not in parameters:
                    raise ValueError(
                        f"Parameter {parameter_name} given in definition of process {definition.name} is not defined."
                    )
                process.add_outflow_share(to_process, parameters[parameter_name])

        if definition.dimension_splitter:
            if definition.dimension_splitter not in parameters:
                raise ValueError(
                    f"Dimension splitter {definition.dimension_splitter} given in definition of process {definition.name} is not defined."
                )
            process.dimension_splitter = parameters[definition.dimension_splitter]
