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


class UnderdeterminedError(Exception):
    """Exception raised when a process is underdetermined."""

    def __init__(self, process: "Process", message: str):
        message = (
            f"Cannot compute process '{process.name}' with ID {process.id}, as it is underdetermined. \n"
            + message
        )
        super().__init__(message)
        self.message = message


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
    dimension_splitter: Optional["FlodymArray"] = None
    stock: Optional[Stock] = None
    _inflow_shares: Dict[str, "FlodymArray"] = {}
    _outflow_shares: Dict[str, "FlodymArray"] = {}
    _total: FlodymArray = None
    _was_overdetermined: bool = None

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

    def add_inflow_share(self, from_process: str, share: FlodymArray):
        if from_process not in self._inflows:
            raise ValueError(
                f"In process {self.name}: Cannot add inflow share from process {from_process}, as no such inflow found."
            )
        self._inflow_shares[from_process] = share

    def add_outflow_share(self, to_process: str, share: FlodymArray):
        if to_process not in self._outflows:
            raise ValueError(
                f"In process {self.name}: Cannot add outflow share to process {to_process}, as no such outflow found."
            )
        self._outflow_shares[to_process] = share

    def remove_inflow_share(self, from_process: str):
        """Remove the inflow share for a given process."""
        if from_process in self._inflow_shares:
            del self._inflow_shares[from_process]
        else:
            raise ValueError(
                f"In process {self.name}: No inflow share for process '{from_process}' found."
            )

    def remove_outflow_share(self, to_process: str):
        """Remove the outflow share for a given process."""
        if to_process in self._outflow_shares:
            del self._outflow_shares[to_process]
        else:
            raise ValueError(
                f"In process {self.name}: No outflow share for process '{to_process}' found."
            )

    @property
    def is_computed(self) -> bool:
        """Check if the process has been computed."""
        if self.unknown_flows("in") or self.unknown_flows("out"):
            return False
        if self.stock is not None and not self.stock.is_computed:
            return False
        return True

    def compute(self, on_underdetermined: ErrorBehavior = "error", recursive: bool = False) -> None:
        """Compute all unknown flows of the process, based on topological information and known in/outflows.
        This covers most cases of MFA processes, but not all.
        Sometimes manual computations are still necessary.
        For example, if there is a 2x2 equation system to solve (see example 1).

        Topological information needed for this includes:
        - the inflows and outflows
        - the shares of some in- or outflows relative to the process total
        - an array to multiply with to apply a dimension split

        The computation assumes the following process structure:
        - one or several inflows
        - combined into a total
        - possible dimension change
        - one or several outflows
        Though quite general, this applies some restrictions on the possible process layouts.
        For example, no dimension expansion of individual in- or outflows through multiplication with shares is allowed.
        However, this can be realized by adding an own process which only applies this dimension splitting.

        The computation follows the following steps:
        - Detect if the total can be calculated from either the known inflows and inflow shares (forward mode),
          or from the known outflows and outflow shares (backward mode)
          Let's assume forward mode in the following, but backward mode works in the same way.
          The total can be calculated...
          - If all inflows are given, the total is simply their sum.
          - If a flow and its share are given, the total is that flow divided by its share
          - If the shares of all unknown flows are given, the total is t = sum(known_flows) / (1 - sum(shares_of_unknown_flows))
          If none of the three conditions is met on neither side (in/out), then the total cannot be calculated, and thus the whole
          process cannot be calculated.
        - Detect if there is a dimension_splitter and if it provides dimension information that the total does not have.
          If this is the case, apply dimension splitter by multiplying the total with it
        - Compute unknown flows. All but one unknown flows on each side need a share. Each of these unknown outflows with share
          is then calculated as total * share. If there is one remaining unknown flow without share, it is calculated as
          total - sum(known_flows_of_this_side).
          If more than one unknown flow on one side has no share defined, it cannot be calculated, and thus the whole process cannot be calculated.
        """
        if self.is_computed:
            logging.debug(f"Process {self.name} with ID {self.id} is already computed.")
            return
        if self.id == 0:
            logging.debug(
                f"Process {self.name} with ID {self.id} is the system boundary and cannot be computed."
            )
            return
        try:

            self._was_overdetermined = self.is_overdetermined
            self.try_compute()

            if not self.is_computed:
                raise ValueError(
                    f"In Process {self.name}: After computation, there are still unknown flows. "
                    "This indicates an internal flodym error."
                )

            if config.checks.mass_balance_processes:
                self.check_mass_balance()

            if recursive:
                for flow in self.inflows.values():
                    flow.from_process.compute(on_underdetermined=on_underdetermined, recursive=True)
                for flow in self.outflows.values():
                    flow.to_process.compute(on_underdetermined=on_underdetermined, recursive=True)

        except UnderdeterminedError as e:
            if on_underdetermined is None:
                on_underdetermined = config.error_behaviors.process_underdetermined
            if on_underdetermined == ErrorBehavior.INFO:
                # shorter message for readability in recursive mode (many will be underdertermined)
                e.message = f"Process {self.name} is underdetermined. Skip computation."
            handle_error(behavior=on_underdetermined, error=e)

    def try_compute(self):
        if self.stock is None:
            self.compute_no_stock()
        else:
            self.stock.compute_process()

    def compute_no_stock(self):
        if self.name == "fabrication":
            pass
        self.compute_total()
        self.apply_dimension_splitter()
        self.compute_flows()
        self.check_shares()

    def flows(self, direction: str):
        if direction == "in":
            return self._inflows
        elif direction == "out":
            return self._outflows
        else:
            raise ValueError("Direction must be 'in' or 'out'.")

    def shares(self, side: str):
        if side == "in":
            return self._inflow_shares
        elif side == "out":
            return self._outflow_shares
        else:
            raise ValueError("Direction must be 'in' or 'out'.")

    def neither_known(self, side: str) -> List[str]:
        return [
            name
            for name, flow in self.flows(side).items()
            if not flow.is_set and name not in self.shares(side)
        ]

    def known_flows(self, side: str) -> Dict[str, Flow]:
        return {name: flow for name, flow in self.flows(side).items() if flow.is_set}

    def unknown_flows(self, side: str) -> Dict[str, Flow]:
        return {name: flow for name, flow in self.flows(side).items() if not flow.is_set}

    def both_known(self, side: str) -> List[str]:
        return [
            name
            for name, flow in self.flows(side).items()
            if flow.is_set and name in self.shares(side)
        ]

    @property
    def is_overdetermined(self) -> bool:
        n_degree_of_freedom = len(self.flows("in")) + len(self.flows("out")) + 1  # +1 for total
        n_conditions = 2  # sum_inflows = total and sum_outflows = total
        n_given = (
            len(self.known_flows("in"))
            + len(self.known_flows("out"))
            + len(self.shares("in"))
            + len(self.shares("out"))
        )
        return n_given > n_degree_of_freedom - n_conditions

    def can_compute_total(self, from_side: str) -> bool:
        """Check if the process can compute the total flow through the process in the given direction."""
        return self.both_known(from_side) or (
            len(self.neither_known(from_side)) == 0 and len(self.known_flows(from_side)) >= 1
        )

    def can_compute_flows(self, sides: List[str] = ["in", "out"]) -> bool:
        """Check if the process can compute the flows in the given direction."""
        return all(len(self.neither_known(side)) <= 1 for side in sides)

    def compute_total(self, try_sides: List[str] = ["in", "out"]):
        """Compute the total flow based on the inflows or outflows."""
        side = None
        for s in try_sides:
            if self.can_compute_total(s):
                side = s
                break
        if side is None:
            message = self.get_underdetermined_error_message_total(try_sides)
            raise UnderdeterminedError(process=self, message=message)

        if len(self.known_flows(side)) == len(self.flows(side)):
            # all flows are known
            self._total = sum(self.flows(side).values())
        elif self.both_known(side):
            name = self.both_known(side)[0]
            excess_dims = self.shares(side)[name].dims - self.flows(side)[name].dims
            if excess_dims:
                names = ", ".join(excess_dims.names)
                raise ValueError(
                    f"In Process {self.name}: Share of flow to/from process {name} has dimensions "
                    f"{names} not contained in the flow's dimensions."
                )
            self._total = self.flows(side)[name] / self.shares(side)[name]
        else:
            sum_known = sum(self.known_flows(side).values())
            shares_unknown = {name: self.shares(side)[name] for name in self.unknown_flows(side)}
            dims_unknown = reduce(
                lambda x, y: x | y.dims, shares_unknown.values(), DimensionSet(dim_list=[])
            )
            if dims_unknown - sum_known.dims:
                share_names = ", ".join(
                    [name for name, share in shares_unknown.items() if share.dims - sum_known.dims]
                )
                excess_names = ", ".join((dims_unknown - sum_known.dims).names)
                raise ValueError(
                    f"In Process {self.name}: Error when trying to infer total flow from known "
                    f"flows and shares of unknown flows: Shares of flows to/from process(es) "
                    f"{share_names} has dimensions "
                    f"{excess_names} not contained in all the "
                    f"known flows from/to processes {', '.join(self.known_flows(side))}"
                )
            sum_shares_unknown = sum([s.cast_to(dims_unknown) for s in shares_unknown.values()])
            self._total = sum_known / (1 - sum_shares_unknown)

    def get_underdetermined_error_message_total(self, sides: List[str]) -> str:
        message = "Failed to compute total flow of process from given in/outflows.\n"
        for side in sides:
            message += f"Tried to compute from {side}flows, but failed. Reasons:\n"
            if not self.known_flows(side):
                message += f"- no flow from this side has an absolute value given\n"
            for linked_process in self.neither_known(side):
                message += f"- for flow from/to process {linked_process}, neither share nor absolute value is given.\n"
        return message

    def get_underdetermined_error_message_flows(self, sides: List[str]) -> str:
        message = ""
        for side in sides:
            message += f"Failed to compute {side}flows from total; For more than one {side}flow, neither share nor absolute value was given.\n"
            message += f"Please provide either share or absolute value for all but one of the flows from/to the processes "
            message += ", ".join(self.neither_known(side)) + ".\n"
        return message

    def apply_dimension_splitter(self, sides: list[str] = ["in", "out"]):
        """Apply the dimension splitter to the total flow."""

        # set union of all unknown flows
        dims_unknown = DimensionSet(dim_list=[])
        for side in sides:
            for flow in self.unknown_flows(side).values():
                dims_unknown |= flow.dims
        missing_dims = dims_unknown - self._total.dims

        if not missing_dims:
            if self.dimension_splitter is not None:
                self.handle_unused_splitter()
            return

        if self.dimension_splitter is None:
            raise ValueError(
                f"Process {self.name} has missing dimensions {missing_dims.names} for unknown flows, but no dimension splitter is set."
            )

        splitter_dims = self.dimension_splitter.dims
        if missing_dims - splitter_dims:
            raise ValueError(
                f"Dimension splitter of process {self.name} does not cover all dimensions {missing_dims.names} needed to determine unknown flows from total flow."
            )

        common_dims = splitter_dims & self._total.dims
        splitter_sum = self.dimension_splitter.sum_to(common_dims.letters)
        if np.max(np.abs((splitter_sum.values - 1))) > self.tolerance:
            raise ValueError(
                f"Dimension splitter of process {self.name} does not sum to 1 if summed to common dimensions with process total {common_dims.names}."
            )

        self._total *= self.dimension_splitter

    def handle_unused_splitter(self):
        message = f"In Process {self.name}: Dimension splitter is set, but not used. Given split may not be fulfilled."
        handle_error(behavior=config.error_behaviors.unused_dimension_splitter, message=message)

    def compute_flows(self, sides: List[str] = ["in", "out"]):
        """Compute the flows based on the total flow."""

        if not self.can_compute_flows(sides):
            message = self.get_underdetermined_error_message_flows(sides)
            raise UnderdeterminedError(process=self, message=message)

        for side in sides:
            # saving in temp flow avoids reducing dimensions.
            # This may be important for inferring the last flow by subtracting all known from the total,
            # because the dimension set intersection is used for that subtraction.
            temp_flows = {}
            for share_name, share in self.shares(side).items():
                if share_name in self.unknown_flows(side):
                    if share.dims - self._total.dims:
                        names = ", ".join((share.dims - self._total.dims).names)
                        raise ValueError(
                            f"In Process {self.name}: Share of flow to/from process {share_name} "
                            f"has dimensions {names} not contained in the processes' total "
                            f"dimensions {', '.join(self._total.dims.names)}. If you wish to "
                            "perform this dimensional split, consider outsourcing it to its own "
                            "process with a dimension_splitter."
                        )
                    temp_flows[share_name] = self._total * share
            # calculate the last by sum, if necessary
            if len(self.unknown_flows(side)) - len(temp_flows) == 1:
                unknown_name = [n for n in self.unknown_flows(side) if n not in temp_flows][0]
                unknown_flow = self.unknown_flows(side)[unknown_name]
                sum_known = sum(self.known_flows(side).values()) + sum(temp_flows.values())
                if isinstance(sum_known, FlodymArray) and unknown_flow.dims - sum_known.dims:
                    known_names = ", ".join(self.known_flows(side))
                    missing_names = ", ".join((unknown_flow.dims - sum_known.dims).names)
                    raise ValueError(
                        f"In Process {self.name}: One of the flows with set values ({known_names}) "
                        f"is missing dimensions ({missing_names}), which are needed to compute "
                        f"the unknown flow from/to process {unknown_name} by subtracting known "
                        "flows from the total."
                    )
                unknown_flow[...] = self._total - sum_known
            # reduce dimensions and transfer to actual flows
            for flow_name, flow in temp_flows.items():
                self.flows(side)[flow_name][...] = flow

    def check_shares(self, sides: List[str] = ["in", "out"]):
        if not config.checks.process_shares:
            return
        for side in sides:
            for name, share in self.shares(side).items():
                error = self._total * share - self.flows(side)[name]
                max_error = np.max(np.abs(error.values))
                if max_error > self.tolerance:
                    message = (
                        f"after computation of process {self.name}: Share of flow to/from process "
                        f"{name} is not fulfilled (error = {max_error})."
                    )
                    if self._was_overdetermined:
                        message += (
                            "This may be due to the process being overdetermined."
                            "If it is not intentional, consider removing some shares or given flows."
                        )
                    else:
                        message += "This may be due to lacking float precision, or an internal flodym error."
                    handle_error(behavior=config.error_behaviors.process_shares, message=message)

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
