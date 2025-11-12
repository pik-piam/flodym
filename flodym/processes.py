from typing import List, Dict, Optional
from pydantic import BaseModel as PydanticBaseModel, model_validator

from .flodym_arrays import Flow
from .stocks import Stock
from .mfa_definition import ProcessDefinition

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
            raise ValueError(f"In process {self.name}: No inflow from process '{from_process}' found.")

    def remove_outflow(self, to_process: str) -> None:
        """Remove an outflow from the process."""
        if to_process in self._outflows:
            del self._outflows[to_process]
        else:
            raise ValueError(f"In process {self.name}: No outflow to process '{to_process}' found.")


def make_processes(definitions: List[str|ProcessDefinition]) -> dict[str, Process]:
    """Create a dictionary of processes from a list of process names."""
    processes = {}
    for definition in definitions:
        if isinstance(definition, str):
            name = definition
            id = len(processes)
        elif isinstance(definition, ProcessDefinition):
            if definition.id != len(processes):
                raise ValueError(f"Processes must be defined with consecutive IDs starting from 0, but found {definition.id} in {len(processes)}'th definition.")
            name = definition.name
            id = definition.id
        processes[name] = Process(name=name, id=id)
    return processes
