from typing import List

from pydantic import BaseModel as PydanticBaseModel, model_validator


class Process(PydanticBaseModel):
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

    @model_validator(mode="after")
    def check_id0(self):
        """Ensure that the process with ID 0 is named 'sysenv'."""
        if self.id == 0 and self.name != "sysenv":
            raise ValueError(
                "The process with ID 0 must be named 'sysenv', as it contains everything outside the system boundary."
            )
        return self


def make_processes(definitions: List[str]) -> dict[str, Process]:
    """Create a dictionary of processes from a list of process names."""
    return {name: Process(name=name, id=id) for id, name in enumerate(definitions)}
