"""Predefined functions to generate flow names from the processes a flow connects.

One of them, or a custom function with the same signature, can be set as
:py:attr:`flodym.Config.flow_naming`.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .processes import Process


def process_names_with_arrow(from_process: "Process", to_process: "Process") -> str:
    """Return a flow name containing the names of both processes, separated by an arrow.

    The name contains the process the flow starts from and the process it ends in.
    """
    return f"{from_process.name} => {to_process.name}"


def process_names_no_spaces(from_process: "Process", to_process: "Process") -> str:
    """Return a flow name containing the names of both processes, separated by '_to_'.

    The name contains the process the flow starts from and the process it ends in, with spaces
    replaced by underscores.
    """
    return f"{from_process.name.replace(' ', '_')}_to_{to_process.name.replace(' ', '_')}"


def process_ids(from_process: "Process", to_process: "Process") -> str:
    """Return a flow name containing the IDs of both processes, e.g. F1_2.

    The name contains the ID of the process the flow starts from and the ID of the process it ends
    in.
    """
    return f"F{from_process.id}_{to_process.id}"
