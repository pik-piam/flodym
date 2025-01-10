from .processes import Process


def process_names_with_arrow(from_process: Process, to_process: Process) -> str:
    """Return the name of a flow as a string containing
    the process it starts from and the process it ends in, separated by an arrow.
    """
    return f"{from_process.name} => {to_process.name}"


def process_names_no_spaces(from_process: Process, to_process: Process) -> str:
    """Return the name of a flow as a string containing
    the process it starts from and the process it ends in, separated by '_to_', without spaces.
    """
    return f"{from_process.name.replace(' ', '_')}_to_{to_process.name.replace(' ', '_')}"


def process_ids(from_process: Process, to_process: Process) -> str:
    """Return the name of a flow as a string containing the id of the process it starts from and the id of the process it ends in, e.g. F1_2."""
    return f"F{from_process.id}_{to_process.id}"
