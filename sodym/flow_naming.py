from .processes import Process


def process_names_with_arrow(from_process: Process, to_process: Process) -> str:
    return f"{from_process.name} => {to_process.name}"


def process_names_no_spaces(from_process: Process, to_process: Process) -> str:
    return f"{from_process.name}_to_{to_process.name}"


def process_ids(from_process: Process, to_process: Process) -> str:
    return f"F{from_process.id}_{to_process.id}"
