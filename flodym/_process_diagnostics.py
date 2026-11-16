"""Explanations why processes of a block are not computed simultaneously.

See :py:func:`flodym.compute_processes`. :py:func:`explain_block` builds the explanation from a
summary of the solution of the equations of a block (``flodym._linear_system``), which keeps
only small arrays, so that the explanation can be built when it is reported.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ._linear_system import SolutionSummary
from ._process_compute import processes_on_cycles, quoted, unique_flows

if TYPE_CHECKING:
    from ._process_equations import ProcessEquations
    from .flodym_arrays import Flow
    from .processes import Process


@dataclass
class Explanation:
    """Why processes of a block are not computed."""

    message: str
    """Explanation for the report, ending with a newline."""
    undetermined_flows: list[str] = field(default_factory=list)
    """Names of unknown flows which are not determined by the equations."""
    data_error: bool = False
    """Whether the given values contradict each other, e.g. in a loop without exit, or are not
    finite. Such explanations are reported as a warning by default."""


@dataclass
class SingularPart:
    """Unknowns whose equations have no unique solution, although their structure suggests one.

    Parts of the same unknown arrays, e.g. for different years, are combined.
    """

    flows: list["Flow"]
    """Unknown flows of the part."""
    items: list[str]
    """Examples of the items concerned, e.g. ``"Time 2001"``."""
    n_parts: int
    """Number of parts combined."""
    loop_processes: list["Process"] = field(init=False)
    """Processes (other than 'sysenv') on a directed cycle of the flows of the part. If there
    are any, the part is a loop without exit."""

    def __post_init__(self) -> None:
        on_cycle = processes_on_cycles(self.flows)
        processes = {f.from_process.id: f.from_process for f in self.flows}
        processes |= {f.to_process.id: f.to_process for f in self.flows}
        self.loop_processes = [processes[i] for i in sorted(processes) if i in on_cycle]

    def message(self) -> str:
        """Explanation for the report."""
        flow_names = quoted(flow.name for flow in self.flows)
        more = len(self.items) < self.n_parts
        items = "; ".join(self.items) + (" and others" if more else "")
        if self.loop_processes:
            return (
                f"- Processes {quoted(process.name for process in self.loop_processes)} form a "
                f"loop without exit at {items}: the shares along the loop multiply to one, so "
                "that nothing leaves it and its equations have no unique solution. No item of "
                "these processes is computed. Lower a share or add an exit flow. Flows "
                f"concerned: {flow_names}."
            )
        return (
            f"- The given values only fix the sum of the flows {flow_names} at {items}. Give a "
            "share or value for all but one of them."
        )


def explain_block(
    processes: Sequence["Process"],
    equations: dict[int, "ProcessEquations"],
    left_out: dict[int, str],
    summary: SolutionSummary,
) -> Explanation:
    """Why processes of a block are not computed, from the summary of its solution.

    Args:
        processes: The processes of the block, sorted by ID.
        equations: The equations of the processes which took part, by process ID.
        left_out: Why the other processes did not take part, by process ID.
        summary: Summary of the solution of the equations.
    """
    not_computed = [process for process in processes if not process.is_computed]
    flows = flows_of(processes)
    undetermined = [
        name for name, flow in flows.items() if not flow.is_set and not summary.is_determined(name)
    ]
    lines = not_finite_lines(summary, equations, not_computed)
    data_error = bool(lines)
    explained: set[str] = set()
    for part in singular_parts(summary, flows):
        lines.append(part.message())
        data_error |= bool(part.loop_processes)
        explained.update(flow.name for flow in part.flows)
    unexplained = [name for name in undetermined if name not in explained]
    if unexplained:
        lines.append(
            f"- The flows {quoted(unexplained)} are not determined by the given flows, shares "
            "and stock values."
        )
    for process in not_computed:
        if process.id in left_out:
            lines.append(f"- Process '{process.name}' is left out: {left_out[process.id]}")
            continue
        hints = equations[process.id].failure_hints()
        if hints:
            lines.append(f"- Process '{process.name}': {' '.join(hints)}")
    message = header(processes, not_computed) + "".join(f"{line}\n" for line in lines)
    return Explanation(message, undetermined, data_error)


def header(processes: Sequence["Process"], not_computed: Sequence["Process"]) -> str:
    """First line of the explanation why processes of a block are not computed."""
    names = quoted(process.name for process in processes)
    if len(processes) == 1:
        return f"Process {names} is not determined:\n"
    unset_flows = [flow for flow in flows_of(processes).values() if not flow.is_set]
    relation_text = relation(processes, unset_flows, not_computed)
    if len(not_computed) == len(processes):
        return f"{relation_text}, but are not determined:\n"
    verb = "is" if len(not_computed) == 1 else "are"
    return (
        f"{relation_text} and were solved together, but "
        f"{quoted(process.name for process in not_computed)} {verb} not determined:\n"
    )


def singular_parts(summary: SolutionSummary, flows: dict[str, "Flow"]) -> list[SingularPart]:
    """The singular parts of a solution, combined if they concern the same unknown arrays."""
    unknowns = summary.singular_unknowns
    if not len(unknowns):
        return []
    blocks = summary.unknown_blocks(unknowns)
    offsets = np.array([block.offset for block in blocks])
    block_of_offset = {block.offset: block for block in blocks}
    # The pairs of part and block (given by its offset) which occur, sorted by part, with the
    # position of the first unknown of each pair. Each part is a run of pairs.
    pairs, first_unknowns = np.unique(
        np.stack([summary.singular_parts, offsets]), axis=1, return_index=True
    )
    part_starts = np.flatnonzero(np.diff(pairs[0])) + 1
    combined: dict[tuple, SingularPart] = {}
    for part_offsets, part_first_unknowns in zip(
        np.split(pairs[1], part_starts), np.split(first_unknowns, part_starts)
    ):
        keys = tuple(block_of_offset[offset].key for offset in part_offsets)
        if keys not in combined:
            part_flows = [flow for name, flow in flows.items() if name in keys]
            combined[keys] = SingularPart(flows=part_flows, items=[], n_parts=0)
        singular_part = combined[keys]
        singular_part.n_parts += 1
        if len(singular_part.items) < 3:
            position = int(part_first_unknowns.min())
            item = blocks[position].item(int(unknowns[position])) or "all items"
            singular_part.items.append(item)
    return [part for part in combined.values() if part.flows]


def not_finite_lines(
    summary: SolutionSummary,
    equations: dict[int, "ProcessEquations"],
    not_computed: Sequence["Process"],
) -> list[str]:
    """Explanation of given values which are not finite, if processes depend on them."""
    rows = summary.not_finite_equations
    not_computed_ids = {process.id for process in not_computed}
    names: list[str] = []
    items: list[str] = []
    seen: set[int] = set()
    for block, row in zip(summary.equation_blocks(rows), rows):
        process = block.key.process
        if process.id not in not_computed_ids or block.offset in seen:
            continue
        seen.add(block.offset)
        names += [
            array.name
            for array in equations[process.id].not_finite_arrays()
            if array.name not in names
        ]
        item = block.item(int(row)) or "all items"
        if len(items) < 3 and item not in items:
            items.append(item)
    if not names:
        return []
    message = (
        f"- The given values {quoted(names)} are not finite (NaN or infinite), e.g. at "
        f"{'; '.join(items)}. What depends on them is not computed."
    )
    return [message]


def relation(
    processes: Sequence["Process"],
    unset_flows: Sequence["Flow"],
    concerned: Sequence["Process"] | None = None,
) -> str:
    """``"Processes 'a', 'b' depend on each other"`` or ``"... are connected by unknown flows"``.

    They depend on each other if the concerned processes (default: all) lie on a cycle of
    unknown flows.
    """
    concerned = processes if concerned is None else concerned
    names = quoted(process.name for process in processes)
    if len(processes) == 1:
        return f"Process {names}"
    on_cycle = processes_on_cycles(unset_flows)
    if any(process.id in on_cycle for process in concerned):
        return f"Processes {names} depend on each other"
    return f"Processes {names} are connected by unknown flows"


def flows_of(processes: Iterable["Process"]) -> dict[str, "Flow"]:
    """All flows of the given processes, by name."""
    return {flow.name: flow for process in processes for flow in unique_flows(process)}
