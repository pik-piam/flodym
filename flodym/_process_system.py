"""Computation of as many processes as possible, see :py:func:`flodym.compute_processes`.

The parts are:

- :py:func:`compute_processes` and :py:func:`run_computation`: the two stages, one process at a
  time and simultaneously, alternating until nothing changes.
- :py:class:`ComputationReport`: which processes were not computed and why, reported once at the
  end.
- :py:class:`BlockAttempts` and :py:func:`find_blocks`: the blocks of processes connected by
  unknown flows, which are solved together.
- :py:func:`solve_block`: solves the equations of a block (``flodym._process_equations``), and
  writes and checks the processes which are determined completely.
"""

import logging
from collections import deque
from collections.abc import Callable, Generator, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property, partial
from typing import TYPE_CHECKING

from ._checks import config_hint
from ._config import ErrorBehavior, config, handle_error
from ._linear_system import Solution, solve
from ._process_compute import (
    SPLITTER_DIRECTIONS,
    Snapshot,
    UnderdeterminedError,
    check_after_computation,
    check_completed_process,
    check_splitter_sum,
    compute_process,
    quoted,
    share_tolerance,
    stock_arrays,
    unique_flows,
)
from ._process_diagnostics import Explanation, explain_block, flows_of, relation
from ._process_equations import ProcessEquations, TooManyUnknownsError, assemble_processes

if TYPE_CHECKING:
    from .flodym_arrays import Flow
    from .processes import Process

logger = logging.getLogger(__name__)

MAX_BLOCK_UNKNOWNS = 2_000_000
"""Blocks with more unknowns are not solved, to limit the memory used."""


def compute_processes(processes: Iterable["Process"], *, simultaneous: bool = True) -> None:
    """Compute the given processes, and all processes which become computable as a consequence.

    This is the engine behind :py:meth:`flodym.MFASystem.compute_all_possible` and the
    ``compute_neighbors`` option of :py:meth:`flodym.Process.compute`. It works in two stages,
    which alternate until nothing changes:

    1. **One process at a time.** The processes to try are kept in a queue, which initially
       contains the given processes. Each is computed as by :py:meth:`flodym.Process.compute`, if
       it is determined by its own flows, shares and stock. When a process is computed, its
       neighbors which are not computed yet are added to the queue, since their flows have
       changed.
    2. **Simultaneously** (if ``simultaneous`` is True). Processes may depend on each other, for
       example in a recycling loop within a time step: production, use, end of life and
       recycling each have an unknown inflow which only another process of the loop provides.
       The processes left by the first stage are grouped into blocks of processes connected by
       unknown flows. The equations of all processes of a block are solved together as one
       linear equation system, item by item of the dimensions of the flows.

    **Equations.** A process without stock has a total throughput. Its dimensions are those that
    all inflows have, and those that all outflows have, except for dimensions which a dimension
    splitter adds. For each side (inflows or outflows), a flow with a share is the share times
    the total; one flow without share is the remaining share times the total (zero where the
    shares sum to one within the tolerance); if two or more flows have no share, the flows of the
    side sum to the total. The total is multiplied with the dimension splitter of a side (the
    forward splitter for outflows, the backward splitter for inflows) if unknown flows of the side
    have dimensions which the flows of the other side do not all have. These are the rules of
    :py:meth:`flodym.Process.compute`, written such that they hold in both directions. For a
    process with stock, the inflows sum to the stock inflow, and the outflows to the stock
    outflow. Supported are stocks which are computed already,
    :py:class:`flodym.SimpleFlowDrivenStock` (its inflow and outflow are given by the flows, and
    the stock is computed from them), and :py:class:`flodym.DynamicStockModel` and its
    subclasses, if they can be computed on their own first: stock-driven with a set stock,
    driven by ``inflow_by_cohort``, or inflow-driven with a set inflow.

    **What is computed.** A process of a block is computed if all its unknown flows and its
    total are determined by the equations. Its flows are then set, and the same checks as after
    :py:meth:`flodym.Process.compute` are run. A flow between two processes which are not
    computed stays unset, even if it is determined. Afterwards, the first stage is run again,
    since the new flows may make further processes computable.

    **What is left out.** A process does not contribute equations if both of its dimension
    splitters would be needed, if a share has dimensions which the total of its side does not
    have, if its arrays mix related dimensions (see :py:attr:`flodym.Dimension.parent`), if its
    stock is not supported (also stock classes which override the computation of a supported
    class), or if anything raises an error while its equations are written or its stock is
    computed on its own, e.g. if lifetime parameters are not set yet. Its flows may still be
    determined by its neighbors. Blocks are not solved if none of their flows and stock arrays is
    known, or if they have more than two million unknowns.

    **How problems are reported.**

    - The processes which stay uncomputed are reported once at the end, at info level, with the
      reasons: the flows which are not determined, and hints such as missing shares or dimension
      splitters.
    - If processes form a loop without exit for some items, i.e. the shares along the loop
      multiply to one, their equations have no unique solution. The loop and all processes
      depending on it stay uncomputed, and the report is a warning.
    - Inconsistent given values of the processes which are computed are handled according to
      ``flodym.config.error_behaviors.process_overdetermined`` and ``mass_balance``, invalid
      shares and splitters according to ``process_shares``, unused splitters according to
      ``unused_dimension_splitter``. If one of these raises an error, all processes of the block
      are restored to their state before. Flows are not checked for negative values; use
      :py:meth:`flodym.MFASystem.check_flows` for that.

    **Known limitation.** If the flows of one side have different dimensions, the total has
    the dimensions common to all of them. The first stage instead takes the dimensions of the
    flow it computes the total from. Both agree without dimension splitters, but with a
    splitter, the results of a process can differ depending on the stage which computes it.

    Args:
        processes: The processes to compute first.
        simultaneous: Whether processes which depend on each other are computed together, by
            solving their equations as one system. If False, processes are only computed one at
            a time.
    """
    run_computation(processes, simultaneous=simultaneous).report(ErrorBehavior.INFO)


def run_computation(processes: Iterable["Process"], *, simultaneous: bool) -> "ComputationReport":
    """Compute processes as described in :py:func:`compute_processes`, without reporting.

    Returns:
        The report of what was not computed.
    """
    report = ComputationReport()
    report.run_queue(processes)
    if not simultaneous:
        return report
    attempts = BlockAttempts()
    while True:
        results = attempts.run(list(report.tried.values()))
        for result in results:
            report.add_block_result(result)
        if not any(result.flows_written for result in results):
            return report
        to_try = [process for result in results for process in result.processes]
        report.run_queue(
            to_try + [neighbor for process in to_try for neighbor in process._neighbors()]
        )


@dataclass
class BlockResult:
    """Outcome of solving the equations of a block of processes together."""

    processes: list["Process"]
    """All processes of the block."""
    flows_written: bool = False
    """Whether flows were set."""
    explain: Callable[[], Explanation] | None = None
    """Creates the explanation why processes are not computed, only when it is reported."""

    @cached_property
    def explanation(self) -> Explanation:
        """Why processes of the block are not computed."""
        return Explanation("") if self.explain is None else self.explain()


class ComputationReport:
    """Which processes were not computed by :py:func:`run_computation`, and why."""

    def __init__(self) -> None:
        self.tried: dict[int, Process] = {}
        """Processes which were tried, by ID."""
        self.reasons: dict[int, str] = {}
        """Why each process could not be computed on its own, by process ID."""
        self.block_results: dict[int, BlockResult] = {}
        """Result of the last block containing each process, by process ID."""

    def run_queue(self, processes: Iterable["Process"]) -> None:
        """Compute processes one at a time, see the first stage of :py:func:`compute_processes`."""
        queue = deque(processes)
        queued = {process.id for process in queue}
        while queue:
            process = queue.popleft()
            queued.discard(process.id)
            if process.id == 0 or process.is_computed:
                continue
            self.tried[process.id] = process
            try:
                compute_process(process, on_underdetermined=ErrorBehavior.ERROR, short_message=True)
            except UnderdeterminedError as error:
                self.reasons[process.id] = error.reason
                continue
            for neighbor in process._neighbors():
                if neighbor.id != 0 and not neighbor.is_computed and neighbor.id not in queued:
                    queue.append(neighbor)
                    queued.add(neighbor.id)

    def add_block_result(self, result: BlockResult) -> None:
        """Record the result of solving a block."""
        for process in result.processes:
            self.tried[process.id] = process
            self.block_results[process.id] = result

    def not_computed(self) -> list["Process"]:
        """The processes which were tried, but are not computed, sorted by ID."""
        return sorted(
            (process for process in self.tried.values() if not process.is_computed),
            key=lambda process: process.id,
        )

    def error(self, not_computed: Sequence["Process"]) -> UnderdeterminedError:
        """Error with the reasons why the given processes are not computed."""
        parts: list[str] = []
        flow_names: list[str] = []
        reported: set[int] = set()
        for process in not_computed:
            result = self.block_results.get(process.id)
            if result is None:
                reason = self.reasons.get(process.id, "It was not tried.").strip()
                parts.append(f"- Process '{process.name}': {_indented(reason)}\n")
            elif id(result) not in reported:
                reported.add(id(result))
                explanation = result.explanation
                parts.append(explanation.message)
                flow_names += [f for f in explanation.undetermined_flows if f not in flow_names]
        return UnderdeterminedError(
            not_computed[0],
            "".join(parts),
            process_names=[process.name for process in not_computed],
            flow_names=flow_names,
        )

    def report(
        self, behavior: ErrorBehavior, not_computed: Sequence["Process"] | None = None
    ) -> None:
        """Report the processes which are not computed, once.

        Args:
            behavior: What to do. With ``"info"``, given values which contradict each other or
                are not finite are reported as a warning.
            not_computed: The processes to report. Defaults to :py:meth:`not_computed`.

        Raises:
            UnderdeterminedError: If processes are not computed and ``behavior`` is ``"error"``.
        """
        if behavior == ErrorBehavior.IGNORE:
            return
        if not_computed is None:
            not_computed = self.not_computed()
        if not not_computed:
            return
        error = self.error(not_computed)
        if behavior == ErrorBehavior.ERROR:
            raise error
        results = [self.block_results.get(process.id) for process in not_computed]
        if behavior == ErrorBehavior.INFO and any(
            result is not None and result.explanation.data_error for result in results
        ):
            behavior = ErrorBehavior.WARN
        handle_error(behavior=behavior, message=str(error))


class BlockAttempts:
    """Blocks solved within one call of :py:func:`run_computation`."""

    def __init__(self) -> None:
        self._signatures: set[tuple[frozenset[int], frozenset[str]]] = set()

    def run(self, processes: Sequence["Process"]) -> list[BlockResult]:
        """Solve the blocks of processes which are not computed, containing the given ones.

        A block is skipped if none of its flows and stock arrays is known, or if its unknown
        flows are the same as at an earlier attempt.

        Returns:
            The results of the blocks which were not skipped as unchanged.
        """
        results = []
        for block in find_blocks(processes):
            unset_flows = [f for f in flows_of(block).values() if not f.is_set]
            signature = (
                frozenset(process.id for process in block),
                frozenset(flow.name for flow in unset_flows),
            )
            if signature in self._signatures:
                continue
            self._signatures.add(signature)
            if _anything_known(block):
                results.append(solve_block(block))
                continue
            message = (
                f"{relation(block, unset_flows)}, but none of their flows and stock arrays is "
                "known (is_set=True).\n"
            )
            results.append(BlockResult(block, explain=lambda message=message: Explanation(message)))
        return results


def find_blocks(processes: Iterable["Process"]) -> list[list["Process"]]:
    """Blocks of processes which are not computed, connected by flows which are not set.

    Only blocks containing one of the given processes are returned. The system boundary
    ('sysenv') is not part of any block. Each block is sorted by process ID.
    """
    seen: set[int] = set()
    blocks = []
    for start in processes:
        if start.id == 0 or start.is_computed or start.id in seen:
            continue
        seen.add(start.id)
        block = []
        stack = [start]
        while stack:
            process = stack.pop()
            block.append(process)
            for flow in unique_flows(process):
                if flow.is_set:
                    continue
                for other in (flow.from_process, flow.to_process):
                    if other.id != 0 and other.id not in seen:
                        seen.add(other.id)
                        stack.append(other)
        blocks.append(sorted(block, key=lambda process: process.id))
    return blocks


def solve_block(processes: Sequence["Process"], max_unknowns: int | None = None) -> BlockResult:
    """Compute processes simultaneously, see :py:func:`compute_processes`.

    Processes which are determined completely are written and checked. If a check raises an
    error, all processes are restored to their state before and the error is re-raised.

    Args:
        processes: The processes, usually connected by unknown flows.
        max_unknowns: If the system has more unknowns, it is not solved. Defaults to
            :py:data:`MAX_BLOCK_UNKNOWNS`.

    Returns:
        Whether flows were written, and why processes are not computed.
    """
    if max_unknowns is None:
        max_unknowns = MAX_BLOCK_UNKNOWNS
    return _Block(processes).solve(max_unknowns)


class BlockSnapshot:
    """Snapshots of all processes of a block, see :py:class:`Snapshot`."""

    def __init__(self, processes: Sequence["Process"]) -> None:
        self._snapshots = {process.id: Snapshot(process) for process in processes}

    def of(self, process: "Process") -> Snapshot:
        """Snapshot of one process."""
        return self._snapshots[process.id]

    def restore(self, processes: Iterable["Process"] | None = None) -> None:
        """Restore the given processes, or all."""
        ids = self._snapshots if processes is None else [process.id for process in processes]
        for process_id in ids:
            self._snapshots[process_id].restore()

    @contextmanager
    def restored_on_error(self) -> Generator[None, None, None]:
        """Restore all processes if the block of the ``with`` statement raises an error."""
        try:
            yield
        except BaseException:
            self.restore()
            raise


class _Block:
    """A block of processes whose equations are solved together, see :py:func:`solve_block`."""

    def __init__(self, processes: Sequence["Process"]) -> None:
        self.processes = sorted(processes, key=lambda process: process.id)
        self.snapshot = BlockSnapshot(self.processes)
        self.equations: dict[int, ProcessEquations] = {}
        """Equations of the processes which take part, by process ID."""
        self.left_out: dict[int, str] = {}
        """Why processes do not take part, by process ID."""

    @property
    def label(self) -> str:
        """``"processes 'a', 'b'"`` or ``"process 'a'"``, for messages."""
        word = "processes" if len(self.processes) > 1 else "process"
        return f"{word} {quoted(process.name for process in self.processes)}"

    def solve(self, max_unknowns: int) -> BlockResult:
        """Solve the equations, write and check the processes which are determined completely."""
        flows = flows_of(self.processes)
        n_unknown_flow_items = sum(flow.size for flow in flows.values() if not flow.is_set)
        if n_unknown_flow_items > max_unknowns:
            return self._too_large(n_unknown_flow_items, max_unknowns)
        with self.snapshot.restored_on_error():
            self._write_equations()
            try:
                system = assemble_processes(list(self.equations.values()), max_unknowns)
                solution = solve(system, share_tolerance(len(self.processes)))
            except TooManyUnknownsError as error:
                self.snapshot.restore()
                return self._too_large(error.n_unknowns, max_unknowns)
            except MemoryError:
                self.snapshot.restore()
                message = f"Not enough memory to solve the equations of {self.label}.\n"
                return BlockResult(self.processes, explain=lambda: Explanation(message))
            complete = [
                equations
                for equations in self.equations.values()
                if equations.is_complete(solution)
            ]
            complete_ids = {equations.process.id for equations in complete}
            self.snapshot.restore(
                process for process in self.processes if process.id not in complete_ids
            )
            written = self._write(complete, solution)
            self._check(complete, solution)
        if complete:
            names = quoted(equations.process.name for equations in complete)
            logger.info(f"Computed processes {names} simultaneously.")
        if all(process.is_computed for process in self.processes):
            return BlockResult(self.processes, flows_written=bool(written))
        return BlockResult(
            self.processes,
            flows_written=bool(written),
            explain=partial(
                explain_block, self.processes, self.equations, self.left_out, solution.summary()
            ),
        )

    def _write_equations(self) -> None:
        """Prepare the stocks and write the equations of all processes which can take part."""
        for process in self.processes:
            try:
                stock = process.stock
                if stock is not None:
                    stock._check_block_computation_supported()
                    stock._prepare_block_computation()
                self.equations[process.id] = ProcessEquations(process)
            except Exception as error:  # noqa: BLE001 - the process is left out, with the reason
                self.snapshot.restore([process])
                reason = error.reason if isinstance(error, UnderdeterminedError) else str(error)
                self.left_out[process.id] = reason.strip()

    def _write(self, complete: Sequence[ProcessEquations], solution: Solution) -> list["Flow"]:
        """Set the unknown flows of the complete processes and compute their stocks.

        Returns:
            The flows which were set.
        """
        written: dict[str, Flow] = {}
        for equations in complete:
            for flow in equations.unknown_flows:
                if flow.name not in written:
                    flow[...] = solution.array_values(flow.name)
                    written[flow.name] = flow
        for equations in complete:
            stock = equations.process.stock
            if stock is not None:
                totals = {
                    side: equations.total(key, solution)
                    for side, key in equations.side_totals.items()
                }
                stock._finish_block_computation(totals)
        return list(written.values())

    def _check(self, complete: Sequence[ProcessEquations], solution: Solution) -> None:
        """Checks after the computation, as after :py:meth:`flodym.Process.compute`."""
        self._check_redundant_equations(complete, solution)
        was_overdetermined = len(solution.redundant_equations) > 0
        for equations in complete:
            process = equations.process
            for side, splitter in equations.splitters.items():
                total_dims = equations.totals[equations.side_totals[side]]
                check_splitter_sum(process, splitter, total_dims, SPLITTER_DIRECTIONS[side])
            for side, process_side in equations.sides.items():
                if process_side.flows:
                    process_side.check_shares_fulfilled(equations.side_total(side, solution))
            check_after_computation(
                process, self.snapshot.of(process), was_overdetermined, equations.unused_splitters
            )
        complete_ids = {equations.process.id for equations in complete}
        for process in self.processes:
            if process.id not in complete_ids and process.is_computed:
                check_completed_process(process)

    def _check_redundant_equations(
        self, complete: Sequence[ProcessEquations], solution: Solution
    ) -> None:
        """Check that the equations not used for the solution are fulfilled.

        Only equations of complete processes are checked. Failures are handled according to
        ``config.error_behaviors.process_overdetermined``.
        """
        behavior = config.error_behaviors.process_overdetermined
        if behavior == ErrorBehavior.IGNORE:
            return
        complete_ids = {equations.process.id for equations in complete}
        failures = []
        for block, max_error, row in solution.max_residuals():
            equation = block.key
            process = equation.process
            if process.id not in complete_ids or max_error <= process.tolerance:
                continue
            location = f" at {block.item(row)}" if block.dims else ""
            failures.append(f"{equation.describe()} (max error {max_error:.3g}{location})")
        if failures:
            handle_error(
                behavior=behavior,
                message=(
                    f"The given values of {self.label} are inconsistent: the processes are "
                    "overdetermined, i.e. more flow values, shares or stock values are given "
                    f"than needed, and these contradict each other. Not fulfilled: "
                    f"{'; '.join(failures)}. Check the shares, dimension splitters and "
                    "prescribed values of these processes."
                    + config_hint("process_overdetermined", tolerance=True)
                ),
            )

    def _too_large(self, n_unknowns: int, max_unknowns: int) -> BlockResult:
        unset_flows = [f for f in flows_of(self.processes).values() if not f.is_set]
        message = (
            f"{relation(self.processes, unset_flows)}, but their equations have {n_unknowns} "
            f"unknowns, more than the limit of {max_unknowns} for solving them together.\n"
        )
        return BlockResult(self.processes, explain=lambda: Explanation(message))


def _anything_known(processes: Sequence["Process"]) -> bool:
    """Whether any flow or stock array of the processes is set."""
    for process in processes:
        if any(flow.is_set for flow in unique_flows(process)):
            return True
        if process.stock is not None and any(array.is_set for array in stock_arrays(process.stock)):
            return True
    return False


def _indented(text: str) -> str:
    """Text with all lines but the first indented, for list items in messages."""
    return text.replace("\n", "\n  ")
