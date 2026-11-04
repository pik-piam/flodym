"""Automatic computation of processes, see :py:meth:`flodym.Process.compute`.

The computation is split into two classes:

- :py:class:`ProcessSide` represents the inflows or the outflows of a process together with
  their shares. It contains everything that concerns only one side: computing the total
  throughput from the flows, and computing the unknown flows from a total.
- :py:class:`ProcessComputation` represents one computation of a process. It checks beforehand
  that the process can be computed, combines the two sides (and the stock, if any), applies the
  dimension splitters, and performs the checks after the computation.

Processes with a stock are computed by the stock (``Stock._compute_process``), which uses the
methods of :py:class:`ProcessComputation`.

All checks whether a process is determined are done before any array is changed, so that an
underdetermined process stays unchanged.
"""

import logging
from collections import deque
from typing import TYPE_CHECKING, Dict, Iterable, List, Literal, Optional, Sequence, Union

import numpy as np

from ._checks import config_hint, max_abs, report_balance_failures
from .config import ErrorBehavior, config, handle_error
from .dimensions import DimensionSet

if TYPE_CHECKING:
    from .flodym_arrays import FlodymArray, Flow
    from .processes import Process

Side = Literal["in", "out"]
"""Side of a process: ``"in"`` for its inflows, ``"out"`` for its outflows."""
SIDES: tuple[Side, Side] = ("in", "out")

Direction = Literal["forward", "backward"]
"""Direction of a dimension splitter: ``"forward"`` splitters are used to compute outflows,
``"backward"`` splitters to compute inflows."""
SPLITTER_SIDES: Dict[Direction, Side] = {"forward": "out", "backward": "in"}
"""The side whose flows a dimension splitter of each direction is used for."""
SPLITTER_DIRECTIONS: Dict[Side, Direction] = {side: d for d, side in SPLITTER_SIDES.items()}

TotalMethod = Literal["sum", "share", "remainder"]
"""Ways to compute the total throughput of a process from the flows of one side, in order of
preference (see :py:meth:`ProcessSide.total_method`)."""
TOTAL_METHODS: tuple[TotalMethod, ...] = ("sum", "share", "remainder")


class UnderdeterminedError(Exception):
    """Raised by :py:meth:`flodym.Process.compute` if the given flows, shares and stock values
    are not sufficient to compute the process."""

    def __init__(self, process: "Process", message: str):
        self.process_name = process.name
        self.short_message = f"Process '{process.name}' is underdetermined. Skip computation."
        super().__init__(
            f"Cannot compute process '{process.name}' (ID {process.id}), as it is "
            f"underdetermined.\n{message}" + config_hint("process_underdetermined")
        )


def share_tolerance(n_summed: int = 1) -> float:
    """Tolerance for checks of dimensionless shares and dimension splitters, summed over
    ``n_summed`` values."""
    return config.relative_tolerance * float(np.finfo(np.float64).eps) * max(1, n_summed)


def check_share_values(process: "Process", share: "FlodymArray", label: str) -> None:
    """Check that a share or dimension splitter is between zero and one, handling failures
    according to ``config.error_behaviors.process_shares``."""
    if config.error_behaviors.process_shares == ErrorBehavior.IGNORE:
        return
    tolerance = share_tolerance()
    values = share.values
    if np.any(np.isnan(values)) or np.any(values < -tolerance) or np.any(values > 1 + tolerance):
        handle_error(
            behavior=config.error_behaviors.process_shares,
            message=(
                f"In process '{process.name}': {label} ('{share.name}') has values which are "
                "NaN or not between zero and one." + config_hint("process_shares")
            ),
        )


def union_of_dims(arrays: Iterable["FlodymArray"]) -> DimensionSet:
    """Union of the dimensions of several arrays."""
    dims = DimensionSet(dim_list=[])
    for array in arrays:
        dims |= array.dims
    return dims


class ProcessSide:
    """The inflows (``side="in"``) or outflows (``side="out"``) of a process, with their shares.

    The shares of a side refer to the total of this side, which is the total throughput of a
    process without stock, and the inflow or outflow of the stock otherwise.
    Flows count as known if they are set (see :py:attr:`flodym.FlodymArray.is_set`).
    """

    def __init__(self, process: "Process", side: Side):
        self.process = process
        self.side = side
        self.flows: Dict[str, "Flow"] = process._inflows if side == "in" else process._outflows
        """Flows of this side, keyed by flow name."""
        self.shares: Dict[str, "FlodymArray"] = (
            process._inflow_shares if side == "in" else process._outflow_shares
        )
        """Shares of flows of this side, keyed by flow name."""

    @property
    def label(self) -> str:
        """``"inflow"`` or ``"outflow"``, for messages."""
        return f"{self.side}flow"

    @property
    def known(self) -> Dict[str, "Flow"]:
        """Flows with known values."""
        return {name: f for name, f in self.flows.items() if f.is_set}

    @property
    def unknown(self) -> Dict[str, "Flow"]:
        """Flows with unknown values."""
        return {name: f for name, f in self.flows.items() if not f.is_set}

    @property
    def neither_known(self) -> List[str]:
        """Names of flows of which neither the value nor the share is given."""
        return [name for name in self.unknown if name not in self.shares]

    @property
    def both_known(self) -> List[str]:
        """Names of flows of which both the value and the share are given."""
        return [name for name in self.known if name in self.shares]

    def check_shares_linked(self) -> None:
        """Check that all shares belong to flows linked to the process."""
        unlinked = [name for name in self.shares if name not in self.flows]
        if unlinked:
            raise ValueError(
                f"In process '{self.process.name}': Shares are given for the {self.label}s "
                f"{', '.join(unlinked)}, which are not linked to the process."
            )

    # Computing the total from the flows ---------------------------------------------------------

    def total_method(self) -> Optional[TotalMethod]:
        """How the total of this side can be computed from its flows, or None if it cannot:

        - ``"sum"``: All flows are known; the total is their sum.
        - ``"share"``: A flow is known and has a share; the total is the flow divided by it.
        - ``"remainder"``: Some flows are known, and all unknown flows have shares; the total is
          ``sum(known flows) / (1 - sum(shares of unknown flows))``.
        """
        if not self.flows:
            return None
        if not self.unknown:
            return "sum"
        if self.both_known:
            return "share"
        if self._remainder_possible():
            return "remainder"
        return None

    def _remainder_possible(self) -> bool:
        return not self.neither_known and bool(self.known)

    def compute_total(self) -> "FlodymArray":
        """Compute the total of this side from its flows, see :py:meth:`total_method`.
        The total has the dimensions common to all flows used."""
        method = self.total_method()
        if method == "sum":
            return self._sum(self.flows.values())
        if method == "share":
            return self._total_from_share()
        if method == "remainder":
            return self._total_from_remainder()
        raise UnderdeterminedError(self.process, self.total_failure_message())

    @staticmethod
    def _sum(arrays: Iterable["FlodymArray"]) -> "FlodymArray":
        """Sum of a non-empty collection of arrays, with the dimensions common to all."""
        arrays = list(arrays)
        return sum(arrays[1:], arrays[0])

    def _total_from_share(self) -> "FlodymArray":
        """Total as a known flow divided by its share. Flows whose share has zeros are skipped,
        since the total is not determined where the share is zero."""
        for name in self.both_known:
            flow, share = self.flows[name], self.shares[name]
            excess_dims = share.dims - flow.dims
            if excess_dims:
                raise ValueError(
                    f"In process '{self.process.name}': The share of flow '{name}' has the "
                    f"dimensions {excess_dims.names}, which the flow does not have."
                )
            if not np.any(share.values == 0):
                return flow / share
        if self._remainder_possible():
            return self._total_from_remainder()
        raise ValueError(
            f"In process '{self.process.name}': Cannot compute the total throughput from the "
            f"{self.label}s {', '.join(self.both_known)} and their shares, since the shares are "
            "zero for some items."
        )

    def _total_from_remainder(self) -> "FlodymArray":
        """Total as ``sum(known flows) / (1 - sum(shares of unknown flows))``."""
        known = self.known
        sum_known = self._sum(known.values())
        shares_unknown = {name: self.shares[name] for name in self.unknown}
        dims_unknown = union_of_dims(shares_unknown.values())
        excess_dims = dims_unknown - sum_known.dims
        if excess_dims:
            share_names = [n for n, s in shares_unknown.items() if s.dims - sum_known.dims]
            raise ValueError(
                f"In process '{self.process.name}': Cannot compute the total throughput from the "
                f"known {self.label}s {', '.join(known)} and the shares of the unknown flows "
                f"{', '.join(share_names)}, since these shares have the dimensions "
                f"{excess_dims.names}, which not all known flows have."
            )
        # shares are cast to common dimensions first, since + would sum them over others
        sum_shares = self._sum(s.cast_to(dims_unknown) for s in shares_unknown.values())
        remaining_share = 1.0 - sum_shares
        if np.any(np.isclose(remaining_share.values, 0.0)):
            raise ValueError(
                f"In process '{self.process.name}': Cannot compute the total throughput from the "
                f"known {self.label}s {', '.join(known)}, since the shares of the unknown flows "
                "sum to one for some items, so the known flows must be zero there."
            )
        return sum_known / remaining_share

    def total_failure_message(self) -> str:
        """Explanation why the total cannot be computed from this side."""
        if not self.flows:
            return f"From the {self.label}s, since the process has none.\n"
        message = f"From the {self.label}s, since:\n"
        if not self.known:
            message += f"- no {self.label} is known (is_set=True)\n"
        for name in self.neither_known:
            message += f"- for flow '{name}', neither share nor value is given\n"
        return message

    # Computing the flows from the total ---------------------------------------------------------

    def can_compute_flows(self) -> bool:
        """Whether the unknown flows can be computed from a total, i.e. whether at most one
        unknown flow has no share."""
        return len(self.neither_known) <= 1

    def flows_failure_message(self) -> str:
        """Explanation why the unknown flows cannot be computed from a total."""
        return (
            f"Cannot compute the {self.label}s from the total, since for more than one of them, "
            f"neither share nor value is given: {', '.join(self.neither_known)}. Give a share or "
            "value for all but one of them.\n"
        )

    def check_flow_dims(self, total_dims: DimensionSet) -> None:
        """Check that the unknown flows can be computed from a total with the given dimensions,
        so that errors are raised before any flow is changed."""
        for name in self.unknown:
            if name not in self.shares:
                continue
            excess_dims = self.shares[name].dims - total_dims
            if excess_dims:
                raise ValueError(
                    f"In process '{self.process.name}': The share of flow '{name}' has the "
                    f"dimensions {excess_dims.names}, which the total throughput of the process "
                    "does not have. To add these dimensions to the flow, insert a separate "
                    "process with a dimension splitter."
                )
        remaining = [name for name in self.unknown if name not in self.shares]
        if not remaining:
            return
        # The remaining flow is the total minus all other flows. Flows computed from shares have
        # the dimensions of the total, so only the known flows can lack dimensions.
        available_dims = total_dims
        for flow in self.known.values():
            available_dims &= flow.dims
        missing_dims = self.flows[remaining[0]].dims - available_dims
        if missing_dims:
            raise ValueError(
                f"In process '{self.process.name}': Cannot compute flow '{remaining[0]}' as the "
                f"total throughput minus the other {self.label}s, since the total or a known "
                f"{self.label} does not have its dimensions {missing_dims.names}."
            )

    def compute_flows(self, total: "FlodymArray") -> None:
        """Compute the unknown flows from the total: flows with a share as total times share, the
        one flow without share (if any) as total minus all other flows."""
        unknown = self.unknown
        # Computed flows are kept with the dimensions of the total until the remaining flow is
        # computed: Reducing them to the flow dimensions first could lose dimensions it needs.
        computed = {
            name: total * self.shares[name] for name in unknown if name in self.shares
        }
        remaining = [name for name in unknown if name not in computed]
        if remaining:
            others = list(self.known.values()) + list(computed.values())
            unknown[remaining[0]][...] = total - self._sum(others) if others else total
        for name, values in computed.items():
            unknown[name][...] = values

    # Checks of the shares -----------------------------------------------------------------------

    def check_share_sums(self) -> None:
        """Check that the shares are between zero and one and sum to at most one, or to exactly
        one if all flows of the side have a share."""
        if not self.shares or config.error_behaviors.process_shares == ErrorBehavior.IGNORE:
            return
        for name, share in self.shares.items():
            check_share_values(self.process, share, f"Share of flow '{name}'")
        dims = union_of_dims(self.shares.values())
        share_sum = self._sum(s.cast_to(dims) for s in self.shares.values()).values
        tolerance = share_tolerance(n_summed=len(self.shares))
        if len(self.shares) == len(self.flows):
            failed = not np.all(np.abs(share_sum - 1.0) <= tolerance)
            expected = "sum to one, since all of them have shares"
        else:
            failed = not np.all(share_sum <= 1.0 + tolerance)
            expected = "sum to at most one"
        if failed:
            handle_error(
                behavior=config.error_behaviors.process_shares,
                message=(
                    f"In process '{self.process.name}': The shares of the {self.label}s "
                    f"({', '.join(self.shares)}) do not {expected}." + config_hint("process_shares")
                ),
            )

    def check_shares_fulfilled(self, total: "FlodymArray") -> None:
        """Check that the flows fulfill their shares after the computation, which may not be the
        case if the process is overdetermined."""
        if config.error_behaviors.process_shares == ErrorBehavior.IGNORE:
            return
        tolerance = self.process.tolerance
        failures = []
        for name, share in self.shares.items():
            max_error = max_abs(total * share - self.flows[name])
            if not max_error <= tolerance:
                failures.append(f"'{name}' (max error: {max_error})")
        if failures:
            handle_error(
                behavior=config.error_behaviors.process_shares,
                message=(
                    f"In process '{self.process.name}': After computation, the shares of the "
                    f"{self.label}s {', '.join(failures)} are not fulfilled. This may be due to "
                    "the process being overdetermined, i.e. more flow values or shares being "
                    "given than needed." + config_hint("process_shares", tolerance=True)
                ),
            )


class ProcessComputation:
    """One computation of a process, see :py:meth:`flodym.Process.compute`.

    Processes without stock are computed by :py:meth:`run`. For processes with stock,
    :py:meth:`run` calls ``Stock._compute_process``, which uses the building blocks
    :py:meth:`require_total`, :py:meth:`require_flows`, :py:meth:`compute_total` and
    :py:meth:`compute_side`.
    """

    def __init__(self, process: "Process"):
        self.process = process
        self.sides: Dict[Side, ProcessSide] = {side: ProcessSide(process, side) for side in SIDES}
        self.unused_splitters: List[Direction] = []
        """Directions of dimension splitters which were not applied when computing the flows of
        their side, since no dimensions had to be added."""

    # Checks before the computation --------------------------------------------------------------

    def require_total(self, sides: Sequence[Side] = SIDES) -> Side:
        """Side from which the total can be computed, preferring the methods in the order of
        :py:data:`TOTAL_METHODS`. Raises an UnderdeterminedError if there is none."""
        candidates = []
        for side in sides:
            method = self.sides[side].total_method()
            if method is not None:
                candidates.append((TOTAL_METHODS.index(method), side))
        if not candidates:
            message = "Cannot compute the total throughput of the process.\n"
            message += "".join(self.sides[side].total_failure_message() for side in sides)
            raise UnderdeterminedError(self.process, message)
        return min(candidates)[1]

    def require_flows(self, sides: Sequence[Side] = SIDES) -> None:
        """Raise an UnderdeterminedError if the unknown flows of the given sides cannot be
        computed from a total."""
        failed = [self.sides[side] for side in sides if not self.sides[side].can_compute_flows()]
        if failed:
            message = "".join(side.flows_failure_message() for side in failed)
            raise UnderdeterminedError(self.process, message)

    # Computation --------------------------------------------------------------------------------

    def run(self) -> None:
        """Compute the process, with its stock, if any."""
        for side in self.sides.values():
            side.check_shares_linked()
        stock = self.process.stock
        if stock is not None:
            stock._compute_process(self)
            return
        # Process without stock: inflows -> total -> outflows. The total of each side is the
        # common total, extended by the dimension splitter of the side if needed.
        total_side = self.require_total()
        self.require_flows()
        total = self.compute_total(total_side)
        side_totals = {}
        for side in SIDES:
            side_totals[side] = self.apply_dimension_splitter(total, side)
            self.sides[side].check_flow_dims(side_totals[side].dims)
        for side in SIDES:
            self.sides[side].compute_flows(side_totals[side])
            self.sides[side].check_shares_fulfilled(side_totals[side])

    def compute_total(self, side: Side) -> "FlodymArray":
        """Total of one side, computed from its flows."""
        return self.sides[side].compute_total()

    def compute_side(self, total: "FlodymArray", side: Side) -> None:
        """Compute the unknown flows of one side from the total of this side, which is extended
        by the dimension splitter of the side if needed. Used for processes with stock."""
        total = self.apply_dimension_splitter(total, side)
        self.sides[side].check_flow_dims(total.dims)
        self.sides[side].compute_flows(total)
        self.sides[side].check_shares_fulfilled(total)

    def apply_dimension_splitter(self, total: "FlodymArray", side: Side) -> "FlodymArray":
        """Multiply the total with the dimension splitter of the given side (the forward splitter
        for the outflows, the backward splitter for the inflows), if unknown flows of the side
        have dimensions which the total does not have. Otherwise, return it unchanged."""
        direction = SPLITTER_DIRECTIONS[side]
        splitter = self.process._dimension_splitters.get(direction)
        unknown_flows = self.sides[side].unknown.values()
        missing_dims = union_of_dims(unknown_flows) - total.dims
        if not missing_dims:
            if splitter is not None and unknown_flows:
                self.unused_splitters.append(direction)
            return total

        name = self.process.name
        label = self.sides[side].label
        if splitter is None:
            raise ValueError(
                f"In process '{name}': Unknown {label}s have the dimensions "
                f"{missing_dims.names}, which their total does not have. Set a {direction} "
                "dimension splitter to add them."
            )
        if missing_dims - splitter.dims:
            raise ValueError(
                f"In process '{name}': The {direction} dimension splitter does not have the "
                f"dimensions {(missing_dims - splitter.dims).names}, which unknown {label}s "
                "have, but their total does not have."
            )
        self._check_splitter_sum(splitter, total.dims, direction)
        return total * splitter

    def _check_splitter_sum(
        self, splitter: "FlodymArray", total_dims: DimensionSet, direction: Direction
    ) -> None:
        """Check that the dimension splitter sums to one over the dimensions it adds."""
        if config.error_behaviors.process_shares == ErrorBehavior.IGNORE:
            return
        added_dims = splitter.dims - total_dims
        splitter_sum = splitter.sum_to((splitter.dims & total_dims).letters)
        tolerance = share_tolerance(n_summed=int(np.prod(added_dims.shape)))
        max_error = max_abs(splitter_sum - 1.0)
        if not max_error <= tolerance:
            handle_error(
                behavior=config.error_behaviors.process_shares,
                message=(
                    f"In process '{self.process.name}': The {direction} dimension splitter does "
                    "not sum to "
                    f"one over the dimensions it adds ({', '.join(added_dims.names)}), with a "
                    f"maximum error of {max_error}." + config_hint("process_shares")
                ),
            )

    # Checks after the computation ---------------------------------------------------------------

    def check_result(self, was_overdetermined: bool) -> None:
        """Checks after a successful computation."""
        process = self.process
        if not process.is_computed:
            raise RuntimeError(
                f"After computation of process '{process.name}', there are still unknown flows or "
                "stock arrays. This indicates an internal flodym error."
            )
        for side in self.sides.values():
            side.check_share_sums()
        for direction in self.unused_splitters:
            label = self.sides[SPLITTER_SIDES[direction]].label
            handle_error(
                behavior=config.error_behaviors.unused_dimension_splitter,
                message=(
                    f"In process '{process.name}': The {direction} dimension splitter is set, but "
                    f"was not used to compute the {label}s, since no dimensions had to be added. "
                    "The split it gives may not be fulfilled."
                    + config_hint("unused_dimension_splitter")
                ),
            )
        if was_overdetermined or config.checks.mass_balance_after_compute:
            hint = config_hint("mass_balance", tolerance=True)
            if was_overdetermined:
                hint = " The process was overdetermined, so the given values may be inconsistent."
                hint += config_hint("mass_balance", tolerance=True)
            report_balance_failures(
                [process._mass_balance_failure()],
                config.error_behaviors.mass_balance,
                hint=hint,
                success_message=f"Mass balance of process '{process.name}' is consistent.",
            )


def is_overdetermined(process: "Process") -> bool:
    """See :py:attr:`flodym.Process.is_overdetermined`."""
    sides = [ProcessSide(process, side) for side in SIDES]
    n_flows = sum(len(side.flows) for side in sides)
    n_given = sum(len(side.known) + len(side.shares) for side in sides)
    stock = process.stock
    if stock is None:
        # unknowns: flows and total; equations: sum of each side, given flows and shares
        n_unknowns = n_flows + 1
        n_equations = 2 + n_given
    else:
        # unknowns: flows, stock inflow, outflow and stock; equations: sum of each side, given
        # flows, shares and stock arrays, and the stock model
        n_unknowns = n_flows + 3
        n_equations = 2 + n_given + stock._n_equations
        n_equations += sum(a.is_set for a in (stock.inflow, stock.outflow, stock.stock))
    return n_equations > n_unknowns


def compute_process(
    process: "Process", on_underdetermined: Union[ErrorBehavior, str, None], short_message: bool
) -> bool:
    """Compute a single process, see :py:meth:`flodym.Process.compute`.

    Args:
        process: The process.
        on_underdetermined: What to do if the process is underdetermined. If None,
            ``config.error_behaviors.process_underdetermined`` is used.
        short_message: Whether to report underdetermination in one line.

    Returns:
        Whether the process is computed afterwards.
    """
    if process.id == 0:
        logging.debug(f"Process '{process.name}' is the system boundary and is not computed.")
        return False
    if process.is_computed:
        logging.debug(f"Process '{process.name}' is already computed.")
        return True
    if on_underdetermined is None:
        on_underdetermined = config.error_behaviors.process_underdetermined
    on_underdetermined = ErrorBehavior(on_underdetermined)

    was_overdetermined = is_overdetermined(process)
    computation = ProcessComputation(process)
    try:
        computation.run()
    except UnderdeterminedError as error:
        if on_underdetermined == ErrorBehavior.ERROR:
            raise
        message = error.short_message if short_message else str(error)
        handle_error(behavior=on_underdetermined, message=message)
        return False
    computation.check_result(was_overdetermined)
    return True


def compute_processes(processes: Iterable["Process"]) -> None:
    """Compute the given processes, and all processes which become computable as a consequence,
    until no further process can be computed (see :py:meth:`flodym.Process.compute`).
    Underdetermined processes are skipped with an info log message.

    The processes to try are kept in a queue, which initially contains the given processes. When
    a process is computed, its neighbors which are not computed yet are added to the queue, since
    their flows have changed. As each process can be computed only once, this terminates.
    """
    queue = deque(processes)
    queued = {id(p) for p in queue}
    while queue:
        process = queue.popleft()
        queued.discard(id(process))
        if process.id == 0 or process.is_computed:
            continue
        if not compute_process(process, on_underdetermined=ErrorBehavior.INFO, short_message=True):
            continue
        for neighbor in process._neighbors():
            if neighbor.id != 0 and not neighbor.is_computed and id(neighbor) not in queued:
                queue.append(neighbor)
                queued.add(id(neighbor))
