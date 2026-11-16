"""Automatic computation of processes, see :py:meth:`flodym.Process.compute`.

The computation is split into these parts:

- :py:class:`ProcessSide` represents the inflows or the outflows of a process together with
  their shares. It contains everything that concerns only one side: computing the total from the
  flows, computing the unknown flows from a total, and checking the shares.
- :py:class:`ProcessComputation` represents one computation of a process. It checks that the
  process can be computed, combines the two sides (and the stock, if any), and applies the
  dimension splitters. Processes with a stock are computed by the stock
  (``Stock._compute_process``), using the methods of :py:class:`ProcessComputation`.
- :py:class:`Snapshot` stores the arrays a computation may change, so that they are restored
  if the computation fails.
- :py:func:`compute_process` runs one computation with the checks before and after it.

The functions which check a process after its computation, and the rules shared with the
simultaneous computation of processes (``flodym._process_equations``), such as the use of
dimension splitters and the remaining share, are defined here as well.
:py:func:`flodym.compute_processes` (in ``flodym._process_system``) computes as many processes as
possible.
"""

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from ._checks import config_hint, max_abs, report_balance_failures
from ._config import ErrorBehavior, config, handle_error
from .dimensions import DimensionSet

if TYPE_CHECKING:
    from .flodym_arrays import FlodymArray, Flow
    from .processes import Process
    from .stocks import Stock

logger = logging.getLogger(__name__)

Side = Literal["in", "out"]
"""Side of a process: ``"in"`` for its inflows, ``"out"`` for its outflows."""
SIDES: tuple[Side, Side] = ("in", "out")
"""Both sides of a process."""
OTHER_SIDE: dict[Side, Side] = {"in": "out", "out": "in"}
"""The opposite of each side."""

Direction = Literal["forward", "backward"]
"""Direction of a dimension splitter: ``"forward"`` splitters are used to compute outflows,
``"backward"`` splitters to compute inflows."""
SPLITTER_SIDES: dict[Direction, Side] = {"forward": "out", "backward": "in"}
"""The side whose flows a dimension splitter of each direction is used for."""
SPLITTER_DIRECTIONS: dict[Side, Direction] = {side: d for d, side in SPLITTER_SIDES.items()}
"""The direction of the dimension splitter used for the flows of each side."""

TotalMethod = Literal["sum", "share", "remainder"]
"""Ways to compute the total of a side from its flows, in order of preference
(see :py:meth:`ProcessSide.total_method`)."""
TOTAL_METHODS: tuple[TotalMethod, ...] = ("sum", "share", "remainder")


class UnderdeterminedError(Exception):
    """Raised if processes cannot be computed from the given values.

    :py:meth:`flodym.Process.compute` raises it for a single process, if the process cannot be
    computed from the given flows, shares and stock values, or if its stock type does not support
    the automatic computation. :py:meth:`flodym.MFASystem.compute_all_possible` raises it with
    ``on_underdetermined="error"`` for all processes it could not compute, including processes
    which depend on each other and were solved together (see
    :py:func:`flodym.compute_processes`).

    Args:
        process: The process which cannot be computed. If the error concerns several processes,
            the first of them.
        message: Explanation why the process or processes cannot be computed.
        process_names: Names of all processes concerned, if the error concerns several processes
            or is raised by :py:meth:`flodym.MFASystem.compute_all_possible`.
        flow_names: Names of the flows which are not determined, if known.
        concerns_flows: Whether the process is underdetermined since its flows, shares and
            dimension splitters do not determine it, rather than because of its stock.
    """

    def __init__(
        self,
        process: "Process",
        message: str,
        *,
        process_names: Sequence[str] | None = None,
        flow_names: Sequence[str] | None = None,
        concerns_flows: bool = False,
    ) -> None:
        self.process_name = process.name
        """Name of the process. If the error concerns several processes, the first of
        :py:attr:`process_names`."""
        self.process_names = [process.name] if process_names is None else list(process_names)
        """Names of all processes concerned."""
        self.flow_names = [] if flow_names is None else list(flow_names)
        """Names of the flows which are not determined by the given values. Empty if not
        known."""
        self.reason = message
        """Explanation why the process cannot be computed."""
        self._concerns_flows = concerns_flows
        names = quoted(self.process_names)
        several = len(self.process_names) > 1
        self.short_message = (
            f"Process{'es' if several else ''} {names} {'are' if several else 'is'} "
            "underdetermined. Skip computation."
        )
        """One-line message, used when underdetermined processes are skipped."""
        if process_names is None:
            header = f"Cannot compute process {names} (ID {process.id}), as it is underdetermined."
            hint = config_hint("process_underdetermined")
        else:
            header = (
                f"Cannot compute process{'es' if several else ''} {names}, as "
                f"{'they are' if several else 'it is'} underdetermined."
            )
            hint = ""
        super().__init__(f"{header}\n{message.rstrip()}{hint}")


DEPENDING_PROCESSES_HINT = (
    "Processes which depend on each other, e.g. in a recycling loop, can only be computed "
    "together, with MFASystem.compute_all_possible()."
)
"""Added to the message of :py:meth:`flodym.Process.compute` if the process lies on a cycle of
unknown flows."""


def quoted(names: Iterable[str]) -> str:
    """Names in quotes, separated by commas, for messages."""
    return ", ".join(f"'{name}'" for name in names)


def share_tolerance(n_summed: int = 1) -> float:
    """Tolerance for checks of dimensionless shares and dimension splitters.

    The checked quantity is a sum over ``n_summed`` values.
    """
    return config.relative_tolerance * float(np.finfo(np.float64).eps) * max(1, n_summed)


def check_share_values(process: "Process", share: "FlodymArray", label: str) -> None:
    """Check that a share or dimension splitter is between zero and one.

    Failures are handled according to ``config.error_behaviors.process_shares``.
    """
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


def common_dims(arrays: Iterable["FlodymArray"]) -> DimensionSet | None:
    """Dimensions which all given arrays have, None if there are no arrays."""
    dims = None
    for array in arrays:
        dims = array.dims if dims is None else dims & array.dims
    return dims


def sum_arrays(arrays: Iterable["FlodymArray"]) -> "FlodymArray":
    """Sum of a non-empty collection of arrays, with the dimensions common to all."""
    arrays = list(arrays)
    return sum(arrays[1:], arrays[0])


def sum_of_shares(shares: Iterable["FlodymArray"]) -> "FlodymArray":
    """Sum of a non-empty collection of shares, with the union of their dimensions."""
    shares = list(shares)
    dims = union_of_dims(shares)
    # shares are cast to common dimensions first, since + would sum them over others
    return sum_arrays(share.cast_to(dims) for share in shares)


def remaining_share(shares: Iterable["FlodymArray"]) -> "FlodymArray":
    """One minus the sum of a non-empty collection of shares.

    Values within the tolerance of zero (see :py:func:`share_tolerance`) are set to exactly zero:
    there, the shares sum to one, so that a total cannot be computed from the flows which
    receive the remaining share.
    """
    shares = list(shares)
    remaining = 1.0 - sum_of_shares(shares)
    remaining.values[np.abs(remaining.values) <= share_tolerance(len(shares))] = 0.0
    return remaining


def unique_flows(process: "Process") -> list["Flow"]:
    """In- and outflows of a process, each once, also a flow from the process to itself."""
    flows = [*process.inflows.values(), *process.outflows.values()]
    return list({id(flow): flow for flow in flows}.values())


def processes_on_cycles(flows: Iterable["Flow"]) -> set[int]:
    """IDs of the processes (other than 'sysenv') on a directed cycle of the given flows."""
    edges = [
        (flow.from_process.id, flow.to_process.id)
        for flow in flows
        if flow.from_process.id != 0 and flow.to_process.id != 0
    ]
    if not edges:
        return set()
    ids = sorted({process_id for edge in edges for process_id in edge})
    index = {process_id: position for position, process_id in enumerate(ids)}
    sources = np.array([index[source] for source, _ in edges])
    targets = np.array([index[target] for _, target in edges])
    graph = sparse.csr_matrix((np.ones(len(edges)), (sources, targets)), shape=(len(ids),) * 2)
    _, components = connected_components(graph, directed=True, connection="strong")
    sizes = np.bincount(components)
    self_loops = {source for source, target in zip(sources, targets) if source == target}
    return {
        process_id
        for position, process_id in enumerate(ids)
        if sizes[components[position]] > 1 or position in self_loops
    }


def on_cycle_of_unknown_flows(process: "Process") -> bool:
    """Whether the process lies on a directed cycle of flows which are not set.

    The system boundary ('sysenv') does not close cycles.
    """
    unknown_flows: dict[int, Flow] = {}
    seen = {process.id}
    stack = [process]
    while stack:
        for flow in unique_flows(stack.pop()):
            if flow.is_set or id(flow) in unknown_flows:
                continue
            unknown_flows[id(flow)] = flow
            for other in (flow.from_process, flow.to_process):
                if other.id != 0 and other.id not in seen:
                    seen.add(other.id)
                    stack.append(other)
    return process.id in processes_on_cycles(unknown_flows.values())


@dataclass
class SplitterUse:
    """How the dimension splitter of a side is used, see :py:meth:`ProcessSide.splitter_use`."""

    splitter: Optional["FlodymArray"] = None
    """The splitter, if it is needed and given."""
    unused: bool = False
    """Whether the splitter is set, but not needed, although the side has unknown flows."""
    missing_message: str | None = None
    """Explanation, if a splitter is needed, but not given."""


def check_splitter_sum(
    process: "Process", splitter: "FlodymArray", total_dims: DimensionSet, direction: Direction
) -> None:
    """Check that a dimension splitter sums to one over the dimensions it adds to a total.

    Failures are handled according to ``config.error_behaviors.process_shares``.
    """
    if config.error_behaviors.process_shares == ErrorBehavior.IGNORE:
        return
    added_dims = splitter.dims - total_dims
    splitter_sum = splitter.sum_to((splitter.dims & total_dims).letters, check_extensivity=False)
    tolerance = share_tolerance(n_summed=int(np.prod(added_dims.shape)))
    max_error = max_abs(splitter_sum - 1.0)
    if not max_error <= tolerance:
        handle_error(
            behavior=config.error_behaviors.process_shares,
            message=(
                f"In process '{process.name}': The {direction} dimension splitter does "
                f"not sum to one over the dimensions it adds ({', '.join(added_dims.names)}), "
                f"with a maximum error of {max_error}." + config_hint("process_shares")
            ),
        )


class ProcessSide:
    """The inflows (``side="in"``) or outflows (``side="out"``) of a process, with their shares.

    The shares of a side refer to the total of this side, which is the total throughput of a
    process without stock, and the inflow or outflow of the stock otherwise.
    Flows count as known if they are set (see :py:attr:`flodym.FlodymArray.is_set`).
    """

    def __init__(self, process: "Process", side: Side) -> None:
        self.process = process
        self.side = side
        self.flows: dict[str, Flow] = process._inflows if side == "in" else process._outflows
        """Flows of this side, keyed by flow name."""
        self.shares: dict[str, FlodymArray] = (
            process._inflow_shares if side == "in" else process._outflow_shares
        )
        """Shares of flows of this side, keyed by flow name."""

    @property
    def label(self) -> str:
        """``"inflow"`` or ``"outflow"``, for messages."""
        return f"{self.side}flow"

    @property
    def direction(self) -> Direction:
        """Direction of the dimension splitter used for the flows of this side."""
        return SPLITTER_DIRECTIONS[self.side]

    @property
    def known(self) -> dict[str, "Flow"]:
        """Flows with known values."""
        return {name: f for name, f in self.flows.items() if f.is_set}

    @property
    def unknown(self) -> dict[str, "Flow"]:
        """Flows with unknown values."""
        return {name: f for name, f in self.flows.items() if not f.is_set}

    @property
    def neither_known(self) -> list[str]:
        """Names of flows of which neither the value nor the share is given."""
        return [name for name in self.unknown if name not in self.shares]

    @property
    def both_known(self) -> list[str]:
        """Names of flows of which both the value and the share are given."""
        return [name for name in self.known if name in self.shares]

    @property
    def n_redundant_shares(self) -> int:
        """Number of shares which give no information beyond the others.

        If all flows of the side have shares, the last one follows from the others, as they sum
        to one.
        """
        return 1 if self.flows and len(self.shares) == len(self.flows) else 0

    def check_shares_linked(self) -> None:
        """Check that all shares belong to flows linked to the process."""
        unlinked = [name for name in self.shares if name not in self.flows]
        if unlinked:
            raise ValueError(
                f"In process '{self.process.name}': Shares are given for the {self.label}s "
                f"{', '.join(unlinked)}, which are not linked to the process."
            )

    # Computing the total from the flows ---------------------------------------------------------

    def total_method(self) -> TotalMethod | None:
        """How the total of this side can be computed from its flows.

        None if it cannot. Otherwise, one of:

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

        The total has the dimensions common to all flows used.

        Raises an UnderdeterminedError if the total cannot be computed, also if it is not
        determined for some items, since the shares used are zero there.
        """
        method = self.total_method()
        if method == "sum":
            return sum_arrays(self.flows.values())
        if method == "share":
            total = self._total_from_share()
            if total is not None:
                return total
            if self._remainder_possible():
                return self._total_from_remainder()
            raise UnderdeterminedError(
                self.process,
                f"From the {self.label}s {', '.join(self.both_known)} and their shares, the total "
                "is not determined for some items, since the shares are zero there.\n",
                concerns_flows=True,
            )
        if method == "remainder":
            return self._total_from_remainder()
        raise UnderdeterminedError(self.process, self.total_failure_message(), concerns_flows=True)

    def _total_from_share(self) -> Optional["FlodymArray"]:
        """Total as a known flow divided by its share.

        None if all shares of known flows have zeros, so that the total is not determined
        everywhere.
        """
        usable = [n for n in self.both_known if not self.shares[n].dims - self.flows[n].dims]
        if not usable:
            name = self.both_known[0]
            raise ValueError(
                f"In process '{self.process.name}': The share of flow '{name}' has the "
                f"dimensions {(self.shares[name].dims - self.flows[name].dims).names}, which the "
                "flow does not have."
            )
        for name in usable:
            if not np.any(self.shares[name].values == 0):
                return self.flows[name] / self.shares[name]
        return None

    def _total_from_remainder(self) -> "FlodymArray":
        """Total as ``sum(known flows) / (1 - sum(shares of unknown flows))``."""
        known = self.known
        sum_known = sum_arrays(known.values())
        shares_unknown = {name: self.shares[name] for name in self.unknown}
        dims_unknown = union_of_dims(shares_unknown.values())
        excess_dims = dims_unknown - sum_known.dims
        if excess_dims:
            share_names = [n for n, s in shares_unknown.items() if s.dims - sum_known.dims]
            raise ValueError(
                f"In process '{self.process.name}': Cannot compute the total from the known "
                f"{self.label}s {', '.join(known)} and the shares of the unknown flows "
                f"{', '.join(share_names)}, since these shares have the dimensions "
                f"{excess_dims.names}, which not all known flows have."
            )
        remaining = remaining_share(shares_unknown.values())
        if np.any(remaining.values == 0.0):
            raise UnderdeterminedError(
                self.process,
                f"From the known {self.label}s {', '.join(known)}, the total is not determined "
                "for some items, since the shares of the unknown flows sum to one there.\n",
                concerns_flows=True,
            )
        return sum_known / remaining

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
        """Whether the unknown flows can be computed from a total.

        This is the case if at most one unknown flow has no share.
        """
        return len(self.neither_known) <= 1

    def flows_failure_message(self) -> str:
        """Explanation why the unknown flows cannot be computed from a total."""
        return (
            f"Cannot compute the {self.label}s from the total, since for more than one of them, "
            f"neither share nor value is given: {', '.join(self.neither_known)}. Give a share or "
            "value for all but one of them.\n"
        )

    def check_share_dims(self, total_dims: DimensionSet, names: Iterable[str]) -> None:
        """Check that the shares of the given flows have no dimensions which the total lacks."""
        for name in names:
            excess_dims = self.shares[name].dims - total_dims
            if excess_dims:
                raise ValueError(
                    f"In process '{self.process.name}': The share of flow '{name}' has the "
                    f"dimensions {excess_dims.names}, which the total of the {self.label}s does "
                    "not have. To add these dimensions to the flow, use a dimension splitter, or "
                    "insert a separate process with a dimension splitter."
                )

    def splitter_use(self, available_dims: DimensionSet | None, lacking: str) -> SplitterUse:
        """How the dimension splitter of this side is used to compute its unknown flows.

        The splitter is needed if unknown flows have dimensions which are not available
        otherwise.

        Args:
            available_dims: Dimensions available without splitter, e.g. those of the total.
                None if no splitter can be applied, since there is no total to apply it to.
            lacking: Who lacks the dimensions which are not available, for messages, e.g.
                ``"their total does not have"``.

        Raises:
            ValueError: If the splitter is needed, but does not have all missing dimensions.
        """
        splitter = self.process._linked_dimension_splitter(self.direction)
        unknown = list(self.unknown.values())
        if available_dims is None or not union_of_dims(unknown) - available_dims:
            return SplitterUse(unused=splitter is not None and bool(unknown))
        missing_dims = union_of_dims(unknown) - available_dims
        if splitter is None:
            return SplitterUse(
                missing_message=(
                    f"Unknown {self.label}s have the dimensions {missing_dims.names}, which "
                    f"{lacking}. Set a {self.direction} dimension splitter to add them."
                )
            )
        if missing_dims - splitter.dims:
            raise ValueError(
                f"In process '{self.process.name}': The {self.direction} dimension splitter does "
                f"not have the dimensions {(missing_dims - splitter.dims).names}, which unknown "
                f"{self.label}s have, but {lacking}."
            )
        return SplitterUse(splitter=splitter)

    def check_flow_dims(self, total_dims: DimensionSet) -> None:
        """Check that the unknown flows can be computed from a total with the given dimensions."""
        self.check_share_dims(total_dims, [name for name in self.unknown if name in self.shares])
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
                f"total minus the other {self.label}s, since the total or a known "
                f"{self.label} does not have its dimensions {missing_dims.names}."
            )

    def compute_flows(self, total: "FlodymArray") -> None:
        """Compute the unknown flows from the total.

        Flows with a share are computed as total times share, the one flow without share (if any)
        as total minus all other flows.
        """
        self.check_flow_dims(total.dims)
        unknown = self.unknown
        # Computed flows are kept with the dimensions of the total until the remaining flow is
        # computed: Reducing them to the flow dimensions first could lose dimensions it needs.
        computed = {name: total * self.shares[name] for name in unknown if name in self.shares}
        remaining = [name for name in unknown if name not in computed]
        if remaining:
            others = list(self.known.values()) + list(computed.values())
            unknown[remaining[0]][...] = total - sum_arrays(others) if others else total
        for name, values in computed.items():
            unknown[name][...] = values

    # Checks of the shares -----------------------------------------------------------------------

    def check_share_sums(self) -> None:
        """Check that the shares are between zero and one and sum to at most one.

        If all flows of the side have a share, they must sum to exactly one.
        """
        if not self.shares or config.error_behaviors.process_shares == ErrorBehavior.IGNORE:
            return
        for name, share in self.shares.items():
            check_share_values(self.process, share, f"Share of flow '{name}'")
        share_sum = sum_of_shares(self.shares.values()).values
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
        """Check that the flows fulfill their shares of the given total of this side.

        This may not be the case if the process is overdetermined. Shares with dimensions the
        total does not have are not checked.
        """
        if config.error_behaviors.process_overdetermined == ErrorBehavior.IGNORE:
            return
        tolerance = self.process.tolerance
        failures: list[str] = []
        for name, share in self.shares.items():
            if share.dims - total.dims:
                continue
            max_error = max_abs(total * share - self.flows[name])
            if not max_error <= tolerance:
                failures.append(f"'{name}' (max error: {max_error})")
        if failures:
            handle_error(
                behavior=config.error_behaviors.process_overdetermined,
                message=(
                    f"In process '{self.process.name}': The shares of the {self.label}s "
                    f"{', '.join(failures)} are not fulfilled. This happens if the process is "
                    "overdetermined, i.e. more flow values or shares are given than needed, and "
                    "these are inconsistent."
                    + config_hint("process_overdetermined", tolerance=True)
                ),
            )


class ProcessComputation:
    """One computation of a process, see :py:meth:`flodym.Process.compute`.

    Processes without stock are computed by :py:meth:`run`. For processes with stock,
    :py:meth:`run` calls ``Stock._compute_process``, which uses the building blocks
    :py:meth:`require_total`, :py:meth:`require_flows`, :py:meth:`compute_total` and
    :py:meth:`compute_side`.
    """

    def __init__(self, process: "Process") -> None:
        process._check_parameters_linked()
        self.process = process
        self.sides: dict[Side, ProcessSide] = {side: ProcessSide(process, side) for side in SIDES}
        self.unused_splitters: list[Direction] = []
        """Directions of dimension splitters which were not applied when computing the flows of
        their side, since no dimensions had to be added."""

    # Checks before the computation --------------------------------------------------------------

    def require_total(self, sides: Sequence[Side] = SIDES) -> list[Side]:
        """Sides from which the total can be computed.

        They are sorted by the preference of their method (see :py:data:`TOTAL_METHODS`).
        Raises an UnderdeterminedError if there is none.
        """
        candidates: list[tuple[int, Side]] = []
        for side in sides:
            method = self.sides[side].total_method()
            if method is not None:
                candidates.append((TOTAL_METHODS.index(method), side))
        if not candidates:
            message = "Cannot compute the total of the process.\n"
            message += "".join(self.sides[side].total_failure_message() for side in sides)
            raise UnderdeterminedError(self.process, message, concerns_flows=True)
        return [side for _, side in sorted(candidates)]

    def require_flows(self, sides: Sequence[Side] = SIDES) -> None:
        """Raise an UnderdeterminedError if unknown flows cannot be computed from a total.

        Only the unknown flows of the given sides are considered.
        """
        failed = [self.sides[side] for side in sides if not self.sides[side].can_compute_flows()]
        if failed:
            message = "".join(side.flows_failure_message() for side in failed)
            raise UnderdeterminedError(self.process, message, concerns_flows=True)

    # Computation --------------------------------------------------------------------------------

    def run(self) -> None:
        """Compute the process, with its stock, if any."""
        for side in self.sides.values():
            side.check_shares_linked()
        stock = self.process.stock
        if stock is not None:
            stock._compute_process(self)
            return
        # Process without stock: inflows -> total -> outflows
        self.require_flows()
        total = self.compute_total(SIDES)
        for side in SIDES:
            self.compute_side(total, side)

    def compute_total(self, sides: Sequence[Side]) -> "FlodymArray":
        """Total, computed from the flows of one of the given sides.

        The first of the given sides from which this is possible is used (see
        :py:meth:`require_total`). Raises an UnderdeterminedError if none.
        """
        errors: list[str] = []
        for side in self.require_total(sides):
            try:
                return self.sides[side].compute_total()
            except UnderdeterminedError as error:
                errors.append(error.reason)
        raise UnderdeterminedError(self.process, "".join(errors), concerns_flows=True)

    def compute_side(self, total: "FlodymArray", side: Side) -> None:
        """Compute the unknown flows of one side from the total of this side.

        The total is extended by the dimension splitter of the side if needed. Afterwards, the
        shares are checked.
        """
        total = self.apply_dimension_splitter(total, side)
        self.sides[side].compute_flows(total)
        self.sides[side].check_shares_fulfilled(total)

    def apply_dimension_splitter(self, total: "FlodymArray", side: Side) -> "FlodymArray":
        """Multiply the total with the dimension splitter of the given side, if needed.

        The splitter is the forward splitter for the outflows and the backward splitter for the
        inflows. It is needed if unknown flows of the side have dimensions which the total does not
        have. Otherwise, the total is returned unchanged.
        """
        process_side = self.sides[side]
        use = process_side.splitter_use(total.dims, "their total does not have")
        if use.missing_message is not None:
            raise ValueError(f"In process '{self.process.name}': {use.missing_message}")
        if use.unused:
            self.unused_splitters.append(process_side.direction)
        if use.splitter is None:
            return total
        check_splitter_sum(self.process, use.splitter, total.dims, process_side.direction)
        return total * use.splitter

    # Checks after the computation ---------------------------------------------------------------

    def check_result(self, was_overdetermined: bool, snapshot: "Snapshot") -> None:
        """Checks after a successful computation, see :py:func:`check_after_computation`."""
        check_after_computation(self.process, snapshot, was_overdetermined, self.unused_splitters)


def check_after_computation(
    process: "Process",
    snapshot: "Snapshot",
    was_overdetermined: bool,
    unused_splitters: Sequence[Direction],
) -> None:
    """Checks after a successful computation of a process, on its own or with others.

    The process must be computed. Its shares must sum to one (or at most one), unused dimension
    splitters are reported, stock arrays which were set before must keep their values, and the
    mass balance is checked if the process was overdetermined or
    ``config.checks.mass_balance_after_compute`` is set.

    Args:
        process: The process.
        snapshot: The state of the process before the computation.
        was_overdetermined: Whether more values were given than needed.
        unused_splitters: Directions of dimension splitters which are set, but were not needed.
    """
    if not process.is_computed:
        raise RuntimeError(
            f"After computation of process '{process.name}', there are still unknown flows or "
            "stock arrays. This indicates an internal flodym error."
        )
    sides = {side: ProcessSide(process, side) for side in SIDES}
    for process_side in sides.values():
        process_side.check_share_sums()
    for direction in unused_splitters:
        label = sides[SPLITTER_SIDES[direction]].label
        handle_error(
            behavior=config.error_behaviors.unused_dimension_splitter,
            message=(
                f"In process '{process.name}': The {direction} dimension splitter is set, but "
                f"was not used to compute the {label}s, since no dimensions had to be added. "
                "The split it gives may not be fulfilled."
                + config_hint("unused_dimension_splitter")
            ),
        )
    check_given_stock_arrays_kept(process, snapshot)
    if was_overdetermined or config.checks.mass_balance_after_compute:
        check_process_mass_balance(process, overdetermined=was_overdetermined)


def check_given_stock_arrays_kept(process: "Process", snapshot: "Snapshot") -> None:
    """Check that stock arrays which were set before the computation kept their values.

    The stock computation overwrites them if the process is overdetermined.
    """
    stock = process.stock
    if stock is None or config.error_behaviors.process_overdetermined == ErrorBehavior.IGNORE:
        return
    tolerance = stock.tolerance
    changed = [
        array.name
        for array, values in snapshot.given_stock_arrays
        if not max_abs(array.values - values) <= tolerance
    ]
    if changed:
        handle_error(
            behavior=config.error_behaviors.process_overdetermined,
            message=(
                f"In process '{process.name}': The computation changed the values of the "
                f"stock arrays {', '.join(changed)}, which were set before. This happens if "
                "the process is overdetermined, i.e. more values are given than needed, and "
                "these are inconsistent." + config_hint("process_overdetermined", tolerance=True)
            ),
        )


def check_process_mass_balance(process: "Process", overdetermined: bool = False) -> None:
    """Check the mass balance of a process after a computation.

    Failures are handled according to ``config.error_behaviors.mass_balance``.
    """
    hint = config_hint("mass_balance", tolerance=True)
    if overdetermined:
        hint = " The process is overdetermined, so the given values may be inconsistent." + hint
    report_balance_failures(
        [process._mass_balance_failure()],
        config.error_behaviors.mass_balance,
        hint=hint,
        success_message=f"Mass balance of process '{process.name}' is consistent.",
    )


def check_completed_process(process: "Process") -> None:
    """Check a process whose flows were all set by the computation of neighboring processes.

    The process is not computed. Its mass balance and its shares, relative to the sum of the
    flows of each side, must be fulfilled.
    """
    check_process_mass_balance(process, overdetermined=True)
    for side in SIDES:
        process_side = ProcessSide(process, side)
        if process_side.shares and process_side.flows:
            process_side.check_shares_fulfilled(sum_arrays(process_side.flows.values()))


def is_overdetermined(process: "Process") -> bool:
    """See :py:attr:`flodym.Process.is_overdetermined`."""
    process._check_parameters_linked()
    sides = [ProcessSide(process, side) for side in SIDES]
    n_flows = sum(len(side.flows) for side in sides)
    n_given = sum(len(side.known) + len(side.shares) - side.n_redundant_shares for side in sides)
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
        n_equations += sum(a.is_set for a in stock_arrays(stock))
    return n_equations > n_unknowns


def stock_arrays(stock: "Stock") -> list["FlodymArray"]:
    """Inflow, outflow, stock and, if present, inflow by cohort of a stock."""
    arrays = [stock.inflow, stock.outflow, stock.stock]
    inflow_by_cohort = getattr(stock, "inflow_by_cohort", None)
    if inflow_by_cohort is not None:
        arrays.append(inflow_by_cohort)
    return arrays


class Snapshot:
    """Values and ``is_set`` flags of all arrays which a computation of a process may change.

    These are its in- and outflows and the arrays of its stock. Used to restore them if the
    computation fails.
    """

    _stock_private_arrays = ("_stock_by_cohort", "_outflow_by_cohort")

    def __init__(self, process: "Process") -> None:
        stock = process.stock
        self.stock = stock
        arrays_of_stock: list[FlodymArray] = [] if stock is None else stock_arrays(stock)
        flows = list(process.inflows.values()) + list(process.outflows.values())
        self.arrays: list[tuple[FlodymArray, np.ndarray, bool]] = [
            (a, a.values.copy(), a.is_set) for a in flows + arrays_of_stock
        ]
        """Arrays with their values and ``is_set`` flags."""
        self.given_stock_arrays: list[tuple[FlodymArray, np.ndarray]] = [
            (a, v) for a, v, was_set in self.arrays[len(flows) :] if was_set
        ]
        """Stock arrays which were set, with their values."""
        # cohort arrays of dynamic stock models are replaced, not modified, so references suffice
        self.stock_private: dict[str, np.ndarray] = {
            name: getattr(stock, name)
            for name in self._stock_private_arrays
            if hasattr(stock, name)
        }

    def restore(self) -> None:
        """Restore the stored values and ``is_set`` flags."""
        for array, values, was_set in self.arrays:
            array.values = values
            if was_set:
                array.mark_set()
            else:
                array.mark_unset()
        for name, value in self.stock_private.items():
            setattr(self.stock, name, value)


def compute_process(
    process: "Process", on_underdetermined: ErrorBehavior | str | None, short_message: bool
) -> bool:
    """Compute a single process, see :py:meth:`flodym.Process.compute`.

    If the computation or one of the checks after it raises an error, all arrays of the process
    and its stock are restored. Afterwards, neighboring processes whose flows were all set by this
    computation are checked (see :py:func:`check_completed_process`).

    Args:
        process: The process.
        on_underdetermined: What to do if the process is underdetermined. If None,
            ``config.error_behaviors.process_underdetermined`` is used.
        short_message: Whether to report underdetermination in one line. Otherwise, the
            message explains why, and points to the simultaneous computation of processes if
            the flows are underdetermined and the process lies on a cycle of unknown flows.

    Returns:
        Whether the process is computed afterwards.
    """
    if process.id == 0:
        logger.debug(f"Process '{process.name}' is the system boundary and is not computed.")
        return False
    if process.is_computed:
        logger.debug(f"Process '{process.name}' is already computed.")
        return True
    if on_underdetermined is None:
        on_underdetermined = config.error_behaviors.process_underdetermined
    on_underdetermined = ErrorBehavior(on_underdetermined)

    incomplete_neighbors = [
        p for p in process._neighbors() if p.id != 0 and p is not process and not p.is_computed
    ]
    was_overdetermined = is_overdetermined(process)
    snapshot = Snapshot(process)
    computation = ProcessComputation(process)
    try:
        computation.run()
        computation.check_result(was_overdetermined, snapshot)
    except UnderdeterminedError as error:
        snapshot.restore()
        if short_message or not (error._concerns_flows and on_cycle_of_unknown_flows(process)):
            reported = error
        else:
            reported = UnderdeterminedError(
                process, f"{error.reason.rstrip()}\n{DEPENDING_PROCESSES_HINT}", concerns_flows=True
            )
        if on_underdetermined == ErrorBehavior.ERROR:
            if reported is error:
                raise
            raise reported from error
        handle_error(
            behavior=on_underdetermined,
            message=reported.short_message if short_message else str(reported),
        )
        return False
    except BaseException:
        snapshot.restore()
        raise

    for neighbor in incomplete_neighbors:
        if neighbor.is_computed:
            check_completed_process(neighbor)
    return True
