"""Equations of processes, for computing processes which depend on each other simultaneously.

Each process contributes the rules of :py:meth:`flodym.Process.compute` as linear equations
between its flows, its total(s) and the arrays of its stock (see
:py:func:`flodym.compute_processes`). :py:class:`ProcessEquations` writes them symbolically as
:py:class:`Equation` objects, and adds them to a :py:class:`flodym._linear_system.LinearSystem`.
:py:func:`assemble_processes` assembles the system of several processes.
"""

import itertools
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ._linear_system import AssembledSystem, LinearSystem, Solution
from ._process_compute import (
    OTHER_SIDE,
    SIDES,
    Direction,
    ProcessSide,
    Side,
    common_dims,
    remaining_share,
    stock_arrays,
    unique_flows,
)
from .dimensions import DimensionSet
from .flodym_arrays import FlodymArray

if TYPE_CHECKING:
    from .flodym_arrays import Flow
    from .processes import Process
    from .stocks import Stock

TOTAL = "total"
"""Key of the total throughput of a process without stock. A process with stock has one total
per side, keyed by the side."""


def total_key(process: "Process", key: str) -> tuple[int, str]:
    """Key of a total of a process among the unknowns of a linear system."""
    return (process.id, key)


@dataclass
class Term:
    """One term ``sign * ⟨c_1 * ... * c_n * x⟩`` of an :py:class:`Equation`.

    ``x`` is either an array (a flow or a stock array: an unknown if it is not set, otherwise a
    constant) or a total of the process, given by its key. The term is summed over the
    dimensions which the equation does not have, see
    :py:meth:`flodym._linear_system.LinearSystem.add_term`.

    Example: for a flow ``f`` with the share ``s`` of the total, the equation ``f - s * T = 0``
    has the terms ``Term(array=f)`` and ``Term(total=TOTAL, coefficients=[s], sign=-1.0)``.
    """

    array: FlodymArray | None = None
    """The array, if the term is not about a total."""
    total: str | None = None
    """Key of the total, if the term is about a total."""
    coefficients: list[FlodymArray] = field(default_factory=list)
    sign: float = 1.0


@dataclass(eq=False)
class Equation:
    """A block of equations of a process, with one equation per item of its dimensions."""

    process: "Process"
    label: str
    """Description, e.g. ``"share of the outflow 'a => b'"``."""
    dims: DimensionSet
    terms: list[Term]

    def describe(self) -> str:
        """Description of the equation, with its process, for messages."""
        return f"{self.label} of process '{self.process.name}'"


class ProcessEquations:
    """The equations of a process, as described in :py:func:`flodym.compute_processes`.

    Flows and stock arrays which are not set are unknowns. If the process has a stock, it must be
    prepared before (see ``Stock._prepare_block_computation``).

    Args:
        process: The process.

    Raises:
        ValueError: If the process cannot be described by the equations, e.g. if both of its
            dimension splitters would be needed.
    """

    def __init__(self, process: "Process") -> None:
        self.process = process
        self.sides = {side: ProcessSide(process, side) for side in SIDES}
        self.unknown_flows: list[Flow] = [flow for flow in unique_flows(process) if not flow.is_set]
        """Flows of the process which are unknowns."""
        self.totals: dict[str, DimensionSet] = {}
        """Dimensions of the totals of the process, by key (:py:data:`TOTAL` or a side)."""
        self.side_totals: dict[Side, str] = {}
        """Key of the total that each side with flows refers to."""
        self.splitters: dict[Side, FlodymArray] = {}
        """Dimension splitters which are applied to the total of their side."""
        self.unused_splitters: list[Direction] = []
        """Directions of dimension splitters which are set, but not needed."""
        self.equations: list[Equation] = []
        self.hints: list[str] = []
        """Explanations which apply if the process is not determined."""
        stock = process.stock
        self.flow_driven_stock = stock is not None and not stock.is_computed
        """Whether the process has a stock whose inflow and outflow are independent unknowns."""

        process._check_parameters_linked()
        for process_side in self.sides.values():
            process_side.check_shares_linked()
        self._check_unrelated_dims()
        if stock is None:
            self._write_without_stock()
        else:
            self._write_with_stock(stock)

    def add_unknowns(self, system: LinearSystem) -> None:
        """Add the totals and the unknown flows of the process to a linear system."""
        for key, dims in self.totals.items():
            system.add_unknown(total_key(self.process, key), dims)
        for flow in self.unknown_flows:
            system.add_unknown(flow.name, flow.dims)

    def add_equations(self, system: LinearSystem) -> None:
        """Add the equations of the process to a linear system with its unknowns.

        The key of each block of equations is its :py:class:`Equation`.
        """
        for equation in self.equations:
            block = system.add_equation(equation, equation.dims)
            for term in equation.terms:
                if term.array is None:
                    factor = system.unknowns[total_key(self.process, str(term.total))]
                elif term.array.is_set:
                    factor = term.array
                else:
                    factor = system.unknowns[term.array.name]
                system.add_term(block, factor, term.coefficients, term.sign)

    def total(self, key: str, solution: Solution) -> FlodymArray:
        """A total of the process, from the solution of a linear system."""
        return FlodymArray(
            dims=self.totals[key],
            values=solution.array_values(total_key(self.process, key)),
            name=f"total of process '{self.process.name}'",
            extensive=True,
        )

    def side_total(self, side: Side, solution: Solution) -> FlodymArray:
        """Total of a side as seen by its flows, i.e. with its dimension splitter applied."""
        total = self.total(self.side_totals[side], solution)
        splitter = self.splitters.get(side)
        return total if splitter is None else total * splitter

    def is_complete(self, solution: Solution) -> bool:
        """Whether all unknowns of the process are determined: its totals and its flows."""
        keys: list[Hashable] = [total_key(self.process, key) for key in self.totals]
        keys += [flow.name for flow in self.unknown_flows]
        return all(solution.is_determined(key) for key in keys)

    def failure_hints(self) -> list[str]:
        """Explanations why the process may not be determined."""
        hints = list(self.hints)
        for process_side in self.sides.values():
            if not process_side.can_compute_flows():
                hints.append(process_side.flows_failure_message())
        stock = self.process.stock
        if stock is not None and self.flow_driven_stock:
            hints.append(
                f"The inflow and outflow of the flow-driven stock '{stock.name}' do not determine "
                "each other. Give a value for its outflow (e.g. with prescribed_outflow of the "
                "StockDefinition), or for the outflows of the process."
            )
        return [hint.strip() for hint in hints]

    def arrays(self) -> list[FlodymArray]:
        """The flows, shares, dimension splitters and stock arrays of the process."""
        arrays: list[FlodymArray] = []
        for process_side in self.sides.values():
            arrays += [*process_side.flows.values(), *process_side.shares.values()]
            splitter = self.process._linked_dimension_splitter(process_side.direction)
            if splitter is not None:
                arrays.append(splitter)
        if self.process.stock is not None:
            arrays += stock_arrays(self.process.stock)
        return arrays

    def not_finite_arrays(self) -> list[FlodymArray]:
        """The given arrays of the process with values which are not finite.

        Given are the shares, dimension splitters and the flows and stock arrays which are set.
        """
        flows_and_stock = {id(flow) for flow in unique_flows(self.process)}
        if self.process.stock is not None:
            flows_and_stock |= {id(array) for array in stock_arrays(self.process.stock)}
        return [
            array
            for array in self.arrays()
            if (array.is_set or id(array) not in flows_and_stock)
            and not np.all(np.isfinite(array.values))
        ]

    def _check_unrelated_dims(self) -> None:
        """Raise an error if the arrays of the process have dimensions related to each other.

        Dimensions are related if they have a common ancestor (see
        :py:attr:`flodym.Dimension.parent`), also if one is an ancestor of the other.
        """
        dims = {dim.letter: dim for array in self.arrays() for dim in array.dims}
        for dim, other in itertools.combinations(dims.values(), 2):
            if dim.common_ancestor(other) is not None:
                raise ValueError(
                    f"In process '{self.process.name}': The dimensions '{dim.name}' and "
                    f"'{other.name}' are related (see Dimension.parent), which the simultaneous "
                    "computation of processes does not support yet."
                )

    def _write_without_stock(self) -> None:
        """Equations of a process without stock, which has one total throughput T.

        With C_in and C_out the dimensions which all inflows and all outflows have, the
        dimensions of T are ``(C_in - A_in) | (C_out - A_out)``, where A_side are the dimensions
        which the dimension splitter of a side adds, if it is needed. It is needed if unknown
        flows of the side have dimensions which not all flows of the other side have. Both sides
        then refer to T (see :py:meth:`_write_side`), which couples inflows and outflows.
        """
        common = {side: common_dims(self.sides[side].flows.values()) for side in SIDES}
        added: dict[Side, DimensionSet] = {}
        for side in SIDES:
            other = OTHER_SIDE[side]
            splitter = self._use_splitter(side, common[other], f"not all {other}flows have")
            other_dims = common[other]
            if splitter is not None and other_dims is not None:
                added[side] = splitter.dims - other_dims
        if len(self.splitters) == len(SIDES):
            raise ValueError(
                f"In process '{self.process.name}': Both dimension splitters would be needed, "
                "since unknown inflows and unknown outflows each have dimensions which the flows "
                "of the other side do not all have. The simultaneous computation of processes "
                "does not support this yet."
            )
        total_dims = DimensionSet.empty()
        for side in SIDES:
            side_dims = common[side]
            if side_dims is not None:
                total_dims |= side_dims - added.get(side, DimensionSet.empty())
        self.totals[TOTAL] = total_dims
        for side in SIDES:
            if self.sides[side].flows:
                self._write_side(side, TOTAL)

    def _write_with_stock(self, stock: "Stock") -> None:
        """Equations of a process with stock, which has a total for each side.

        The total of a side has the dimensions which all its flows have, except for those which
        its dimension splitter adds. The splitter is needed if unknown flows of the side have
        dimensions which the stock does not have. The total, summed to the dimensions of the
        stock, equals the stock inflow (or outflow) if that is known. A stock whose inflow or
        outflow is unknown here is flow-driven: they follow from the totals after the solution.
        """
        for side in SIDES:
            process_side = self.sides[side]
            if not process_side.flows:
                continue
            splitter = self._use_splitter(side, stock.dims, "the stock does not have")
            total_dims = common_dims(process_side.flows.values()) or DimensionSet.empty()
            if splitter is not None:
                total_dims = total_dims - (splitter.dims - stock.dims)
            stock._check_total_dims(total_dims, process_side.label)
            self.totals[side] = total_dims
            self._write_side(side, side)
            array = stock.inflow if side == "in" else stock.outflow
            if array.is_set:
                self.equations.append(
                    Equation(
                        process=self.process,
                        label=f"stock {process_side.label}",
                        dims=stock.dims,
                        terms=[Term(total=side), Term(array=array, sign=-1.0)],
                    )
                )

    def _use_splitter(
        self, side: Side, available_dims: DimensionSet | None, lacking: str
    ) -> FlodymArray | None:
        """The dimension splitter of a side, if it is needed, see ProcessSide.splitter_use.

        Unused splitters and hints for missing ones are recorded.
        """
        process_side = self.sides[side]
        use = process_side.splitter_use(available_dims, lacking)
        if use.unused:
            self.unused_splitters.append(process_side.direction)
        if use.missing_message is not None:
            self.hints.append(use.missing_message)
        if use.splitter is not None:
            self.splitters[side] = use.splitter
        return use.splitter

    def _write_side(self, side: Side, key: str) -> None:
        """Equations of the flows of one side, relative to the total of this side.

        With E the total, multiplied by the dimension splitter of the side if it is needed, the
        equations are, as in :py:meth:`flodym.Process.compute`:

        - share: ``f - s * E = 0`` for each flow f with a share s,
        - remainder: ``r - (1 - sum of the shares) * E = 0`` if exactly one flow r has no share.
          Without other shares, the factor is one, i.e. ``r = E``.
        - sum: ``sum of the flows - E = 0`` if two or more flows have no share.

        Each equation has the dimensions which both its flows and E have; the terms are summed
        over the others.
        """
        process_side = self.sides[side]
        flows, shares = process_side.flows, process_side.shares
        self.side_totals[side] = key
        splitter = self.splitters.get(side)
        side_total_dims = self.totals[key]
        splitter_factors = []
        if splitter is not None:
            side_total_dims = side_total_dims | splitter.dims
            splitter_factors = [splitter]
        process_side.check_share_dims(side_total_dims, shares)
        # factor of E for each flow; None stands for a factor of one
        factors: dict[str, FlodymArray | None] = dict(shares)
        without_share = [name for name in flows if name not in shares]
        if len(without_share) == 1:
            factors[without_share[0]] = remaining_share(shares.values()) if shares else None
        for name, factor in factors.items():
            kind = "share" if name in shares else "remainder"
            coefficients = ([] if factor is None else [factor]) + splitter_factors
            self.equations.append(
                Equation(
                    process=self.process,
                    label=f"{kind} of the {process_side.label} '{name}'",
                    dims=flows[name].dims & side_total_dims,
                    terms=[
                        Term(array=flows[name]),
                        Term(total=key, coefficients=coefficients, sign=-1.0),
                    ],
                )
            )
        if len(without_share) >= 2:
            side_dims = common_dims(flows.values()) or DimensionSet.empty()
            self.equations.append(
                Equation(
                    process=self.process,
                    label=f"sum of the {process_side.label}s",
                    dims=side_dims & side_total_dims,
                    terms=[Term(array=flow) for flow in flows.values()]
                    + [Term(total=key, coefficients=splitter_factors, sign=-1.0)],
                )
            )


class TooManyUnknownsError(Exception):
    """Raised by :py:func:`assemble_processes` if the system would have too many unknowns.

    Args:
        n_unknowns: Number of unknowns of the system.
    """

    def __init__(self, n_unknowns: int) -> None:
        super().__init__(f"The system would have {n_unknowns} unknowns.")
        self.n_unknowns = n_unknowns


def assemble_processes(
    all_equations: Sequence[ProcessEquations], max_unknowns: int | None = None
) -> AssembledSystem:
    """Assemble the equations of processes into one linear system.

    Args:
        all_equations: The equations of the processes.
        max_unknowns: If given, the maximum number of unknowns.

    Raises:
        TooManyUnknownsError: If the system has more unknowns than ``max_unknowns``. The check
            comes before the equations are added.
    """
    system = LinearSystem()
    for equations in all_equations:
        equations.add_unknowns(system)
    if max_unknowns is not None and system.n_unknowns > max_unknowns:
        raise TooManyUnknownsError(system.n_unknowns)
    for equations in all_equations:
        equations.add_equations(system)
    return system.assemble()
