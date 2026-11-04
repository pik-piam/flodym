import logging
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, PrivateAttr, model_validator

from ._checks import balance_failure, config_hint, max_abs, report_balance_failures
from .config import ErrorBehavior, config, handle_error
from .dimensions import DimensionSet
from .mfa_definition import ProcessDefinition

if TYPE_CHECKING:
    from .flodym_arrays import FlodymArray, Flow, Parameter
    from .stocks import Stock

Side = Literal["in", "out"]
"""Side of a process: ``"in"`` for its inflows, ``"out"`` for its outflows."""
SIDES: tuple[Side, Side] = ("in", "out")


class UnderdeterminedError(Exception):
    """Raised by :py:meth:`Process.compute` if the given flows, shares and stock values are not
    sufficient to compute the process."""

    def __init__(self, process: "Process", message: str):
        self.process_name = process.name
        self.short_message = f"Process '{process.name}' is underdetermined. Skip computation."
        super().__init__(
            f"Cannot compute process '{process.name}' (ID {process.id}), as it is "
            f"underdetermined.\n{message}" + config_hint("process_underdetermined")
        )


class Process(PydanticBaseModel):
    """Processes serve as nodes for the MFA system layout definition.
    Flows are defined between two processes. Stocks are connected to a process.
    Processes do not contain values themselves.

    Processes get an ID by the order they are defined in the :py:attribute::`MFASystem.definition`,
    or explicitly by a :py:class:`flodym.ProcessDefinition`.
    The process with ID 0 necessarily contains everything outside the system boundary.
    It has to be named 'sysenv'.

    When an :py:class:`flodym.MFASystem` is set up, the process is linked to its inflows,
    outflows and stock (see :py:attr:`inflows`, :py:attr:`outflows`, :py:attr:`stock`).

    Optionally, shares of in- and outflows and a dimension splitter can be given (see
    :py:meth:`set_inflow_share`, :py:meth:`set_outflow_share` and
    :py:meth:`set_dimension_splitter`, or :py:class:`flodym.ProcessDefinition`), which allow to
    compute unknown flows of the process with :py:meth:`compute`.
    """

    name: str
    """Name of the process."""
    id: int
    """ID of the process."""
    _inflows: Dict[str, "Flow"] = PrivateAttr(default_factory=dict)
    _outflows: Dict[str, "Flow"] = PrivateAttr(default_factory=dict)
    _stock: Optional["Stock"] = PrivateAttr(default=None)
    _inflow_shares: Dict[str, "FlodymArray"] = PrivateAttr(default_factory=dict)
    _outflow_shares: Dict[str, "FlodymArray"] = PrivateAttr(default_factory=dict)
    _dimension_splitter: Optional["FlodymArray"] = PrivateAttr(default=None)
    _splitter_used: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def check_id0(self):
        """Ensure that the process with ID 0 is named 'sysenv'."""
        if self.id == 0 and self.name != "sysenv":
            raise ValueError(
                "The process with ID 0 must be named 'sysenv', as it contains everything outside the system boundary."
            )
        return self

    def __repr_args__(self):
        """Arguments shown by ``repr()`` and ``str()``. Linked flows and the stock are shown by
        name only, since they link back to the process."""
        yield "name", self.name
        yield "id", self.id
        yield "inflows", list(self._inflows)
        yield "outflows", list(self._outflows)
        yield "stock", None if self._stock is None else self._stock.name
        if self._inflow_shares:
            yield "inflow_shares", {k: v.name for k, v in self._inflow_shares.items()}
        if self._outflow_shares:
            yield "outflow_shares", {k: v.name for k, v in self._outflow_shares.items()}
        if self._dimension_splitter is not None:
            yield "dimension_splitter", self._dimension_splitter.name

    @property
    def inflows(self) -> Dict[str, "Flow"]:
        """Inflows to the process, keyed by flow name."""
        return self._inflows

    @property
    def outflows(self) -> Dict[str, "Flow"]:
        """Outflows from the process, keyed by flow name."""
        return self._outflows

    @property
    def stock(self) -> Optional["Stock"]:
        """Stock connected to the process, if any."""
        return self._stock

    def add_inflow(self, flow: "Flow") -> None:
        """Link a flow as inflow to the process."""
        if flow.to_process is not self:
            raise ValueError(f"Flow '{flow.name}' does not go to process '{self.name}'.")
        self._inflows[flow.name] = flow

    def add_outflow(self, flow: "Flow") -> None:
        """Link a flow as outflow from the process."""
        if flow.from_process is not self:
            raise ValueError(f"Flow '{flow.name}' does not come from process '{self.name}'.")
        self._outflows[flow.name] = flow

    def set_stock(self, stock: "Stock") -> None:
        """Link a stock to the process. A process can have only one stock."""
        if stock.process is not self:
            raise ValueError(f"Stock '{stock.name}' is not connected to process '{self.name}'.")
        if self._stock is not None and self._stock is not stock:
            raise ValueError(
                f"Process '{self.name}' already has the stock '{self._stock.name}'; "
                f"it cannot also have the stock '{stock.name}'."
            )
        self._stock = stock

    def remove_inflow(self, flow_name: str) -> None:
        """Remove the link to an inflow, given by its name."""
        if flow_name not in self._inflows:
            raise ValueError(f"In process {self.name}: No inflow '{flow_name}' found.")
        del self._inflows[flow_name]

    def remove_outflow(self, flow_name: str) -> None:
        """Remove the link to an outflow, given by its name."""
        if flow_name not in self._outflows:
            raise ValueError(f"In process {self.name}: No outflow '{flow_name}' found.")
        del self._outflows[flow_name]

    def remove_stock(self) -> None:
        """Remove the link to the stock."""
        self._stock = None

    def clear_links(self) -> None:
        """Remove the links to all inflows, outflows and the stock."""
        self._inflows = {}
        self._outflows = {}
        self._stock = None

    def get_sum_inflows(self) -> Union["FlodymArray", float]:
        """Sum of all inflows to the process, reduced to the dimensions common to all inflows.
        Zero if the process has no inflows."""
        return sum(self._inflows.values(), 0.0)

    def get_sum_outflows(self) -> Union["FlodymArray", float]:
        """Sum of all outflows from the process, reduced to the dimensions common to all outflows.
        Zero if the process has no outflows."""
        return sum(self._outflows.values(), 0.0)

    def get_net_inflow(self) -> Union["FlodymArray", float]:
        """Sum of all inflows minus sum of all outflows of the process.
        Unless the process is 'sysenv', this equals the net inflow of its stock (or zero), if the
        mass balance is satisfied."""
        return self.get_sum_inflows() - self.get_sum_outflows()

    def get_mass_balance(
        self, mass_change: Union["FlodymArray", float, None] = None
    ) -> Union["FlodymArray", float]:
        """Mass balance error of the process: The net inflow of the process
        (see :py:meth:`get_net_inflow`) minus its mass change.
        Zero if the mass balance is satisfied.

        Args:
            mass_change: The mass change of the process per year. If None, it is the net inflow of
                the stock of the process, or zero if the process has no stock.
                For the 'sysenv' process, it must be given, as the negative mass change of the
                whole system, see :py:meth:`flodym.MFASystem.check_mass_balance`.
        """
        if mass_change is None:
            if self.id == 0:
                raise ValueError(
                    "The mass balance of the 'sysenv' process depends on the stock changes of the "
                    "whole MFA system and can only be computed if mass_change is given. "
                    "Use MFASystem.check_mass_balance() instead."
                )
            mass_change = 0.0 if self._stock is None else self._stock.net_inflow
        return self.get_net_inflow() - mass_change

    @property
    def _absolute_float_precision(self) -> float:
        """Machine epsilon times the largest absolute value of the in- and outflows of the process
        and of the in- and outflow of its stock."""
        arrays = list(self._inflows.values()) + list(self._outflows.values())
        if self._stock is not None:
            arrays += [self._stock.inflow, self._stock.outflow]
        return max((a._absolute_float_precision for a in arrays), default=0.0)

    @property
    def tolerance(self) -> float:
        """Absolute tolerance for the mass balance check of the process, see
        :py:attr:`flodym.config.Config.relative_tolerance`."""
        return config.get_tolerance(self._absolute_float_precision)

    def check_mass_balance(
        self,
        tolerance: Optional[float] = None,
        error_behavior: Optional[ErrorBehavior | str] = None,
        mass_change: Union["FlodymArray", float, None] = None,
    ) -> bool:
        """Check whether the mass balance of the process (see :py:meth:`get_mass_balance`) is
        satisfied within a tolerance.

        Args:
            tolerance: Absolute tolerance. If None, :py:attr:`tolerance` is used.
            error_behavior: What to do if the check fails. If None,
                ``flodym.config.error_behaviors.mass_balance`` is used (default: raise an error).
            mass_change: See :py:meth:`get_mass_balance`.

        Returns:
            Whether the check passed.
        """
        if error_behavior is None:
            error_behavior = config.error_behaviors.mass_balance
        return report_balance_failures(
            [self._mass_balance_failure(tolerance, mass_change)],
            error_behavior,
            hint=config_hint("mass_balance", tolerance=True),
            success_message=f"Mass balance of process '{self.name}' is consistent.",
        )

    def _mass_balance_failure(
        self,
        tolerance: Optional[float] = None,
        mass_change: Union["FlodymArray", float, None] = None,
    ) -> Optional[str]:
        """Description of the failed mass balance check for an error message, or None if it
        passed. If no tolerance is given, :py:attr:`tolerance` is used, extended by the float
        precision of ``mass_change`` if given."""
        if tolerance is None:
            precision = self._absolute_float_precision
            if mass_change is not None and hasattr(mass_change, "_absolute_float_precision"):
                precision = max(precision, mass_change._absolute_float_precision)
            tolerance = config.get_tolerance(precision)
        balance = self.get_mass_balance(mass_change=mass_change)
        return balance_failure(f"process '{self.name}'", balance, tolerance)


    # ------------------------------------------------------------------------------------------
    # Shares and dimension splitter

    @property
    def inflow_shares(self) -> Dict[str, "FlodymArray"]:
        """Shares of inflows in the total throughput of the process, keyed by flow name.
        Set them with :py:meth:`set_inflow_share`."""
        return dict(self._inflow_shares)

    @property
    def outflow_shares(self) -> Dict[str, "FlodymArray"]:
        """Shares of outflows in the total throughput of the process, keyed by flow name.
        Set them with :py:meth:`set_outflow_share`."""
        return dict(self._outflow_shares)

    @property
    def dimension_splitter(self) -> Optional["FlodymArray"]:
        """Array by which the total throughput of the process is multiplied in
        :py:meth:`compute`, to add dimensions which it does not have, but the flows to compute do.
        Set it with :py:meth:`set_dimension_splitter`."""
        return self._dimension_splitter

    def set_inflow_share(self, flow: str, share: "FlodymArray") -> None:
        """Set the share of an inflow in the total throughput of the process, used by
        :py:meth:`compute`. The array is stored by reference, so later changes of its values are
        taken into account.

        Args:
            flow: Name of the inflow, or name of the process it comes from, if only one flow
                comes from there. The process must be linked to its flows, see
                :py:meth:`flodym.MFASystem.relink`.
            share: The share. Its dimensions must be contained in those of the total throughput
                of the process.
        """
        self._set_share("in", flow, share)

    def set_outflow_share(self, flow: str, share: "FlodymArray") -> None:
        """Set the share of an outflow in the total throughput of the process.
        See :py:meth:`set_inflow_share`."""
        self._set_share("out", flow, share)

    def remove_inflow_share(self, flow: str) -> None:
        """Remove the share of an inflow, given by flow name or name of the process it comes
        from."""
        self._remove_share("in", flow)

    def remove_outflow_share(self, flow: str) -> None:
        """Remove the share of an outflow, given by flow name or name of the process it goes
        to."""
        self._remove_share("out", flow)

    def set_dimension_splitter(self, splitter: Optional["FlodymArray"]) -> None:
        """Set the dimension splitter of the process (see :py:attr:`dimension_splitter`), or
        remove it by passing None. It must sum to one over the dimensions it adds to the total
        throughput of the process."""
        if splitter is not None:
            self._check_share_values(splitter, "Dimension splitter")
        self._dimension_splitter = splitter

    def _flows(self, side: Side) -> Dict[str, "Flow"]:
        if side == "in":
            return self._inflows
        if side == "out":
            return self._outflows
        raise ValueError(f"Side must be 'in' or 'out', not '{side}'.")

    def _shares(self, side: Side) -> Dict[str, "FlodymArray"]:
        if side == "in":
            return self._inflow_shares
        if side == "out":
            return self._outflow_shares
        raise ValueError(f"Side must be 'in' or 'out', not '{side}'.")

    def _resolve_flow_name(self, side: Side, flow: str) -> str:
        """Get the name of an in- or outflow, given by flow name or by the name of the process
        it comes from or goes to."""
        flows = self._flows(side)
        if flow in flows:
            return flow
        matches = [
            name
            for name, f in flows.items()
            if (f.from_process if side == "in" else f.to_process).name == flow
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"In process '{self.name}': Several {side}flows connect to process '{flow}' "
                f"({', '.join(matches)}). Give the flow name instead."
            )
        raise ValueError(
            f"In process '{self.name}': No {side}flow named '{flow}' or connected to a process "
            f"named '{flow}' found. Existing {side}flows: {', '.join(flows) or 'none'}. "
            "Are the flows linked to the process (see MFASystem.relink())?"
        )

    def _set_share(self, side: Side, flow: str, share: "FlodymArray") -> None:
        from .flodym_arrays import FlodymArray

        if not isinstance(share, FlodymArray):
            raise TypeError(f"Share must be a FlodymArray, not {type(share).__name__}.")
        name = self._resolve_flow_name(side, flow)
        self._check_share_values(share, f"Share of flow '{name}'")
        self._shares(side)[name] = share

    def _remove_share(self, side: Side, flow: str) -> None:
        shares = self._shares(side)
        name = flow if flow in shares else self._resolve_flow_name(side, flow)
        if name not in shares:
            raise ValueError(f"In process '{self.name}': No share for {side}flow '{name}' found.")
        del shares[name]

    # ------------------------------------------------------------------------------------------
    # Computation

    @property
    def is_computed(self) -> bool:
        """Whether all in- and outflows of the process and its stock (if any) are set
        (see :py:attr:`flodym.FlodymArray.is_set`)."""
        if self._unknown_flows("in") or self._unknown_flows("out"):
            return False
        return self._stock is None or self._stock.is_computed

    @property
    def is_overdetermined(self) -> bool:
        """Whether more flow values, shares and stock values are given than needed to compute
        the process. In this case, the given values may be inconsistent.

        Unknowns are the in- and outflows and the totals of the in- and outflow side (one shared
        total without stock; with stock, its inflow, outflow and stock). Equations are the sums
        of each side, the given flows, shares and stock arrays, and the stock model equations.
        """
        n_flows = len(self._inflows) + len(self._outflows)
        n_given = len(self._known_flows("in")) + len(self._known_flows("out"))
        n_given += len(self._inflow_shares) + len(self._outflow_shares)
        if self._stock is None:
            n_unknowns = n_flows + 1
            n_equations = 2 + n_given
        else:
            stock_arrays = (self._stock.inflow, self._stock.outflow, self._stock.stock)
            n_unknowns = n_flows + 3
            n_equations = 2 + n_given + self._stock._n_equations
            n_equations += sum(a.is_set for a in stock_arrays)
        return n_equations > n_unknowns

    def compute(
        self,
        on_underdetermined: Optional[ErrorBehavior | str] = None,
        recursive: bool = False,
    ) -> bool:
        """Compute the unknown in- and outflows of the process (and its stock, if any), from the
        known ones, the shares and the dimension splitter.

        Flows are known if they are set (see :py:attr:`flodym.FlodymArray.is_set`). This covers
        many, but not all MFA processes; others have to be computed manually, for example if two
        processes depend on each other and form an equation system.

        The computation assumes the following structure of a process without stock: The inflows
        sum to a total throughput, which may be multiplied with the dimension splitter, and which
        is split into the outflows. It proceeds as follows:

        - The total is computed from the inflows or from the outflows (whichever is possible) as

          - the sum of the flows of that side, if all are known,
          - a known flow divided by its share, if both are given,
          - or as ``sum(known flows) / (1 - sum(shares of unknown flows))``, if the shares of all
            unknown flows of that side are given.

          This means that the shares of a side refer to the total of that side.
        - If unknown flows have dimensions the total does not have, the total is multiplied
          with the dimension splitter, which must sum to one over these dimensions.
        - Each unknown flow with a share is computed as total times share.
          One unknown flow per side may have no share; it is the total minus all other flows.

        Dimensions of flows are thus only added by the dimension splitter. To add dimensions to
        single flows, insert a process with a dimension splitter.

        For a process with stock, the total of the inflows is the inflow of the stock, and the
        total of the outflows is its outflow. See :py:meth:`flodym.Stock.compute_process` for how
        stocks of each type are computed.

        Args:
            on_underdetermined: What to do if the process cannot be computed from the given
                values. If None, ``flodym.config.error_behaviors.process_underdetermined`` is used
                (default: raise an :py:class:`flodym.UnderdeterminedError`).
            recursive: If True, after a successful computation, the neighboring processes are
                computed as well, recursively. Underdetermined neighbors are skipped with an info
                log message. See also :py:meth:`flodym.MFASystem.compute_all_possible`.

        Returns:
            Whether the process is computed afterwards.
        """
        if self.id == 0:
            logging.debug(f"Process '{self.name}' is the system boundary and is not computed.")
            return False
        if self.is_computed:
            logging.debug(f"Process '{self.name}' is already computed.")
            return True
        if on_underdetermined is None:
            on_underdetermined = config.error_behaviors.process_underdetermined
        on_underdetermined = ErrorBehavior(on_underdetermined)

        try:
            was_overdetermined = self.is_overdetermined
            self._check_shares_linked()
            self._splitter_used = False
            if self._stock is None:
                self._compute_without_stock()
            else:
                self._stock.compute_process()
        except UnderdeterminedError as error:
            if on_underdetermined == ErrorBehavior.ERROR:
                raise
            message = error.short_message if recursive else str(error)
            handle_error(behavior=on_underdetermined, message=message)
            return False

        if not self.is_computed:
            raise RuntimeError(
                f"After computation of process '{self.name}', there are still unknown flows or "
                "stock arrays. This indicates an internal flodym error."
            )
        # checked after the computation, to avoid repeated messages for underdetermined processes
        self._check_share_sums()
        if self._dimension_splitter is not None and not self._splitter_used:
            handle_error(
                behavior=config.error_behaviors.unused_dimension_splitter,
                message=(
                    f"In process '{self.name}': The dimension splitter is set, but was not used, "
                    "since no dimensions had to be added. The split it gives may not be fulfilled."
                    + config_hint("unused_dimension_splitter")
                ),
            )
        if was_overdetermined or config.checks.mass_balance_after_compute:
            hint = config_hint("mass_balance", tolerance=True)
            if was_overdetermined:
                hint = (
                    " The process was overdetermined, so the given values may be inconsistent."
                    + hint
                )
            report_balance_failures(
                [self._mass_balance_failure()],
                config.error_behaviors.mass_balance,
                hint=hint,
                success_message=f"Mass balance of process '{self.name}' is consistent.",
            )

        if recursive:
            neighbors = [f.from_process for f in self._inflows.values()]
            neighbors += [f.to_process for f in self._outflows.values()]
            for neighbor in neighbors:
                neighbor.compute(on_underdetermined=ErrorBehavior.INFO, recursive=True)
        return True

    def _compute_without_stock(self) -> None:
        side = self._choose_total_side(SIDES)
        self._check_can_compute_flows(SIDES)
        total = self._compute_total(side)
        total = self._apply_dimension_splitter(total, SIDES)
        for side in SIDES:
            self._compute_flows(total, side)
            self._check_shares_fulfilled(total, side)

    def _compute_side_from_total(self, total: "FlodymArray", side: Side) -> None:
        """Compute the unknown flows of one side from a given total of this side, which may be
        extended by the dimension splitter. Used for processes with stock."""
        total = self._apply_dimension_splitter(total, (side,))
        self._compute_flows(total, side)
        self._check_shares_fulfilled(total, side)

    def _known_flows(self, side: Side) -> Dict[str, "Flow"]:
        return {name: f for name, f in self._flows(side).items() if f.is_set}

    def _unknown_flows(self, side: Side) -> Dict[str, "Flow"]:
        return {name: f for name, f in self._flows(side).items() if not f.is_set}

    def _neither_known(self, side: Side) -> List[str]:
        """Flows of which neither the value nor the share is given."""
        return [name for name in self._unknown_flows(side) if name not in self._shares(side)]

    def _both_known(self, side: Side) -> List[str]:
        """Flows of which both the value and the share are given."""
        return [name for name in self._known_flows(side) if name in self._shares(side)]

    def _total_method(self, side: Side) -> Optional[str]:
        """How the total can be computed from a side, or None if it cannot."""
        if not self._flows(side):
            return None
        if not self._unknown_flows(side):
            return "sum"
        if self._both_known(side):
            return "share"
        if not self._neither_known(side) and self._known_flows(side):
            return "remainder"
        return None

    def can_compute_total(self, side: Side) -> bool:
        """Whether the total throughput of the process can be computed from the given side."""
        return self._total_method(side) is not None

    def can_compute_flows(self, sides: Sequence[Side] = SIDES) -> bool:
        """Whether the unknown flows of the given sides can be computed from a total, i.e.
        whether at most one unknown flow per side has no share."""
        return all(len(self._neither_known(side)) <= 1 for side in sides)

    def _choose_total_side(self, sides: Sequence[Side]) -> Side:
        """The side from which the total can be computed, preferring sides where all flows are
        known. Raises an UnderdeterminedError if there is none."""
        ranking = ["sum", "share", "remainder"]
        candidates = [(ranking.index(m), s) for s in sides if (m := self._total_method(s))]
        if not candidates:
            raise UnderdeterminedError(self, self._total_failure_message(sides))
        return min(candidates)[1]

    def _check_can_compute_total(self, side: Side) -> None:
        if not self.can_compute_total(side):
            raise UnderdeterminedError(self, self._total_failure_message((side,)))

    def _check_can_compute_flows(self, sides: Sequence[Side]) -> None:
        if not self.can_compute_flows(sides):
            message = ""
            for side in sides:
                if len(self._neither_known(side)) > 1:
                    message += (
                        f"Cannot compute the {side}flows from the total, since for more than one "
                        f"of them, neither share nor value is given: "
                        f"{', '.join(self._neither_known(side))}. Give a share or value for all "
                        "but one of them.\n"
                    )
            raise UnderdeterminedError(self, message)

    def _total_failure_message(self, sides: Sequence[Side]) -> str:
        message = "Cannot compute the total throughput of the process.\n"
        for side in sides:
            message += f"From the {side}flows, since:\n"
            if not self._flows(side):
                message += f"- the process has no {side}flows\n"
                continue
            if not self._known_flows(side):
                message += f"- no {side}flow is known (is_set=True)\n"
            for name in self._neither_known(side):
                message += f"- for flow '{name}', neither share nor value is given\n"
        return message

    def _compute_total(self, side: Side) -> "FlodymArray":
        """Total throughput of the process, computed from the flows of one side."""
        method = self._total_method(side)
        flows = self._flows(side)
        shares = self._shares(side)
        if method == "sum":
            return sum(flows.values(), 0.0)  # type: ignore[return-value]
        if method == "share":
            for name in self._both_known(side):
                excess_dims = shares[name].dims - flows[name].dims
                if excess_dims:
                    raise ValueError(
                        f"In process '{self.name}': The share of flow '{name}' has dimensions "
                        f"{excess_dims.names}, which the flow does not have."
                    )
                if not np.any(shares[name].values == 0):
                    return flows[name] / shares[name]
            # all shares have zeros: the total is not determined where the share is zero
            if self._total_method_remainder_possible(side):
                return self._compute_total_from_remainder(side)
            raise ValueError(
                f"In process '{self.name}': Cannot compute the total throughput of the process "
                f"from the {side}flows {', '.join(self._both_known(side))} and their shares, "
                "since the shares are zero for some items."
            )
        if method == "remainder":
            return self._compute_total_from_remainder(side)
        raise UnderdeterminedError(self, self._total_failure_message((side,)))

    def _total_method_remainder_possible(self, side: Side) -> bool:
        return not self._neither_known(side) and bool(self._known_flows(side))

    def _compute_total_from_remainder(self, side: Side) -> "FlodymArray":
        """Total as sum(known flows) / (1 - sum(shares of unknown flows))."""
        shares = self._shares(side)
        known_flows = self._known_flows(side)
        sum_known = sum(known_flows.values(), 0.0)
        shares_unknown = {name: shares[name] for name in self._unknown_flows(side)}
        dims_unknown = DimensionSet(dim_list=[])
        for share in shares_unknown.values():
            dims_unknown |= share.dims
        excess_dims = dims_unknown - sum_known.dims
        if excess_dims:
            share_names = [n for n, s in shares_unknown.items() if s.dims - sum_known.dims]
            raise ValueError(
                f"In process '{self.name}': Cannot compute the total throughput of the process "
                f"from the known {side}flows {', '.join(known_flows)} and the shares of the "
                f"unknown flows {', '.join(share_names)}, since these shares have the dimensions "
                f"{excess_dims.names}, which not all known flows have."
            )
        sum_shares = sum((s.cast_to(dims_unknown) for s in shares_unknown.values()), 0.0)
        remaining_share = 1.0 - sum_shares
        if np.any(np.isclose(np.asarray(getattr(remaining_share, "values", remaining_share)), 0)):
            raise ValueError(
                f"In process '{self.name}': Cannot compute the total throughput of the process "
                f"from the known {side}flows {', '.join(known_flows)}, since the shares of the "
                "unknown flows sum to one for some items, so the known flows must be zero there."
            )
        return sum_known / remaining_share

    def _apply_dimension_splitter(
        self, total: "FlodymArray", sides: Sequence[Side]
    ) -> "FlodymArray":
        """Multiply the total with the dimension splitter, if unknown flows of the given sides
        have dimensions which the total does not have."""
        dims_unknown = DimensionSet(dim_list=[])
        for side in sides:
            for flow in self._unknown_flows(side).values():
                dims_unknown |= flow.dims
        missing_dims = dims_unknown - total.dims
        if not missing_dims:
            return total

        splitter = self._dimension_splitter
        if splitter is None:
            raise ValueError(
                f"In process '{self.name}': Unknown flows have the dimensions "
                f"{missing_dims.names}, which the total throughput of the process does not have. "
                "Set a dimension splitter to add them."
            )
        if missing_dims - splitter.dims:
            raise ValueError(
                f"In process '{self.name}': The dimension splitter does not have the dimensions "
                f"{(missing_dims - splitter.dims).names}, which unknown flows have, but the total "
                "throughput of the process does not have."
            )
        added_dims = splitter.dims - total.dims
        if config.error_behaviors.process_shares != ErrorBehavior.IGNORE:
            splitter_sum = splitter.sum_to((splitter.dims & total.dims).letters)
            tolerance = self._share_tolerance(n_summed=int(np.prod(added_dims.shape)))
            max_error = max_abs(splitter_sum - 1.0)
            if not max_error <= tolerance:
                handle_error(
                    behavior=config.error_behaviors.process_shares,
                    message=(
                        f"In process '{self.name}': The dimension splitter does not sum to one "
                        f"over the dimensions it adds ({', '.join(added_dims.names)}), with a "
                        f"maximum error of {max_error}." + config_hint("process_shares")
                    ),
                )
        self._splitter_used = True
        return total * splitter

    def _compute_flows(self, total: "FlodymArray", side: Side) -> None:
        """Compute the unknown flows of a side from the total."""
        unknown_flows = self._unknown_flows(side)
        if not unknown_flows:
            return
        shares = self._shares(side)
        # Computed flows are first kept with the dimensions of the total: Reducing them to the
        # flow dimensions before computing the remaining flow could lose dimensions it needs.
        computed = {}
        for name in unknown_flows:
            if name not in shares:
                continue
            excess_dims = shares[name].dims - total.dims
            if excess_dims:
                raise ValueError(
                    f"In process '{self.name}': The share of flow '{name}' has the dimensions "
                    f"{excess_dims.names}, which the total throughput of the process does not "
                    "have. To add these dimensions to the flow, insert a separate process with "
                    "a dimension splitter."
                )
            computed[name] = total * shares[name]
        remaining = [name for name in unknown_flows if name not in computed]
        if remaining:
            name = remaining[0]
            flow = unknown_flows[name]
            other_flows = list(self._known_flows(side).values()) + list(computed.values())
            sum_others = sum(other_flows, 0.0)
            from .flodym_arrays import FlodymArray

            if isinstance(sum_others, FlodymArray) and flow.dims - sum_others.dims:
                raise ValueError(
                    f"In process '{self.name}': Cannot compute flow '{name}' as the total "
                    f"throughput minus the other {side}flows, since not all of these have its "
                    f"dimensions {(flow.dims - sum_others.dims).names}."
                )
            flow[...] = total - sum_others
        for name, values in computed.items():
            unknown_flows[name][...] = values

    def _check_shares_linked(self) -> None:
        for side in SIDES:
            unlinked = [name for name in self._shares(side) if name not in self._flows(side)]
            if unlinked:
                raise ValueError(
                    f"In process '{self.name}': Shares are given for the {side}flows "
                    f"{', '.join(unlinked)}, which are not linked to the process."
                )

    def _share_tolerance(self, n_summed: int = 1) -> float:
        """Tolerance for checks of dimensionless shares and splitters."""
        return config.relative_tolerance * float(np.finfo(np.float64).eps) * max(1, n_summed)

    def _check_share_values(self, share: "FlodymArray", label: str) -> None:
        """Check that a share or dimension splitter is between zero and one."""
        if config.error_behaviors.process_shares == ErrorBehavior.IGNORE:
            return
        tolerance = self._share_tolerance()
        values = share.values
        if np.any(np.isnan(values)) or np.any(values < -tolerance) or np.any(values > 1 + tolerance):
            handle_error(
                behavior=config.error_behaviors.process_shares,
                message=(
                    f"In process '{self.name}': {label} ('{share.name}') has values which are "
                    "NaN or not between zero and one." + config_hint("process_shares")
                ),
            )

    def _check_share_sums(self) -> None:
        """Check that the shares of each side sum to at most one, and to one if all flows of the
        side have a share."""
        if config.error_behaviors.process_shares == ErrorBehavior.IGNORE:
            return
        for side in SIDES:
            shares = self._shares(side)
            if not shares:
                continue
            for name, share in shares.items():
                self._check_share_values(share, f"Share of flow '{name}'")
            dims = DimensionSet(dim_list=[])
            for share in shares.values():
                dims |= share.dims
            share_sum = sum((s.cast_to(dims) for s in shares.values()), 0.0)
            sum_values = np.asarray(getattr(share_sum, "values", share_sum))
            tolerance = self._share_tolerance(n_summed=len(shares))
            all_have_shares = len(shares) == len(self._flows(side))
            if all_have_shares:
                failed = not np.all(np.abs(sum_values - 1.0) <= tolerance)
                expected = "sum to one, since all of them have shares"
            else:
                failed = not np.all(sum_values <= 1.0 + tolerance)
                expected = "sum to at most one"
            if failed:
                handle_error(
                    behavior=config.error_behaviors.process_shares,
                    message=(
                        f"In process '{self.name}': The shares of the {side}flows "
                        f"({', '.join(shares)}) do not {expected}." + config_hint("process_shares")
                    ),
                )

    def _check_shares_fulfilled(self, total: "FlodymArray", side: Side) -> None:
        """Check that the computed flows of a side fulfill their shares, which may not be the
        case if the process is overdetermined."""
        if config.error_behaviors.process_shares == ErrorBehavior.IGNORE:
            return
        flows = self._flows(side)
        failures = []
        for name, share in self._shares(side).items():
            error = total * share - flows[name]
            max_error = max_abs(error)
            if not max_error <= self.tolerance:
                failures.append(f"'{name}' (max error: {max_error})")
        if failures:
            handle_error(
                behavior=config.error_behaviors.process_shares,
                message=(
                    f"In process '{self.name}': After computation, the shares of the {side}flows "
                    f"{', '.join(failures)} are not fulfilled. This may be due to the process "
                    "being overdetermined, i.e. more flow values or shares being given than "
                    "needed." + config_hint("process_shares", tolerance=True)
                ),
            )


def make_processes(definitions: List[str | ProcessDefinition]) -> dict[str, Process]:
    """Create a dictionary of processes from a list of process names or process definitions.

    Processes given as :py:class:`flodym.ProcessDefinition` with an ``id`` get that ID. IDs need
    not be contiguous, but must be unique and non-negative. Processes given as names or as
    definitions without ``id`` get the smallest IDs not used by any definition, in the order they
    are listed. If no IDs are given explicitly, the IDs are thus the positions in the list.

    The parameters of process definitions (such as shares) are set separately, by
    :py:func:`set_process_parameters`.
    The process with ID 0 must be named 'sysenv'.
    """
    explicit_ids = [
        d.id for d in definitions if isinstance(d, ProcessDefinition) and d.id is not None
    ]
    if any(i < 0 for i in explicit_ids):
        raise ValueError(f"Process IDs must be non-negative, but got {sorted(explicit_ids)}.")
    if len(set(explicit_ids)) != len(explicit_ids):
        raise ValueError(f"Process IDs must be unique, but got {sorted(explicit_ids)}.")
    used_ids = set(explicit_ids)
    next_free_id = 0
    processes = {}
    for definition in definitions:
        name = definition.name if isinstance(definition, ProcessDefinition) else definition
        if isinstance(definition, ProcessDefinition) and definition.id is not None:
            id = definition.id
        else:
            while next_free_id in used_ids:
                next_free_id += 1
            id = next_free_id
            used_ids.add(id)
        if name in processes:
            raise ValueError(f"Process name '{name}' is defined more than once.")
        processes[name] = Process(name=name, id=id)
    return processes


def set_process_parameters(
    processes: Dict[str, Process],
    definitions: List[str | ProcessDefinition],
    parameters: Dict[str, "Parameter"],
) -> None:
    """Set the shares and dimension splitters given in process definitions (see
    :py:class:`flodym.ProcessDefinition`) to the processes. Process names in ``definitions``
    are skipped. The processes must be linked to their flows (see
    :py:meth:`flodym.MFASystem.relink`). Called by :py:meth:`flodym.MFASystem.from_data_reader`.

    Args:
        processes: The processes, keyed by name.
        definitions: Process names or definitions, as in :py:attr:`flodym.MFADefinition.processes`.
        parameters: The parameters, keyed by name.
    """
    for definition in definitions:
        if isinstance(definition, str):
            continue
        if definition.name not in processes:
            raise ValueError(f"Process '{definition.name}' is not defined.")
        process = processes[definition.name]

        def get_parameter(name: str) -> "Parameter":
            if name not in parameters:
                raise ValueError(
                    f"Parameter '{name}' given in the definition of process '{definition.name}' "
                    "is not defined."
                )
            return parameters[name]

        for flow, parameter_name in definition.inflow_shares.items():
            process.set_inflow_share(flow, get_parameter(parameter_name))
        for flow, parameter_name in definition.outflow_shares.items():
            process.set_outflow_share(flow, get_parameter(parameter_name))
        if definition.dimension_splitter is not None:
            process.set_dimension_splitter(get_parameter(definition.dimension_splitter))
