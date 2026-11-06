from typing import TYPE_CHECKING, Dict, List, Optional, Union

from pydantic import BaseModel as PydanticBaseModel, PrivateAttr, model_validator

from ._checks import balance_failure, config_hint, report_balance_failures
from ._process_compute import (
    SPLITTER_SIDES,
    Direction,
    ProcessSide,
    Side,
    UnderdeterminedError as UnderdeterminedError,
    check_share_values,
    compute_process,
    compute_processes,
    is_overdetermined,
)
from .config import ErrorBehavior, config
from .mfa_definition import ProcessDefinition

if TYPE_CHECKING:
    from .flodym_arrays import FlodymArray, Flow
    from .stocks import Stock


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

    Optionally, shares of in- and outflows and dimension splitters can be given (see
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
    # Shares and splitters hold parameter names until linked (see _link_parameters).
    # Before linking, share keys are given as in the definition: flow or process names.
    _inflow_shares: Dict[str, Union["FlodymArray", str]] = PrivateAttr(default_factory=dict)
    _outflow_shares: Dict[str, Union["FlodymArray", str]] = PrivateAttr(default_factory=dict)
    _dimension_splitters: Dict[Direction, Union["FlodymArray", str]] = PrivateAttr(
        default_factory=dict
    )

    @model_validator(mode="after")
    def check_id0(self):
        """Ensure that the process with ID 0 is named 'sysenv', and vice versa."""
        if (self.id == 0) != (self.name == "sysenv"):
            raise ValueError(
                "The process with ID 0 must be named 'sysenv', as it contains everything outside "
                f"the system boundary, and no other process may have this name (got process "
                f"'{self.name}' with ID {self.id})."
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
            yield "inflow_shares", {k: _label(v) for k, v in self._inflow_shares.items()}
        if self._outflow_shares:
            yield "outflow_shares", {k: _label(v) for k, v in self._outflow_shares.items()}
        for direction, splitter in self._dimension_splitters.items():
            yield f"{direction}_dimension_splitter", _label(splitter)

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
    # Shares and dimension splitters

    @property
    def inflow_shares(self) -> Dict[str, Union["FlodymArray", str]]:
        """Shares of inflows in the total throughput of the process, keyed by flow name.
        Set them with :py:meth:`set_inflow_share`, or by a :py:class:`flodym.ProcessDefinition`.
        Shares from a definition are parameter names, keyed as in the definition, until they are
        linked by :py:meth:`flodym.MFASystem.link_parameters`."""
        return dict(self._inflow_shares)

    @property
    def outflow_shares(self) -> Dict[str, Union["FlodymArray", str]]:
        """Shares of outflows in the total throughput of the process, keyed by flow name.
        See :py:attr:`inflow_shares`."""
        return dict(self._outflow_shares)

    @property
    def forward_dimension_splitter(self) -> Optional["FlodymArray"]:
        """Array by which the total throughput of the process is multiplied in
        :py:meth:`compute` before computing the outflows, to add dimensions which the total does
        not have, but the outflows to compute do. Set it with :py:meth:`set_dimension_splitter`.
        """
        return self._dimension_splitters.get("forward")

    @property
    def backward_dimension_splitter(self) -> Optional["FlodymArray"]:
        """Array by which the total throughput of the process is multiplied in
        :py:meth:`compute` before computing the inflows, to add dimensions which the total does
        not have, but the inflows to compute do. Set it with :py:meth:`set_dimension_splitter`.
        """
        return self._dimension_splitters.get("backward")

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

    def set_dimension_splitter(
        self, direction: Direction, splitter: Optional["FlodymArray"]
    ) -> None:
        """Set a dimension splitter of the process, or remove it by passing None. The array is
        stored by reference, so later changes of its values are taken into account.

        Args:
            direction: ``"forward"`` for the splitter used to compute outflows (see
                :py:attr:`forward_dimension_splitter`), ``"backward"`` for the one used to compute
                inflows (see :py:attr:`backward_dimension_splitter`).
            splitter: The splitter. It must sum to one over the dimensions it adds to the total.
        """
        if direction not in SPLITTER_SIDES:
            raise ValueError(f"Direction must be 'forward' or 'backward', not '{direction}'.")
        if splitter is None:
            self._dimension_splitters.pop(direction, None)
            return
        check_share_values(self, splitter, f"The {direction} dimension splitter")
        self._dimension_splitters[direction] = splitter

    def _resolve_flow_name(self, side: Side, flow: str) -> str:
        """Get the name of an in- or outflow, given by flow name or by the name of the process
        it comes from or goes to."""
        flows = ProcessSide(self, side).flows
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
        check_share_values(self, share, f"Share of flow '{name}'")
        ProcessSide(self, side).shares[name] = share

    def _rename_share_keys(self, new_keys: Dict[str, str]) -> None:
        """Rename the flow names under which shares are stored, keeping their order.
        Used by :py:meth:`flodym.MFASystem.relink` when flows are renamed.

        Args:
            new_keys: New flow names, keyed by old flow names.
        """
        for shares in (self._inflow_shares, self._outflow_shares):
            renamed = {new_keys.get(name, name): share for name, share in shares.items()}
            shares.clear()
            shares.update(renamed)

    def _set_parameter_names(self, definition: ProcessDefinition) -> None:
        """Store the names of the shares and dimension splitters given in a process definition.
        They are replaced by the parameters in :py:meth:`_link_parameters`."""
        self._inflow_shares.update(definition.inflow_shares)
        self._outflow_shares.update(definition.outflow_shares)
        splitters = {
            "forward": definition.forward_dimension_splitter,
            "backward": definition.backward_dimension_splitter,
        }
        for direction, parameter_name in splitters.items():
            if parameter_name is not None:
                self._dimension_splitters[direction] = parameter_name

    def _unlinked_parameter_names(self) -> List[str]:
        """Names of the parameters of shares and dimension splitters which are not linked yet."""
        values = (
            list(self._inflow_shares.values())
            + list(self._outflow_shares.values())
            + list(self._dimension_splitters.values())
        )
        return [value for value in values if isinstance(value, str)]

    def _link_parameters(self, parameters: Dict[str, "FlodymArray"]) -> None:
        """Replace the parameter names of shares and dimension splitters by the parameters.
        Share keys given as process names are resolved to flow names, so the process must be
        linked to its flows (see :py:meth:`flodym.MFASystem.relink`). Linked entries are kept."""
        for side in ("in", "out"):
            shares = ProcessSide(self, side).shares
            linked = {}
            for key, share in shares.items():
                if isinstance(share, str):
                    key = self._resolve_flow_name(side, key)
                    share = parameters[share]
                    check_share_values(self, share, f"Share of flow '{key}'")
                if key in linked:
                    raise ValueError(
                        f"In process '{self.name}': Several shares are given for {side}flow "
                        f"'{key}'."
                    )
                linked[key] = share
            shares.clear()
            shares.update(linked)
        for direction, splitter in self._dimension_splitters.items():
            if isinstance(splitter, str):
                splitter = parameters[splitter]
                check_share_values(self, splitter, f"The {direction} dimension splitter")
                self._dimension_splitters[direction] = splitter

    def _check_parameters_linked(self) -> None:
        """Raise an error if shares or dimension splitters are not linked to parameters yet."""
        unlinked = self._unlinked_parameter_names()
        if unlinked:
            raise ValueError(
                f"In process '{self.name}': The parameters {', '.join(unlinked)} of shares or "
                "dimension splitters are not linked. Call MFASystem.link_parameters() first."
            )

    def _remove_share(self, side: Side, flow: str) -> None:
        shares = ProcessSide(self, side).shares
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
        if any(not f.is_set for f in self._inflows.values()):
            return False
        if any(not f.is_set for f in self._outflows.values()):
            return False
        return self._stock is None or self._stock.is_computed

    @property
    def is_overdetermined(self) -> bool:
        """Whether more flow values, shares and stock values are given than needed to compute
        the process with :py:meth:`compute`. In this case, the given values may be inconsistent,
        which :py:meth:`compute` checks.

        Unknowns are the in- and outflows and the totals: one total throughput without stock,
        or the inflow, outflow and stock of the stock. Equations are the sums of the in- and of
        the outflows, the given flow values, shares and stock arrays, and the equations of the
        stock model.
        """
        return is_overdetermined(self)

    def compute(
        self,
        on_underdetermined: Optional[ErrorBehavior | str] = None,
        compute_neighbors: bool = False,
    ) -> bool:
        """Compute the unknown in- and outflows of the process (and its stock, if any), from the
        known ones, the shares and the dimension splitters.

        Flows are known if they are set (see :py:attr:`flodym.FlodymArray.is_set`). This covers
        many, but not all MFA processes; others have to be computed manually, for example if two
        processes depend on each other and form an equation system.

        The computation assumes the following structure of a process without stock: The inflows
        sum to a total throughput, which is split into the outflows. It proceeds as follows:

        - The total is computed from the inflows or from the outflows (whichever is possible) as

          - the sum of the flows of that side, if all are known,
          - a known flow divided by its share, if both are given,
          - or as ``sum(known flows) / (1 - sum(shares of unknown flows))``, if the shares of all
            unknown flows of that side are given.

          This means that the shares of a side refer to the total of that side.
        - If unknown outflows have dimensions the total does not have, the total is multiplied
          with the forward dimension splitter before computing them. If unknown inflows have
          dimensions the total does not have, it is multiplied with the backward dimension
          splitter before computing them. Splitters must sum to one over the dimensions they
          add. Two splitters are needed if inflows and outflows each have dimensions the other
          side does not have, e.g. inflows by region and outflows by good.
        - Each unknown flow with a share is computed as total times share.
          One unknown flow per side may have no share; it is the total minus all other flows.

        Dimensions of flows are thus only added by the dimension splitters. To add dimensions to
        single flows, insert a process with a dimension splitter.

        For a process with stock, the inflows and the outflows each have their own total: the
        inflow and the outflow of the stock. The shares of each side refer to its total.
        The stock is computed along with the process, depending on its type:

        - :py:class:`flodym.SimpleFlowDrivenStock`: The totals of both sides are computed from
          their flows (or taken from the stock inflow or outflow, if set), and the stock from
          them.
        - :py:class:`flodym.DynamicStockModel` and its subclasses: The mode is chosen from the set
          arrays as in :py:meth:`flodym.DynamicStockModel.compute`. A stock-driven model needs a
          set stock. An inflow-driven model takes the set stock inflow, or computes it as the
          total of the inflows. The unknown in- and outflows are then computed from the stock
          inflow and outflow.

        Other stock types are not supported.

        If the process is overdetermined (see :py:attr:`is_overdetermined`), the given values are
        checked for consistency afterwards: its mass balance, its shares, and whether stock
        arrays which were set kept their values. Neighboring processes whose flows were all set
        by this computation are checked in the same way.

        If the computation or one of the checks raises an error (also if the process is
        underdetermined), the flows of the process and the arrays of its stock are restored to
        their state before the computation.

        Args:
            on_underdetermined: What to do if the process cannot be computed from the given
                values. If None, ``flodym.config.error_behaviors.process_underdetermined`` is used
                (default: raise an :py:class:`flodym.UnderdeterminedError`).
            compute_neighbors: If True, after a successful computation, the neighboring
                processes are computed as well, and so on, until no further process can be
                computed (see :py:func:`compute_processes`). Underdetermined neighbors are skipped
                with an info log message. See also :py:meth:`flodym.MFASystem.compute_all_possible`.

        Returns:
            Whether the process is computed afterwards.
        """
        is_computed = compute_process(self, on_underdetermined, short_message=False)
        if compute_neighbors and is_computed:
            compute_processes(self._neighbors())
        return is_computed

    def _neighbors(self) -> List["Process"]:
        """Processes connected to this one by a flow, without duplicates."""
        neighbors = [f.from_process for f in self._inflows.values()]
        neighbors += [f.to_process for f in self._outflows.values()]
        return list({id(p): p for p in neighbors}.values())


def make_processes(definitions: List[str | ProcessDefinition]) -> dict[str, Process]:
    """Create a dictionary of processes from a list of process names or process definitions.

    Processes given as :py:class:`flodym.ProcessDefinition` with an ``id`` get that ID. IDs need
    not be contiguous, but must be unique and non-negative. Processes given as names or as
    definitions without ``id`` get the smallest IDs not used by any definition, in the order they
    are listed. If no IDs are given explicitly, the IDs are thus the positions in the list.

    The names of the parameters of process definitions (shares and dimension splitters) are
    stored in the processes. They are replaced by the parameters in
    :py:meth:`flodym.MFASystem.link_parameters`.
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
        process = Process(name=name, id=id)
        if isinstance(definition, ProcessDefinition):
            process._set_parameter_names(definition)
        processes[name] = process
    return processes


def _label(value: Union["FlodymArray", str]) -> str:
    """Name of a parameter array, or the parameter name itself if not linked yet."""
    return value if isinstance(value, str) else value.name
