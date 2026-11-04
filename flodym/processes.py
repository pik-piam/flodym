import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, PrivateAttr, model_validator

from .config import ErrorBehavior, _config_hint, _exceeds_tolerance, config, handle_error
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
    """

    name: str
    """Name of the process."""
    id: int
    """ID of the process."""
    _inflows: Dict[str, "Flow"] = PrivateAttr(default_factory=dict)
    _outflows: Dict[str, "Flow"] = PrivateAttr(default_factory=dict)
    _stock: Optional["Stock"] = PrivateAttr(default=None)

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
        max_error = _max_abs(self.get_mass_balance(mass_change=mass_change))
        if tolerance is None:
            tolerance = self.tolerance
        if _exceeds_tolerance(max_error, tolerance):
            if error_behavior is None:
                error_behavior = config.error_behaviors.mass_balance
            handle_error(
                behavior=error_behavior,
                message=f"Mass balance check failed for process '{self.name}' "
                f"(max error: {max_error}, tolerance: {tolerance})."
                + _config_hint("mass_balance", tolerance=True),
            )
            return False
        logging.info(f"Mass balance of process '{self.name}' is consistent.")
        return True


def _max_abs(array: Union["FlodymArray", float]) -> float:
    """Largest absolute value of a FlodymArray or scalar."""
    values = np.asarray(getattr(array, "values", array))
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(values)))


def make_processes(definitions: List[str | ProcessDefinition]) -> dict[str, Process]:
    """Create a dictionary of processes from a list of process names or process definitions.

    Processes given as :py:class:`flodym.ProcessDefinition` get the ID of the definition. IDs need
    not be contiguous, but must be unique and non-negative. Processes given as names get the
    smallest IDs not used by any definition, in the order they are listed. If all processes are
    given as names, their IDs are thus their positions in the list.
    The process with ID 0 must be named 'sysenv'.
    """
    explicit_ids = [d.id for d in definitions if isinstance(d, ProcessDefinition)]
    if any(i < 0 for i in explicit_ids):
        raise ValueError(f"Process IDs must be non-negative, but got {sorted(explicit_ids)}.")
    if len(set(explicit_ids)) != len(explicit_ids):
        raise ValueError(f"Process IDs must be unique, but got {sorted(explicit_ids)}.")
    used_ids = set(explicit_ids)
    next_free_id = 0
    processes = {}
    for definition in definitions:
        if isinstance(definition, ProcessDefinition):
            name, id = definition.name, definition.id
        else:
            while next_free_id in used_ids:
                next_free_id += 1
            name, id = definition, next_free_id
            used_ids.add(id)
        if name in processes:
            raise ValueError(f"Process name '{name}' is defined more than once.")
        processes[name] = Process(name=name, id=id)
    return processes
