"""The model function of an MFA system, see :py:func:`flodym.make_model_function`.

The model function saves the state of the system, writes the given parameter values into it,
resets and computes it, and restores the state afterwards.
"""

from collections.abc import Callable, Iterable, Mapping
from typing import Any, NamedTuple

import numpy as np

from ._checks import config_hint
from ._config import config, handle_error
from ._process_compute import (
    restore_arrays,
    restore_cohort_arrays,
    save_arrays,
    save_cohort_arrays,
)
from ._sampling import ModelCheckError
from .flodym_arrays import FlodymArray
from .mfa_system import MFASystem
from .stocks import DynamicStockModel

STOCK_ARRAYS = ("stock", "inflow", "outflow")
"""Arrays of a stock returned by the model function, as ``"<stock name>.<array>"``."""


def make_model_function(
    mfa: MFASystem,
    *,
    track: Iterable[str] | None = None,
    derived: Mapping[str, Callable[[MFASystem], FlodymArray | float]] | None = None,
    compute: Callable[[MFASystem], object] | None = None,
) -> Callable[[Mapping[str, FlodymArray]], dict[str, FlodymArray | float]]:
    """Turn an MFA system into a function of its parameters, for :py:func:`flodym.run_monte_carlo`.

    Each call of the returned function

    1. saves the state of the system: values and ``is_set`` of flows and stock arrays, the
       cohort arrays of dynamic stocks, the values of all parameters and the entries of
       :py:attr:`flodym.MFASystem.parameters`,
    2. puts the parameters the system had when the function was made back into
       :py:attr:`flodym.MFASystem.parameters`, and writes all given values into them, so that
       shares, dimension splitters and prescribed arrays linked to them use the given values,
    3. sets all flows and stock arrays to zero and unset, as in a new system, sets the
       prescribed arrays (see :py:meth:`flodym.MFASystem.set_arrays_from_prescribed`) and calls
       ``compute(mfa)``, by default ``mfa.compute()``,
    4. returns copies of the tracked flows and stock arrays and the derived quantities,
    5. restores the saved state, also if ``compute`` raises.

    The parameters of the system are taken when the function is made, so make it after setting
    up the parameters, e.g. after :py:meth:`flodym.MFASystem.link_parameters`. Parameters with
    integer or read-only values get new float values during a call, so that sampled values are
    not rounded; their values are restored afterwards.

    Lifetime model parameters and initial stocks of dynamic stocks are not reset. A lifetime
    model copies its parameters when they are assigned, so lifetime parameters assigned outside
    ``compute()`` do not follow the given parameters. If a dynamic stock is computed with the
    lifetime parameters it had before ``compute()``, this is reported according to
    ``flodym.config.error_behaviors.fixed_lifetime`` (default: a warning, once per stock). Assign
    them in ``compute()``, or write a model function by hand for everything else that
    ``compute()`` does not do.

    The returned function can be pickled with the system, e.g. to run it in other processes,
    if ``derived`` and ``compute`` are functions defined at module level.

    **Example**

        >>> def copper_content(mfa: fd.MFASystem) -> fd.FlodymArray:
        ...     secondary_steel = mfa.flows["remelting => sysenv"]
        ...     return secondary_steel["Cu"] / secondary_steel.sum_over(("e",))
        >>> model = fd.make_model_function(
        ...     mfa, track=["remelting => sysenv"], derived={"copper content": copper_content}
        ... )
        >>> result = fd.run_monte_carlo(model, mfa.parameters, uncertainty, n_samples=1000)

    Args:
        mfa: The MFA system.
        track: Names of flows, stocks and stock arrays to return. A flow is returned under its
            name, a stock as ``"<name>.stock"``, ``"<name>.inflow"`` and ``"<name>.outflow"``,
            which can also be given individually. If None, all flows and stocks are returned.
        derived: Functions of the computed system, by name, e.g. for a recycling rate or a flow
            summed over regions. Quantiles of a ratio or a sum must be computed per sample, so
            compute such quantities here. They are returned in addition to the tracked arrays.
        compute: What to call instead of ``mfa.compute()``, e.g.
            ``lambda mfa: mfa.compute_all_possible()``.

    Returns:
        A function of the parameters, by name, returning the outputs, by name. It must be
        called with all parameters of the system.

    Raises:
        ValueError: If names in ``track`` are not names of flows, stocks or stock arrays (all
            are listed), or if a derived quantity has the name of a tracked array. The returned
            function raises it if the names of the given parameters differ from those of the
            system, or if their dimensions differ.
    """
    return ModelFunction(mfa, track=track, derived=derived, compute=compute)


class TrackedArray(NamedTuple):
    """A flow or stock array returned by the model function."""

    key: str
    """Name of the output."""
    stock_array: str | None
    """For a stock array, which one (``"stock"``, ``"inflow"`` or ``"outflow"``)."""
    name: str
    """Name of the flow or stock."""


class LifetimeState(NamedTuple):
    """A dynamic stock with the objects of its lifetime model before a computation."""

    stock: DynamicStockModel
    objects: list[Any]
    """See ``LifetimeModel._parameter_objects()``."""


class ModelFunction:
    """The function returned by :py:func:`flodym.make_model_function`.

    A class instead of a closure, so that it can be pickled.
    """

    def __init__(
        self,
        mfa: MFASystem,
        track: Iterable[str] | None,
        derived: Mapping[str, Callable[[MFASystem], FlodymArray | float]] | None,
        compute: Callable[[MFASystem], object] | None,
    ) -> None:
        self.mfa = mfa
        self.tracked = _tracked(mfa, track)
        self.derived = dict(derived or {})
        self.compute = compute
        tracked_keys = [tracked.key for tracked in self.tracked]
        colliding = [name for name in self.derived if name in tracked_keys]
        if colliding:
            raise ValueError(
                f"Derived quantities {colliding} have the names of tracked flows or stock arrays. "
                "Rename them."
            )
        self.parameters = dict(mfa.parameters)
        """The parameters of the system when the function was made."""
        self.reported_stocks: set[str] = set()
        """Names of stocks reported for lifetime parameters assigned outside compute()."""

    def __call__(self, parameters: Mapping[str, FlodymArray]) -> dict[str, FlodymArray | float]:
        _check_parameters(parameters, self.parameters)
        state = SystemState(self.mfa, self.parameters.values())
        try:
            state.use_parameters(self.parameters)
            _write_parameter_values(self.parameters, parameters)
            state.reset_arrays()
            lifetimes_before = _lifetimes(self.mfa)
            self.mfa.set_arrays_from_prescribed()
            if self.compute is None:
                self.mfa.compute()
            else:
                self.compute(self.mfa)
            outputs: dict[str, FlodymArray | float] = {
                tracked.key: self._array(tracked).copy() for tracked in self.tracked
            }
            for name, function in self.derived.items():
                value = function(self.mfa)
                outputs[name] = value.copy() if isinstance(value, FlodymArray) else value
            self._report_fixed_lifetimes(lifetimes_before)
            return outputs
        finally:
            state.restore()

    def _array(self, tracked: TrackedArray) -> FlodymArray:
        if tracked.stock_array is None:
            return self.mfa.flows[tracked.name]
        return getattr(self.mfa.stocks[tracked.name], tracked.stock_array)

    def _report_fixed_lifetimes(self, lifetimes_before: dict[str, LifetimeState]) -> None:
        """Report dynamic stocks computed with the lifetime parameters they had before.

        Raises:
            ModelCheckError: If this is configured as error, see
                :py:attr:`flodym.ErrorBehaviors.fixed_lifetime`.
        """
        for name, stock in self.mfa.stocks.items():
            if name in self.reported_stocks or name not in lifetimes_before:
                continue
            before = lifetimes_before[name]
            if stock is not before.stock or not stock.is_computed:
                continue
            objects = before.stock.lifetime_model._parameter_objects()
            if len(objects) != len(before.objects) or any(
                a is not b for a, b in zip(objects, before.objects)
            ):
                continue
            message = (
                f"Dynamic stock '{name}' was computed with the lifetime model parameters it had "
                "before compute(). Lifetime model parameters are copied when they are assigned, "
                "so if they depend on parameters, assigning them outside compute() means they do "
                "not follow the given parameters. Assign them in compute()."
                + config_hint("fixed_lifetime")
            )
            handle_error(config.error_behaviors.fixed_lifetime, error=ModelCheckError(message))
            self.reported_stocks.add(name)


class SystemState:
    """The state of an MFA system when this object is created, see :py:class:`ModelFunction`."""

    def __init__(self, mfa: MFASystem, parameters: Iterable[FlodymArray]) -> None:
        self.mfa = mfa
        self.parameters_dict = mfa.parameters
        """The dictionary of the parameters, restored if compute() replaces it."""
        self.parameter_entries = dict(mfa.parameters)
        self.computed_arrays = [array for _, array in mfa._prescribable_arrays()]
        """Flows and stock arrays."""
        self.arrays = save_arrays(
            self.computed_arrays + list(parameters) + list(self.parameter_entries.values())
        )
        self.cohort_arrays = [(stock, save_cohort_arrays(stock)) for stock in mfa.stocks.values()]

    def use_parameters(self, entries: Mapping[str, FlodymArray]) -> None:
        """Make the saved dictionary of parameters the one of the system, with these entries."""
        if self.mfa.parameters is not self.parameters_dict:
            self.mfa.parameters = self.parameters_dict
        self.parameters_dict.clear()
        self.parameters_dict.update(entries)

    def reset_arrays(self) -> None:
        """Set flows, stock arrays and cohort arrays to zero, and flows and stock arrays unset."""
        for array in self.computed_arrays:
            if array.values.flags.writeable:
                array.values[...] = 0
            else:
                array.values = np.zeros_like(array.values)
        self.mfa.mark_all_unset()
        for stock, cohort_arrays in self.cohort_arrays:
            for name, values in cohort_arrays.items():
                setattr(stock, name, np.zeros_like(values))

    def restore(self) -> None:
        """Restore the saved state."""
        self.use_parameters(self.parameter_entries)
        restore_arrays(self.arrays)
        for stock, cohort_arrays in self.cohort_arrays:
            restore_cohort_arrays(stock, cohort_arrays)


def _lifetimes(mfa: MFASystem) -> dict[str, LifetimeState]:
    """Dynamic stocks with the objects of their lifetime models, by stock name."""
    return {
        name: LifetimeState(stock, stock.lifetime_model._parameter_objects())
        for name, stock in mfa.stocks.items()
        if isinstance(stock, DynamicStockModel)
    }


def _tracked(mfa: MFASystem, track: Iterable[str] | None) -> list[TrackedArray]:
    """The tracked flows and stock arrays.

    Raises:
        ValueError: If names are neither flows, stocks nor stock arrays.
    """
    if track is None:
        track = list(mfa.flows) + list(mfa.stocks)
    elif isinstance(track, str):
        track = [track]
    tracked: dict[str, TrackedArray] = {}
    unknown = []
    for name in track:
        stock_name, _, array_name = name.rpartition(".")
        if name in mfa.flows:
            tracked[name] = TrackedArray(name, None, name)
        elif name in mfa.stocks:
            for array in STOCK_ARRAYS:
                tracked[f"{name}.{array}"] = TrackedArray(f"{name}.{array}", array, name)
        elif stock_name in mfa.stocks and array_name in STOCK_ARRAYS:
            tracked[name] = TrackedArray(name, array_name, stock_name)
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(
            f"Names {unknown} in track are neither flows nor stocks of the MFA system, nor stock "
            f"arrays such as '<stock>.stock'. Flows: {list(mfa.flows)}. Stocks: "
            f"{list(mfa.stocks)}."
        )
    return list(tracked.values())


def _check_parameters(
    parameters: Mapping[str, FlodymArray], original_parameters: dict[str, FlodymArray]
) -> None:
    """Raise a ValueError if the parameter names or dimensions differ from the system's."""
    missing = [name for name in original_parameters if name not in parameters]
    extra = [name for name in parameters if name not in original_parameters]
    if missing or extra:
        raise ValueError(
            "The model function of an MFASystem must be called with the parameters of the "
            f"system. Missing parameters: {missing}. Unknown parameters: {extra}. For derived "
            "parameters, write a model function by hand, or derive them in compute()."
        )
    for name, original in original_parameters.items():
        given = parameters[name]
        if not isinstance(given, FlodymArray) or given.dims != original.dims:
            raise ValueError(
                f"Parameter '{name}' given to the model function has other dimensions than the "
                "parameter of the MFA system."
            )


def _write_parameter_values(
    original_parameters: dict[str, FlodymArray], parameters: Mapping[str, FlodymArray]
) -> None:
    """Write the given values into the parameters of the system.

    The values are written in place, so that shares, dimension splitters and prescribed arrays
    linked to the parameters use them. Integer or read-only values are replaced by new arrays,
    which the links see as well, since they refer to the parameter objects.
    """
    for name, original in original_parameters.items():
        given = parameters[name]
        if given is original:
            continue
        in_place = original.values.flags.writeable and np.can_cast(
            given.values.dtype, original.values.dtype, casting="same_kind"
        )
        if in_place:
            original.values[...] = given.values
        else:
            original.values = given.values.astype(np.result_type(original.values, given.values))
        if given.is_set:
            original.mark_set()
        else:
            original.mark_unset()
