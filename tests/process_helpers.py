"""Helpers for the tests of the automatic computation of processes.

Used by test_process_compute.py and test_process_system.py.
"""

import numpy as np

import flodym as fd
from flodym import (
    DataReader,
    Dimension,
    DimensionDefinition,
    DimensionSet,
    FixedLifetime,
    FlowDefinition,
    MFADefinition,
    MFASystem,
    Parameter,
    ParameterDefinition,
    ProcessDefinition,
    StockDefinition,
)
from flodym._config import ErrorBehavior, config, handle_error

# solve_block is private; it is used to run the simultaneous computation on single processes
from flodym._process_compute import UnderdeterminedError, check_completed_process
from flodym._process_system import solve_block

TIME = [2000, 2001, 2002, 2003]
REGIONS = ["north", "south"]
GOODS = ["cars", "houses", "tools"]
DIM_ITEMS = {"Time": TIME, "Region": REGIONS, "Cohort": TIME, "Good": GOODS}
DIM_DEFINITIONS = [
    DimensionDefinition(name="Time", letter="t", dtype=int),
    DimensionDefinition(name="Region", letter="r", dtype=str),
]


class DictReader(DataReader):
    """Reads dimension items and parameter values from dictionaries."""

    def __init__(self, parameter_values: dict[str, np.ndarray], dim_items: dict | None = None):
        self.parameter_values = parameter_values
        self.dim_items = DIM_ITEMS if dim_items is None else {**DIM_ITEMS, **dim_items}

    def read_dimension(self, definition: DimensionDefinition) -> Dimension:
        return Dimension(
            name=definition.name, letter=definition.letter, items=self.dim_items[definition.name]
        )

    def read_parameter_values(self, parameter_name: str, dims: DimensionSet) -> Parameter:
        values = np.broadcast_to(self.parameter_values[parameter_name], dims.shape).copy()
        return Parameter(dims=dims, values=values, name=parameter_name)


def make_mfa(
    processes: list,
    flows: list[FlowDefinition],
    parameter_values: dict[str, np.ndarray],
    parameter_dims: dict[str, tuple[str, ...]],
    stocks: list[StockDefinition] | None = None,
    dimensions: list[DimensionDefinition] = DIM_DEFINITIONS,
    dim_items: dict | None = None,
) -> MFASystem:
    definition = MFADefinition(
        dimensions=dimensions,
        processes=processes,
        flows=flows,
        stocks=stocks or [],
        parameters=[
            ParameterDefinition(name=name, dim_letters=dims)
            for name, dims in parameter_dims.items()
        ],
    )
    mfa = MFASystem.from_data_reader(definition, DictReader(parameter_values, dim_items))
    # fill prescribed flows, so that single processes can be computed without
    # compute_all_possible()
    mfa.set_arrays_from_prescribed()
    return mfa


def flow(from_process: str, to_process: str, dims=("t",), **kwargs) -> FlowDefinition:
    return FlowDefinition(
        from_process_name=from_process, to_process_name=to_process, dim_letters=dims, **kwargs
    )


def split_mfa(**process_kwargs) -> MFASystem:
    """sysenv -> a -> (b, c), b -> sysenv, c -> sysenv, with a share of a -> b of 0.3."""
    return make_mfa(
        processes=[
            "sysenv",
            ProcessDefinition(name="a", outflow_shares={"b": "share b"}, **process_kwargs),
            "b",
            "c",
        ],
        flows=[
            flow("sysenv", "a", prescribed="input"),
            flow("a", "b"),
            flow("a", "c"),
            flow("b", "sysenv"),
            flow("c", "sysenv"),
        ],
        parameter_values={"input": np.array([1.0, 2.0, 3.0, 4.0]), "share b": 0.3},
        parameter_dims={"input": ("t",), "share b": ()},
    )


def splitter_mfa(split_values, **process_kwargs) -> MFASystem:
    """sysenv -> a -> b -> sysenv, with the outflow of a by region and the inflow not."""
    return make_mfa(
        processes=["sysenv", ProcessDefinition(name="a", **process_kwargs), "b"],
        flows=[
            flow("sysenv", "a", prescribed="input"),
            flow("a", "b", dims=("t", "r")),
            flow("b", "sysenv", dims=("t", "r")),
        ],
        parameter_values={"input": np.array([1.0, 2.0, 3.0, 4.0]), "split": split_values},
        parameter_dims={"input": ("t",), "split": ("r",)},
    )


def stock_mfa(
    subclass: type,
    lifetime: bool = True,
    parameter_values: dict | None = None,
    parameter_dims: dict | None = None,
    **stock_kwargs,
) -> MFASystem:
    """sysenv -> use (with stock) -> (eol, sysenv), with a share of use -> eol of 0.6."""
    if lifetime:
        stock_kwargs["lifetime_model_class"] = FixedLifetime
    mfa = make_mfa(
        processes=["sysenv", ProcessDefinition(name="use", outflow_shares={"eol": "share"}), "eol"],
        flows=[
            flow("sysenv", "use"),
            flow("use", "eol"),
            flow("use", "sysenv"),
            flow("eol", "sysenv"),
        ],
        parameter_values={"share": 0.6, **(parameter_values or {})},
        parameter_dims={"share": (), **(parameter_dims or {})},
        dimensions=DIM_DEFINITIONS + [DimensionDefinition(name="Cohort", letter="c", dtype=int)],
        stocks=[
            StockDefinition(
                name="use", process="use", dim_letters=("t",), subclass=subclass, **stock_kwargs
            )
        ],
    )
    if lifetime:
        mfa.stocks["use"].lifetime_model.mean = 1
    return mfa


def compute_process_via_block(process: fd.Process, on_underdetermined, short_message: bool) -> bool:
    """Replacement of ``compute_process`` which solves the equations of the single process."""
    if process.id == 0 or process.is_computed:
        return process.id != 0
    neighbors = [p for p in process._neighbors() if p.id != 0 and not p.is_computed]
    result = solve_block([process])
    if not process.is_computed:
        error = UnderdeterminedError(process, result.explanation.message)
        message = error.short_message if short_message else str(error)
        if on_underdetermined is None:
            on_underdetermined = config.error_behaviors.process_underdetermined
        if ErrorBehavior(on_underdetermined) == ErrorBehavior.ERROR:
            raise error
        handle_error(behavior=on_underdetermined, message=message)
        return False
    for neighbor in neighbors:
        if neighbor.is_computed:
            check_completed_process(neighbor)
    return True
