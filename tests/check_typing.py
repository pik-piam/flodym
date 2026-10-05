from numbers import Number
from typing import Literal, SupportsFloat

import numpy as np

from flodym import Dimension, DimensionSet, FlodymArray, Flow, MFASystem


def function_accepting_float(value: float) -> None:
    pass


function_accepting_float(3.14)
function_accepting_float(2)
function_accepting_float(np.float64(3.14))


def function_accepting_number(value: Number) -> None:
    pass


# Numbers are not supported by static type checkers: https://github.com/astral-sh/ty/issues/2563#issuecomment-3768776772
function_accepting_number(3.14)  # type: ignore[ty:invalid-argument-type]
function_accepting_number(2)  # type: ignore[ty:invalid-argument-type]
function_accepting_number(np.float64(3.14))  # type: ignore[ty:invalid-argument-type]


def function_accepting_supportsfloat(value: SupportsFloat) -> None:
    pass


function_accepting_supportsfloat(3.14)
function_accepting_supportsfloat(2)
function_accepting_supportsfloat(np.float64(3.14))

# Check: FlodymArray can be multiplied by scalars
array = FlodymArray(dims={"a": "first"}, values=[1.0, 2.0, 3.0])
array_times_pi: FlodymArray = array * 3.14
array_times_2: FlodymArray = array * 2
array_times_np: FlodymArray = array * np.float64(3.14)
array_times_str: FlodymArray = array * "3.14"  # type: ignore[ty:unsupported-operator]

# Check: dimension letters declared as a type argument are checked
Letters = Literal["t", "e"]


class TypedMFA(MFASystem[Letters]):
    def compute(self) -> None:
        flow = self.flows["sysenv => shredder"]
        flow.sum_over(("t",))
        flow.sum_over(("x",))  # type: ignore[ty:invalid-argument-type]
        flow.sum_to(("t", "e"))
        flow.sum_to(("t", "x"))  # type: ignore[ty:invalid-argument-type]
        flow.get_shares_over(("e",))
        flow.get_shares_over(("x",))  # type: ignore[ty:invalid-argument-type]
        flow.cumsum("t")
        flow.cumsum("x")  # type: ignore[ty:invalid-argument-type]
        flow.to_df(dim_to_columns="x")  # type: ignore[ty:invalid-argument-type]

        # results of operations keep the letters
        recycled: FlodymArray[Literal["t", "e"]] = flow * self.parameters["shredder yield"]
        recycled.sum_over(("e",))
        recycled.sum_over(("x",))  # type: ignore[ty:invalid-argument-type]
        (flow * 0.5).sum_over(("x",))  # type: ignore[ty:invalid-argument-type]
        (1 - flow).sum_over(("x",))  # type: ignore[ty:invalid-argument-type]
        flow[{"e": "Fe"}].sum_over(("x",))  # type: ignore[ty:invalid-argument-type]

        self.dims["t"]
        self.dims["x"]  # type: ignore[ty:invalid-argument-type]
        self.dims.get_subset(("t", "x"))  # type: ignore[ty:invalid-argument-type]
        self.get_new_array(("t", "e"))
        self.get_new_array(("t", "x"))  # type: ignore[ty:invalid-argument-type]
        # TODO: The type of the new array should be FlodymArray[Literal["t"]]
        new_array: FlodymArray[Literal["t", "e"]] = self.get_new_array(("t",))


# Check: without a type argument, any letter is accepted
def sum_over_any_letter(flow: Flow) -> FlodymArray:
    return flow.sum_over(("x",))


# Check: creating a MFASystem with explicitly typed dimensions
dimensions = DimensionSet[Letters](
    dim_list=(
        Dimension(name="time", letter="t", items=("2024",)),
        Dimension(name="region", letter="r", items=("Europe",)),
    )
)
mfasytem = MFASystem[Letters](dims=dimensions, flows={}, parameters={}, processes={})

# Check: catching typos in dimensions declaration
# TODO: This is not yet caught by the type checker
dimensions_with_error = DimensionSet[Letters](
    dim_list=(
        Dimension(name="time", letter="w", items=("2024",)),
        Dimension(name="region", letter="b", items=("Europe",)),
    )
)
mfasytem_with_error = MFASystem[Letters](
    dims=dimensions_with_error, flows={}, parameters={}, processes={}
)
# But this is caught by the type checker
dimension_with_error = Dimension[Letters](name="time", letter="w", items=("2024",))  # type: ignore[ty:invalid-argument-type]
