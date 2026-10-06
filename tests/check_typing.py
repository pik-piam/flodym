from numbers import Number
from typing import SupportsFloat

import numpy as np

from flodym import FlodymArray


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

# Check: shares can be taken over a single dimension or a tuple of dimensions
array_shares: FlodymArray = array.get_shares_over("a")
array_shares_tuple: FlodymArray = array.get_shares_over(("a",))
