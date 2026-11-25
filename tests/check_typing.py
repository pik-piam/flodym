from numbers import Number
from typing import SupportsFloat

import numpy as np

from flodym import (
    FlodymArray,
    LogNormalDistribution,
    MFASystem,
    MonteCarloResult,
    NormalDistribution,
    ParameterUncertainty,
    SampledArray,
    UncertaintyDefinition,
    make_model_function,
    run_monte_carlo,
)


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

# Check: Monte Carlo uncertainty propagation
uncertainty = UncertaintyDefinition(
    parameters={
        "a": NormalDistribution(relative_std=0.1, lower=0.0),
        "b": ParameterUncertainty(
            distribution=LogNormalDistribution(std=array), independent_over="a"
        ),
        "c": [
            NormalDistribution(std=1),
            ParameterUncertainty(distribution=NormalDistribution(std=2)),
        ],
    },
    normalize_over={"a": "a"},
)


def model(parameters: dict[str, FlodymArray]) -> dict[str, FlodymArray | float]:
    return {"a": parameters["a"], "total": 1.0}


result: MonteCarloResult = run_monte_carlo(
    model, {"a": array, "b": array}, uncertainty, n_samples=2, seed=0
)
part: MonteCarloResult = run_monte_carlo(
    model, {"a": array, "b": array}, uncertainty, sample_indices=range(2, 4), seed=0
)
joined: MonteCarloResult = MonteCarloResult.concat([result, part])
summed: SampledArray = result.outputs["a"].sum_over(("a",))
sliced: SampledArray = result.outputs["a"][{"a": "first"}]
mean: FlodymArray = result.outputs["a"].mean
bands: FlodymArray = result.outputs["a"].quantile([0.05, 0.95])
rejection_rate: float = result.rejection_rate
inputs: dict[str, SampledArray] = uncertainty.sample({"a": array, "b": array}, n_samples=2, seed=0)


def copper_content(mfa: MFASystem) -> FlodymArray:
    return mfa.flows["a => b"]


def make_system_model(mfa: MFASystem) -> None:
    system_model = make_model_function(
        mfa, track=["a => b"], derived={"copper": copper_content}, compute=MFASystem.compute
    )
    run_monte_carlo(system_model, mfa.parameters, uncertainty, n_samples=2)
