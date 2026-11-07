"""Tests for the extensive flag of FlodymArrays, its propagation and its use in operations."""

import logging
import operator

import numpy as np
import pytest

from flodym import (
    DataReader,
    Dimension,
    DimensionDefinition,
    DimensionSet,
    FlodymArray,
    Flow,
    FlowDefinition,
    MFADefinition,
    MFASystem,
    Parameter,
    ParameterDefinition,
    Process,
    ProcessDefinition,
    StockArray,
    config,
)
from flodym.example_objects import get_example_mfa
from flodym.flodym_array_helper import flodym_array_stack

TIME = Dimension(name="Time", letter="t", items=[2000, 2001, 2002])
REGION = Dimension(name="Region", letter="r", items=["north", "south"])
PLACE = Dimension(name="Place", letter="p", items=["World"])
DIMS = DimensionSet(dim_list=[TIME, REGION, PLACE])
E, I, U = True, False, None


def make_array(letters, extensive=None, cls=FlodymArray, value=1.0, **kwargs):
    dims = DIMS.get_subset(letters)
    return cls(dims=dims, values=np.full(dims.shape, value), extensive=extensive, **kwargs)


def make_flow(letters, value=1.0):
    dims = DIMS.get_subset(letters)
    processes = {"from_process": Process(name="a", id=1), "to_process": Process(name="b", id=2)}
    return Flow(dims=dims, values=np.full(dims.shape, value), **processes)


@pytest.fixture
def strict():
    """Raise on any extensivity report."""
    with config.override(error_behaviors={"extensivity": "error"}):
        yield


def test_defaults():
    dims = DIMS.get_subset(("t",))
    assert FlodymArray(dims=dims).extensive is None
    assert Parameter(dims=dims).extensive is None
    assert StockArray(dims=dims).extensive is True
    assert make_flow(("t",)).extensive is True
    assert ParameterDefinition(name="p", dim_letters=("t",)).extensive is None


@pytest.mark.parametrize(
    "first, second, op, expected",
    [
        (E, I, operator.mul, E),
        (I, I, operator.mul, I),
        (E, E, operator.truediv, I),
        (E, I, operator.truediv, E),
        (I, E, operator.truediv, U),
        (E, E, operator.mul, U),
        (E, U, operator.mul, U),
        (E, E, operator.add, E),
        (U, I, operator.add, I),
        (I, U, operator.sub, I),
        (U, U, operator.add, U),
    ],
)
def test_binary_propagation(strict, first, second, op, expected):
    result = op(make_array(("t",), first), make_array(("t",), second))
    assert result.extensive is expected


def test_higher_degree_is_tracked(strict):
    a, b, c = (make_array(("t",), E) for _ in range(3))
    product = a * b
    assert product.extensive is None
    assert product._degree == 2
    assert product.copy()._degree == 2
    assert (product / c).extensive is True
    assert (a * (b / c)).extensive is True
    product.extensive = False
    assert product._degree == 0


def test_scalar_operands(strict):
    extensive = make_array(("t",), E)
    intensive = make_array(("t",), I)
    assert (extensive + 1).extensive is True
    assert (1 - intensive).extensive is False
    assert (extensive * 2).extensive is True
    assert (extensive / 2).extensive is True
    assert (1 / extensive)._degree == -1
    assert (1 / intensive).extensive is False
    assert (intensive**2).extensive is False
    assert (extensive**2)._degree == 2
    assert (extensive**0.5)._degree is None
    assert (intensive ** make_array(("t",), I)).extensive is False


def test_unary_operations(strict):
    extensive = make_array(("t", "r"), E)
    assert (-extensive).extensive is True
    assert abs(extensive).extensive is True
    assert extensive.abs().extensive is True
    assert extensive.sign().extensive is False
    assert extensive.cumsum("t").extensive is True
    assert extensive.apply(np.sqrt).extensive is None
    extensive.apply(np.sqrt, inplace=True)
    assert extensive.extensive is True
    assert extensive[{"r": "north"}].extensive is True
    assert all(a.extensive for a in extensive.split("r").values())


def test_intensive_addition_broadcasts(strict):
    by_time = make_array(("t",), I, value=1.0)
    by_region = make_array(("r",), I, value=2.0)
    for result in (by_time + by_region, by_region + by_time):
        assert set(result.dims.letters) == {"t", "r"}
        np.testing.assert_allclose(result.values, 3.0)
        assert result.extensive is False
    result = by_time.maximum(by_region)
    assert set(result.dims.letters) == {"t", "r"}
    np.testing.assert_allclose(result.values, 2.0)


def test_extensive_addition_sums(strict):
    result = make_array(("t", "r"), E) + make_array(("t",), E)
    assert result.dims.letters == ("t",)
    np.testing.assert_allclose(result.values, 3.0)


def test_unknown_addition_keeps_convention():
    result = make_array(("t", "r")) + make_array(("t",))
    assert result.dims.letters == ("t",)
    np.testing.assert_allclose(result.values, 3.0)


def test_mixed_addition_is_reported(caplog):
    with caplog.at_level(logging.WARNING):
        result = make_array(("t",), E) + make_array(("t",), I)
    assert "same extensivity" in caplog.text
    assert result.extensive is None
    with config.override(error_behaviors={"extensivity": "error"}):
        with pytest.raises(ValueError, match="same extensivity"):
            make_array(("t",), E) - make_array(("t",), I)


def test_explicit_sum_of_intensive_is_reported(caplog):
    intensive = make_array(("t", "r"), I)
    with caplog.at_level(logging.WARNING):
        result = intensive.sum_to(("t",))
    assert "is summed over the dimensions ['Region']" in caplog.text
    np.testing.assert_allclose(result.values, 2.0)
    assert result.extensive is False
    with config.override(error_behaviors={"extensivity": "error"}):
        with pytest.raises(ValueError, match="summed over"):
            intensive.sum_over(("r",))


def test_explicit_sum_not_reported(strict):
    make_array(("t", "r"), E).sum_to(("t",))
    make_array(("t", "r"), U).sum_over(("r",))
    # dimensions with a single item are ignored
    make_array(("t", "p"), I).sum_to(("t",))


def test_explicit_cast_of_extensive_is_reported(caplog):
    with caplog.at_level(logging.WARNING):
        result = make_array(("t",), E).cast_to(DIMS.get_subset(("t", "r")))
    assert "is copied along the dimensions ['Region']" in caplog.text
    assert result.extensive is True
    with config.override(error_behaviors={"extensivity": "error"}):
        with pytest.raises(ValueError, match="copied along"):
            make_flow(("t",)).cast_to(DIMS.get_subset(("t", "r")), inplace=True)


def test_explicit_cast_not_reported(strict):
    make_array(("t",), I).cast_to(DIMS.get_subset(("t", "r")))
    make_array(("t",), U).cast_to(DIMS.get_subset(("t", "r")))
    make_array(("t",), E).cast_to(DIMS.get_subset(("t", "p")))


def test_assignment_of_intensive_broadcasts(strict):
    target = make_array(("t", "r"), value=0.0)
    source = FlodymArray(dims=DIMS.get_subset(("r",)), values=np.array([0.2, 0.8]), extensive=I)
    target[...] = source
    np.testing.assert_allclose(target.values, [[0.2, 0.8]] * 3)
    assert target.extensive is False
    # slices
    target[{"t": 2000}] = make_array((), I, value=0.5)
    np.testing.assert_allclose(target.values[0], 0.5)


def test_assignment_with_extra_intensive_dims_is_reported():
    target = make_array(("t",), I)
    with config.override(error_behaviors={"extensivity": "error"}):
        with pytest.raises(ValueError, match="summed over"):
            target[...] = make_array(("t", "r"), I)


def test_assignment_of_not_intensive_with_missing_dims_raises():
    for extensive in (E, U):
        with pytest.raises(ValueError, match="does not have"):
            make_flow(("t", "r"))[...] = make_array(("t",), extensive)


def test_assignment_adopts_and_checks_degree(caplog, strict):
    target = make_array(("t",))
    target[...] = make_array(("t", "r"), E)
    np.testing.assert_allclose(target.values, 2.0)
    assert target.extensive is True
    with config.override(error_behaviors={"extensivity": "warn"}), caplog.at_level(logging.WARNING):
        make_flow(("t",))[...] = make_array(("t",), I)
    assert "is assigned to Flow" in caplog.text


def test_to_class(caplog, strict):
    assert make_array(("t",), I, cls=Parameter).to_class(FlodymArray).extensive is False
    processes = {"from_process": Process(name="a", id=1), "to_process": Process(name="b", id=2)}
    assert make_array(("t",)).to_class(Flow, **processes).extensive is True
    assert make_array(("t",), I).to_class(Flow, extensive=True, **processes).extensive is True
    assert make_array(("t",), E).to_class(Parameter).extensive is True
    with config.override(error_behaviors={"extensivity": "warn"}), caplog.at_level(logging.WARNING):
        Flow.from_array(make_array(("t",), I), **processes)
    assert "is converted to Flow" in caplog.text


def test_shares_are_intensive(strict):
    extensive = make_array(("t", "r"), E)
    assert extensive.get_shares_over("r").extensive is False
    assert extensive.get_shares_over(("t", "r")).extensive is False


@pytest.mark.parametrize("inplace", [False, True])
def test_stack_of_flows(strict, inplace):
    flows = [make_flow(("t",), value=float(i)) for i in range(2)]
    stacked = flodym_array_stack(flows, REGION, inplace=inplace)
    assert stacked.extensive is True
    np.testing.assert_allclose(stacked.values, [[0.0, 1.0]] * 3)
    assert (stacked is flows[0]) == inplace


class DictReader(DataReader):
    def __init__(self, parameter_values):
        self.parameter_values = parameter_values

    def read_dimension(self, definition):
        return {"Time": TIME, "Region": REGION}[definition.name]

    def read_parameter_values(self, parameter_name, dims):
        values = np.broadcast_to(self.parameter_values[parameter_name], dims.shape).copy()
        return Parameter(dims=dims, values=values, name=parameter_name)


def flow_definition(from_process, to_process, dims=("t",), **kwargs):
    return FlowDefinition(
        from_process_name=from_process, to_process_name=to_process, dim_letters=dims, **kwargs
    )


def split_mfa(parameter_definitions=None, **process_kwargs):
    """sysenv -> a -> (b by region, c by region), with a share and a forward splitter."""
    definition = MFADefinition(
        dimensions=[
            DimensionDefinition(name="Time", letter="t", dtype=int),
            DimensionDefinition(name="Region", letter="r", dtype=str),
        ],
        processes=[
            "sysenv",
            ProcessDefinition(
                name="a",
                outflow_shares={"b": "share"},
                forward_dimension_splitter="split",
                **process_kwargs,
            ),
            "b",
            "c",
        ],
        flows=[
            flow_definition("sysenv", "a", prescribed="input"),
            flow_definition("a", "b", dims=("t", "r")),
            flow_definition("a", "c", dims=("t", "r")),
            flow_definition("b", "sysenv", dims=("t", "r")),
            flow_definition("c", "sysenv", dims=("t", "r")),
        ],
        parameters=parameter_definitions
        or [
            ParameterDefinition(name="input", dim_letters=("t",)),
            ParameterDefinition(name="share", dim_letters=()),
            ParameterDefinition(name="split", dim_letters=("r",)),
        ],
    )
    reader = DictReader({"input": np.array([1.0, 2.0, 3.0]), "share": 0.3, "split": [0.4, 0.6]})
    return MFASystem.from_data_reader(definition, reader)


def test_link_parameters_sets_extensivity(strict):
    mfa = split_mfa()
    assert mfa.parameters["input"].extensive is True
    assert mfa.parameters["share"].extensive is False
    assert mfa.parameters["split"].extensive is False
    assert mfa.compute_all_possible() == []
    np.testing.assert_allclose(mfa.flows["a => b"].values, np.outer([0.3, 0.6, 0.9], [0.4, 0.6]))
    assert mfa.check_mass_balance()


def test_parameter_definition_extensivity_is_read_and_checked(caplog):
    parameters = [
        ParameterDefinition(name="input", dim_letters=("t",)),
        ParameterDefinition(name="share", dim_letters=(), extensive=True),
        ParameterDefinition(name="split", dim_letters=("r",), extensive=False),
    ]
    with caplog.at_level(logging.WARNING):
        mfa = split_mfa(parameter_definitions=parameters)
    assert "Parameter 'share' has extensive=True, but is used intensive" in caplog.text
    assert mfa.parameters["share"].extensive is True
    assert mfa.parameters["split"].extensive is False


def test_conflicting_parameter_uses_are_reported():
    definition = MFADefinition(
        dimensions=[DimensionDefinition(name="Time", letter="t", dtype=int)],
        processes=["sysenv", ProcessDefinition(name="a", outflow_shares={"b": "p"}), "b"],
        flows=[
            flow_definition("sysenv", "a", prescribed="p"),
            flow_definition("a", "b"),
            flow_definition("a", "sysenv"),
        ],
        parameters=[ParameterDefinition(name="p", dim_letters=("t",))],
    )
    with config.override(error_behaviors={"extensivity": "error"}):
        with pytest.raises(ValueError, match="used with different extensivity"):
            MFASystem.from_data_reader(definition, DictReader({"p": 0.5}))


def test_example_mfa_without_reports(strict):
    mfa = get_example_mfa()
    mfa.compute()
    assert mfa.check_mass_balance()
