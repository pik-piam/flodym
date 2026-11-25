"""Tests for MFASystem.from_definition and MFASystem.set_parameter_values."""

import numpy as np
import pandas as pd
import pytest

import flodym as fd

TIME = fd.DimensionDefinition(name="Time", letter="t", dtype=int, items=[2020, 2030])
REGION = fd.DimensionDefinition(name="Region", letter="r", dtype=str)
SUPPLY = np.array([[1.0, 2.0], [3.0, 4.0]])


def make_definition(dimensions=None) -> fd.MFADefinition:
    """System sysenv -> a -> b / c -> sysenv, with prescribed input and a share of a -> b."""
    return fd.MFADefinition(
        dimensions=dimensions or [TIME, REGION],
        processes=[
            "sysenv",
            fd.ProcessDefinition(name="a", outflow_shares={"b": "share b"}),
            "b",
            "c",
        ],
        flows=[
            fd.FlowDefinition(
                from_process_name="sysenv",
                to_process_name="a",
                dim_letters=("t", "r"),
                prescribed="supply",
            ),
            fd.FlowDefinition(from_process_name="a", to_process_name="b", dim_letters=("t", "r")),
            fd.FlowDefinition(from_process_name="a", to_process_name="c", dim_letters=("t", "r")),
            fd.FlowDefinition(
                from_process_name="b", to_process_name="sysenv", dim_letters=("t", "r")
            ),
            fd.FlowDefinition(
                from_process_name="c", to_process_name="sysenv", dim_letters=("t", "r")
            ),
        ],
        parameters=[
            fd.ParameterDefinition(name="supply", dim_letters=("t", "r")),
            fd.ParameterDefinition(name="share b", dim_letters=("r",)),
        ],
    )


class _Reader(fd.DataReader):
    """Reader returning fixed regions and parameter values."""

    def read_dimension(self, definition):
        return fd.Dimension(
            name=definition.name, letter=definition.letter, items=["EUR", "USA"], dtype=str
        )

    def read_parameter_values(self, parameter_name, dims):
        values = {"supply": SUPPLY, "share b": np.array([0.25, 0.5])}[parameter_name]
        return fd.Parameter(name=parameter_name, dims=dims, values=values)


# Dimensions ------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["r", "Region"])
def test_dimension_items_by_letter_or_name(key):
    mfa = fd.MFASystem.from_definition(make_definition(), {key: ["EUR", "USA"]})
    assert mfa.dims["t"].items == [2020, 2030]
    assert mfa.dims["r"].items == ["EUR", "USA"]
    assert mfa.dims.letters == ("t", "r")


def test_dimension_items_from_callable():
    mfa = fd.MFASystem.from_definition(make_definition(), _Reader().read_dimension)
    assert mfa.dims["r"].items == ["EUR", "USA"]


def test_all_items_in_definition():
    region = REGION.model_copy(update={"items": ["EUR"]})
    mfa = fd.MFASystem.from_definition(make_definition([TIME, region]))
    assert mfa.dims["r"].items == ["EUR"]


def test_invalid_dimension_items_are_listed_together():
    definition = make_definition()
    with pytest.raises(ValueError) as error:
        fd.MFASystem.from_definition(definition, {"x": [1], "t": [2020], "Unknown": [1]})
    message = str(error.value)
    assert "'x' is neither" in message
    assert "'Unknown' is neither" in message
    assert "'Time' has items in its definition already" in message
    assert "No items are given for the dimensions ['Region']" in message


def test_dimension_given_twice():
    with pytest.raises(ValueError, match="given twice"):
        fd.MFASystem.from_definition(make_definition(), {"r": ["EUR"], "Region": ["EUR"]})


def test_items_missing_without_dimension_items():
    with pytest.raises(ValueError, match=r"No items are given for the dimensions \['Region'\]"):
        fd.MFASystem.from_definition(make_definition())


def test_list_for_dimension_without_parent_required():
    with pytest.raises(ValueError, match="must be given as a list"):
        fd.MFASystem.from_definition(make_definition(), {"r": {"EUR": "World"}})


# Sub-dimensions --------------------------------------------------------------------------------


def sub_dimension_definitions():
    return [
        fd.DimensionDefinition(name="World region", letter="w", dtype=str, items=["EUR", "USA"]),
        fd.DimensionDefinition(name="Country", letter="c", dtype=str, parent="w"),
        fd.DimensionDefinition(name="Building type", letter="b", dtype=str),
        fd.DimensionDefinition(
            name="Residential type", letter="e", dtype=str, items=["Sin", "Mul"]
        ),
        fd.DimensionDefinition(
            name="Building sub-type", letter="s", dtype=str, parent="b", splitting={"Res": "e"}
        ),
    ]


def sub_dimension_system(dimension_items):
    definition = fd.MFADefinition(
        dimensions=sub_dimension_definitions(), processes=["sysenv"], flows=[]
    )
    return fd.MFASystem.from_definition(definition, dimension_items)


def test_sub_dimensions_from_items():
    mfa = sub_dimension_system(
        {"c": {"DEU": "EUR", "FRA": "EUR", "USA": "USA"}, "b": ["Com", "Res"]}
    )
    countries = mfa.dims["c"]
    assert countries.items == ["DEU", "FRA", "USA"]
    assert countries.parent is mfa.dims["w"]
    assert countries.parent_items == ["EUR", "EUR", "USA"]
    assert mfa.dims["s"].items == ["Com", "Res|Sin", "Res|Mul"]
    assert mfa.dims["s"].parent is mfa.dims["b"]


def test_sub_dimension_needs_mapping():
    with pytest.raises(ValueError, match="mapping from each item to its parent item"):
        sub_dimension_system({"c": ["DEU"], "b": ["Com", "Res"]})


def test_split_dimension_cannot_get_items():
    with pytest.raises(ValueError, match="created by splitting"):
        sub_dimension_system({"c": {"USA": "USA", "DEU": "EUR"}, "b": ["Com", "Res"], "s": ["Com"]})


def test_parent_item_not_in_parent():
    with pytest.raises(ValueError, match="not items of the parent"):
        sub_dimension_system({"c": {"DEU": "EUR", "USA": "USA", "JPN": "ASI"}, "b": ["Res"]})


def test_parent_items_in_definition():
    definitions = sub_dimension_definitions()
    definitions[1] = definitions[1].model_copy(
        update={"items": ["DEU", "USA"], "parent_items": ["EUR", "USA"]}
    )
    definition = fd.MFADefinition(dimensions=definitions, processes=["sysenv"], flows=[])
    mfa = fd.MFASystem.from_definition(definition, {"b": ["Com", "Res"]})
    assert mfa.dims["c"].parent_items == ["EUR", "USA"]
    assert mfa.dims["c"].parent is mfa.dims["w"]


def test_reader_not_called_for_items_in_definition():
    class _FailingReader(_Reader):
        def read_dimension(self, definition):
            assert definition.letter == "r", f"Reader called for '{definition.name}'"
            return super().read_dimension(definition)

    dims = _FailingReader().read_dimensions([TIME, REGION])
    assert dims["t"].items == [2020, 2030]


# Parameters ------------------------------------------------------------------------------------


@pytest.fixture
def mfa():
    return fd.MFASystem.from_definition(make_definition(), {"r": ["EUR", "USA"]})


def test_parameters_unset_and_linked(mfa):
    supply = mfa.parameters["supply"]
    assert not supply.is_set
    assert np.all(supply.values == 0.0)
    assert mfa.flows["sysenv => a"]._prescribed is supply
    assert mfa.processes["a"].outflow_shares["a => b"] is mfa.parameters["share b"]


def test_unset_parameter_raises_in_computation(mfa):
    with pytest.raises(ValueError, match="Parameter 'supply' is used in"):
        mfa.compute_all_possible()


def assert_computed(mfa):
    mfa.compute_all_possible()
    assert np.allclose(mfa.flows["a => b"].values, SUPPLY * np.array([0.25, 0.5]))
    assert np.allclose(mfa.flows["a => c"].values, SUPPLY * np.array([0.75, 0.5]))


def test_set_values_from_arrays_and_scalars(mfa):
    supply = mfa.parameters["supply"]
    mfa.set_parameter_values({"supply": SUPPLY, "share b": [0.25, 0.5]})
    assert mfa.parameters["supply"] is supply
    assert supply.is_set
    assert_computed(mfa)


def test_set_values_from_flodym_array_and_df(mfa):
    share = fd.FlodymArray(dims=mfa.dims["r",], values=np.array([0.25, 0.5]))
    supply_df = pd.DataFrame(
        {
            "Time": [2020, 2020, 2030, 2030],
            "Region": ["EUR", "USA", "EUR", "USA"],
            "value": SUPPLY.flatten(),
        }
    )
    mfa.set_parameter_values({"supply": supply_df, "share b": share})
    assert_computed(mfa)


def test_set_values_from_callable(mfa):
    mfa.set_parameter_values({"share b": [0.25, 0.5]})
    called = []

    def read(name, dims):
        called.append(name)
        return _Reader().read_parameter_values(name, dims)

    mfa.set_parameter_values(read)
    assert called == ["supply"]
    assert_computed(mfa)


def test_same_result_as_from_data_reader(mfa):
    mfa.set_parameter_values(_Reader().read_parameter_values)
    mfa.compute_all_possible()
    read_mfa = fd.MFASystem.from_data_reader(make_definition(), _Reader())
    read_mfa.compute_all_possible()
    for name, flow in read_mfa.flows.items():
        assert np.allclose(mfa.flows[name].values, flow.values)


def test_unknown_parameter_names(mfa):
    with pytest.raises(ValueError, match=r"unknown parameters: \['nope'\]"):
        mfa.set_parameter_values({"nope": 1.0})


def test_share_values_checked_when_set(mfa):
    with (
        fd.config.override(error_behaviors={"process_shares": "error"}),
        pytest.raises(ValueError, match="not between zero and one"),
    ):
        mfa.set_parameter_values({"share b": [0.5, 1.5]})


def test_unset_share_raises_in_computation(mfa):
    mfa.set_parameter_values({"supply": SUPPLY})
    with pytest.raises(ValueError, match="Parameter 'share b' is used in"):
        mfa.compute_all_possible()
