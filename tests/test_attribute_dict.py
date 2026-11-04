import copy
import logging
import pickle

import numpy as np
import pytest

from flodym import (
    AttributeDict,
    FlowDefinition,
    SimpleFlowDrivenStock,
    StockDefinition,
    make_empty_flows,
    make_empty_stocks,
)
from flodym._attribute_dict import _to_attribute_name
from flodym.example_objects import get_example_mfa


@pytest.mark.parametrize(
    "key, expected",
    [
        ("foo", "foo"),
        ("sysenv => process 1", "sysenv__to__process_1"),
        ("a=>b", "a__to__b"),
        ("slag piles", "slag_piles"),
        ("a - b", "a_b"),
        ("1st", "_1st"),
        ("class", "class_"),
        ("", "_"),
    ],
)
def test_to_attribute_name(key, expected):
    assert _to_attribute_name(key) == expected
    assert expected.isidentifier()


def test_attribute_access():
    attribute_dict = AttributeDict({"foo": 1, "sysenv => process 1": 2})
    assert isinstance(attribute_dict, dict)
    assert attribute_dict.foo == 1
    assert attribute_dict.sysenv__to__process_1 == 2
    assert attribute_dict["sysenv => process 1"] == 2


def test_missing_attribute():
    attribute_dict = AttributeDict({"foo": 1})
    with pytest.raises(AttributeError):
        attribute_dict.bar
    assert not hasattr(attribute_dict, "bar")


def test_alias_collision(caplog):
    attribute_dict = AttributeDict({"a b": 1, "a_b": 2, "c": 3})
    with caplog.at_level(logging.WARNING):
        assert attribute_dict.c == 3
    assert "a_b" in caplog.text
    with pytest.raises(AttributeError):
        attribute_dict.a_b
    assert attribute_dict["a b"] == 1
    assert attribute_dict["a_b"] == 2


def test_dict_method_wins(caplog):
    attribute_dict = AttributeDict({"values": 1, "foo": 2})
    with caplog.at_level(logging.WARNING):
        assert attribute_dict.foo == 2
    assert "values" in caplog.text
    assert callable(attribute_dict.values)
    assert attribute_dict["values"] == 1


def test_warning_only_on_rebuild(caplog):
    attribute_dict = AttributeDict({"items": 1, "foo": 2})
    attribute_dict.foo
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        attribute_dict.foo
    assert caplog.text == ""


def test_no_attribute_assignment():
    attribute_dict = AttributeDict({"foo": 1})
    with pytest.raises(AttributeError):
        attribute_dict.foo = 2
    with pytest.raises(AttributeError):
        attribute_dict.bar = 2
    with pytest.raises(AttributeError):
        del attribute_dict.foo
    assert attribute_dict == {"foo": 1}


def test_aliases_follow_item_changes():
    attribute_dict = AttributeDict({"foo": 1})
    assert attribute_dict.foo == 1
    attribute_dict["bar baz"] = 2
    assert attribute_dict.bar_baz == 2
    del attribute_dict["foo"]
    with pytest.raises(AttributeError):
        attribute_dict.foo


def test_dir():
    attribute_dict = AttributeDict({"foo": 1, "a => b": 2})
    names = dir(attribute_dict)
    assert "foo" in names
    assert "a__to__b" in names
    assert "items" in names


def test_copy_and_pickle():
    attribute_dict = AttributeDict({"foo": [1], "a => b": 2})
    attribute_dict.foo  # fill the alias cache
    for copied in [copy.deepcopy(attribute_dict), pickle.loads(pickle.dumps(attribute_dict))]:
        assert isinstance(copied, AttributeDict)
        assert copied == attribute_dict
        assert copied.a__to__b == 2
        copied["new"] = 3
        assert copied.new == 3


def test_mfa_system_containers():
    mfa = get_example_mfa()
    for container in [mfa.flows, mfa.stocks, mfa.processes, mfa.parameters]:
        assert isinstance(container, AttributeDict)

    flow = mfa.flows["sysenv => shredder"]
    assert mfa.flows.sysenv__to__shredder is flow
    assert mfa.stocks.slag_piles is mfa.stocks["slag piles"]
    assert mfa.processes.sysenv is mfa.processes["sysenv"]

    mfa.flows.sysenv__to__shredder[...] = 2.0
    np.testing.assert_array_equal(flow.values, 2.0)


def test_make_empty_helpers_as_attribute_dict():
    mfa = get_example_mfa()
    flow_definitions = [
        FlowDefinition(from_process_name="sysenv", to_process_name="shredder", dim_letters=("t",))
    ]
    stock_definitions = [
        StockDefinition(
            name="slag piles",
            process="slag piles",
            dim_letters=("t",),
            subclass=SimpleFlowDrivenStock,
        )
    ]

    flows = make_empty_flows(mfa.processes, flow_definitions, mfa.dims)
    stocks = make_empty_stocks(stock_definitions, mfa.processes, mfa.dims)
    assert type(flows) is dict
    assert type(stocks) is dict

    flows = make_empty_flows(mfa.processes, flow_definitions, mfa.dims, as_attribute_dict=True)
    stocks = make_empty_stocks(
        stock_definitions, mfa.processes, mfa.dims, as_attribute_dict=True
    )
    assert isinstance(flows, AttributeDict)
    assert isinstance(stocks, AttributeDict)
    assert flows.sysenv__to__shredder is flows["sysenv => shredder"]
    assert stocks.slag_piles is stocks["slag piles"]
