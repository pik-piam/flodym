import pytest
from pydantic_core import ValidationError

from flodym import DimensionDefinition, MFADefinition, SimpleFlowDrivenStock


def test_validate_dimension_definition():
    # example that raises a validation error
    with pytest.raises(ValidationError) as error_msg:
        DimensionDefinition(name="something", letter="too_long", dtype=str)
    assert "letter" in str(error_msg.value)

    # example that does not raise an error
    DimensionDefinition(name="something", letter="s", dtype=str)


def test_validate_mfa_definition():
    # example with no dimensions defined
    with pytest.raises(ValidationError) as error_msg:
        MFADefinition(
            dimensions=[],
            processes=[],
            flows=[],
            stocks=[
                {"name": "stock_zero", "dim_letters": ("t"), "subclass": SimpleFlowDrivenStock},
            ],
            parameters=[],
        )
    assert "dim_letters" in str(error_msg.value)

    dimensions = [
        {"name": "time", "letter": "t", "dtype": int},
        {"name": "place", "letter": "p", "dtype": str},
    ]

    # example with mismatching dimensions between attributes
    with pytest.raises(ValidationError) as error_msg:
        MFADefinition(
            dimensions=dimensions,
            processes=[],
            flows=[],
            stocks=[
                {"name": "bad_stock", "dim_letters": ("t", "x"), "subclass": SimpleFlowDrivenStock},
            ],
            parameters=[],
        )
    assert "bad_stock" in str(error_msg.value)

    # example with valid dimensions in stock
    for letter_combinations in ("t", "p"), ("t",), ("p",):
        MFADefinition(
            dimensions=dimensions,
            processes=[],
            flows=[],
            stocks=[
                {
                    "name": "good_stock",
                    "dim_letters": letter_combinations,
                    "subclass": SimpleFlowDrivenStock,
                },
            ],
            parameters=[],
        )


def test_dimension_definition_items():
    definition = DimensionDefinition(name="Time", letter="t", dtype=int, items=[2020, 2030])
    assert definition.items == [2020, 2030]
    DimensionDefinition(
        name="Country", letter="c", dtype=str, parent="w", items=["DEU"], parent_items=["EUR"]
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"items": ["2020"]}, "datatype int"),
        ({"parent_items": [1]}, "parent_items, but no items"),
        ({"items": [1], "parent": "w"}, "if and only if it has a parent"),
        ({"items": [1], "parent_items": [1]}, "if and only if it has a parent"),
        ({"items": [1, 2], "parent": "w", "parent_items": [1]}, "same length"),
        ({"items": [1], "parent": "w", "splitting": {1: "x"}}, "splitting or combine_splits"),
        ({"items": [1], "combine_splits": ("x", "y")}, "splitting or combine_splits"),
    ],
)
def test_invalid_dimension_definition_items(kwargs, match):
    with pytest.raises(ValidationError, match=match):
        DimensionDefinition(name="Something", letter="s", dtype=int, **kwargs)
