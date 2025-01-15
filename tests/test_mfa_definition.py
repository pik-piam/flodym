from pydantic_core import ValidationError
import pytest

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
