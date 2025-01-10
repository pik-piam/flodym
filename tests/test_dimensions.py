from pydantic_core import ValidationError
import pytest

from flodym import DimensionSet


def test_validate_dimension_set():
    # example valid DimensionSet
    dimensions = [
        {"name": "time", "letter": "t", "items": [1990, 2000, 2010]},
        {"name": "place", "letter": "p", "items": ["World"]},
    ]
    DimensionSet(dim_list=dimensions)

    # example with repeated dimension letters in DimensionSet
    dimensions.append({"name": "another_time", "letter": "t", "items": [2020, 2030]})
    with pytest.raises(ValidationError) as error_msg:
        DimensionSet(dim_list=dimensions)
    assert "letter" in str(error_msg.value)


def test_get_subset():
    subset_dimensions = [
        {"name": "time", "letter": "t", "items": [1990, 2000, 2010]},
        {"name": "place", "letter": "p", "items": ["World"]},
    ]
    material_dimension = {"name": "material", "letter": "m", "items": ["material_0", "material_1"]}

    parent_dimensions = subset_dimensions + [material_dimension]
    dimension_set = DimensionSet(dim_list=parent_dimensions)

    # example of subsetting the dimension set using dimension letters
    subset_from_letters = dimension_set.get_subset(dims=("t", "p"))
    assert subset_from_letters == DimensionSet(dim_list=subset_dimensions)

    # example of subsetting the dimension set using dimension names
    subset_from_names = dimension_set.get_subset(dims=("time", "place"))
    assert subset_from_names == subset_from_letters

    # example where the requested subset dimension doesn't exist
    with pytest.raises(KeyError):
        dimension_set.get_subset(dims=("s", "p"))
