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


def test_index_with_letters_and_names():
    """Test that index() method accepts both letters and names."""
    dimensions = [
        {"name": "time", "letter": "t", "items": [1990, 2000, 2010]},
        {"name": "place", "letter": "p", "items": ["World"]},
        {"name": "material", "letter": "m", "items": ["steel", "aluminum"]},
    ]
    dimension_set = DimensionSet(dim_list=dimensions)

    # Test with letters
    assert dimension_set.index("t") == 0
    assert dimension_set.index("p") == 1
    assert dimension_set.index("m") == 2

    # Test with names
    assert dimension_set.index("time") == 0
    assert dimension_set.index("place") == 1
    assert dimension_set.index("material") == 2

    # Test with non-existent key
    with pytest.raises(KeyError):
        dimension_set.index("nonexistent")


def test_size_with_letters_and_names():
    """Test that size() method accepts both letters and names."""
    dimensions = [
        {"name": "time", "letter": "t", "items": [1990, 2000, 2010]},
        {"name": "place", "letter": "p", "items": ["World"]},
    ]
    dimension_set = DimensionSet(dim_list=dimensions)

    # Test with letters
    assert dimension_set.size("t") == 3
    assert dimension_set.size("p") == 1

    # Test with names
    assert dimension_set.size("time") == 3
    assert dimension_set.size("place") == 1


def test_drop_with_letters_and_names():
    """Test that drop() method accepts both letters and names."""
    dimensions = [
        {"name": "time", "letter": "t", "items": [1990, 2000, 2010]},
        {"name": "place", "letter": "p", "items": ["World"]},
        {"name": "material", "letter": "m", "items": ["steel", "aluminum"]},
    ]
    dimension_set = DimensionSet(dim_list=dimensions)

    # Test drop with letter
    dropped = dimension_set.drop("t")
    assert dropped.letters == ("p", "m")
    assert dropped.names == ("place", "material")

    # Test drop with name
    dropped = dimension_set.drop("place")
    assert dropped.letters == ("t", "m")
    assert dropped.names == ("time", "material")


def test_replace_with_letters_and_names():
    """Test that replace() method accepts both letters and names."""
    from flodym import Dimension

    dimensions = [
        {"name": "time", "letter": "t", "items": [1990, 2000, 2010]},
        {"name": "place", "letter": "p", "items": ["World"]},
    ]
    dimension_set = DimensionSet(dim_list=dimensions)

    new_dim = Dimension(name="NewDim", letter="n", items=[1, 2, 3])

    # Test replace with letter
    replaced = dimension_set.replace("t", new_dim)
    assert replaced.names == ("NewDim", "place")
    assert replaced.letters == ("n", "p")

    # Test replace with name
    replaced = dimension_set.replace("place", new_dim)
    assert replaced.names == ("time", "NewDim")
    assert replaced.letters == ("t", "n")
