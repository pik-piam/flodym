import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from pydantic_core import ValidationError
import pytest
from copy import deepcopy

from flodym import FlodymArray, DimensionSet, Dimension


places = Dimension(name="place", letter="p", items=["Earth", "Sun", "Moon", "Venus"])
local_places = Dimension(name="local place", letter="l", items=["Earth"])
time = Dimension(name="time", letter="t", items=[1990, 2000, 2010])
historic_time = Dimension(name="historic time", letter="h", items=[1990, 2000])
animals = Dimension(name="animal", letter="a", items=["cat", "mouse"])

base_dim_list = [places, time]

dims = DimensionSet(dim_list=base_dim_list)
values = np.random.rand(4, 3)
numbers = FlodymArray(name="two", dims=dims, values=values)

dims_incl_animals = DimensionSet(dim_list=base_dim_list + [animals])
animal_values = np.random.rand(4, 3, 2)
space_animals = FlodymArray(name="space_animals", dims=dims_incl_animals, values=animal_values)


def test_flodym_array_validations():
    dims = DimensionSet(dim_list=[local_places, time])

    # example with values with the correct shape
    FlodymArray(name="numbers", dims=dims, values=np.array([[1, 2, 3]]))

    # example with dimensions reversed
    with pytest.raises(ValidationError):
        FlodymArray(name="numbers", dims=dims, values=np.array([[1], [2], [3]]))

    # example with too many values
    with pytest.raises(ValidationError):
        FlodymArray(name="numbers", dims=dims, values=np.array([[1, 2, 3, 4]]))

    # example with no values passed -> filled with zeros
    zero_values = FlodymArray(name="numbers", dims=dims)
    assert zero_values.values.shape == (1, 3)
    assert np.all([zero_values.values == 0])


def test_cast_to():
    # example of duplicating values along new axis (e.g. same number of cats and mice)
    casted_flodym_array = numbers.cast_to(target_dims=dims_incl_animals)
    assert casted_flodym_array.dims == dims_incl_animals
    assert casted_flodym_array.values.shape == (4, 3, 2)
    assert_almost_equal(np.sum(casted_flodym_array.values), 2 * np.sum(values))

    # example with differently ordered dimensions
    target_dims = DimensionSet(dim_list=[animals] + base_dim_list[::-1])
    casted_flodym_array = numbers.cast_to(target_dims=target_dims)
    assert casted_flodym_array.values.shape == (2, 3, 4)


def test_sum_to():
    # sum over one dimension
    summed_flodym_array = space_animals.sum_to(result_dims=("p", "t"))
    assert summed_flodym_array.dims == DimensionSet(dim_list=base_dim_list)
    assert_array_almost_equal(summed_flodym_array.values, np.sum(animal_values, axis=2))

    # sum over two dimensions
    summed_flodym_array = space_animals.sum_to(result_dims=("t"))
    assert_array_almost_equal(
        summed_flodym_array.values, np.sum(np.sum(animal_values, axis=2), axis=0)
    )

    # example attempt to get a resulting dimension that does not exist
    with pytest.raises(KeyError):
        space_animals.sum_to(result_dims=("s"))

    # example where dimensions to sum over are specified rather than the remaining dimensions
    summed_over = space_animals.sum_over(sum_over_dims=("p", "a"))
    assert_array_almost_equal(summed_over.values, summed_flodym_array.values)

    # example sum over dimension that doesn't exist
    with pytest.raises(KeyError):
        space_animals.sum_over(sum_over_dims=("s"))


def test_get_shares_over():
    # example of getting shares over one dimension
    shares = space_animals.get_shares_over(dim_letters=("p"))
    assert shares.dims == space_animals.dims
    wanted_values = np.einsum("pta,ta->pta", animal_values, 1 / np.sum(animal_values, axis=0))
    assert_array_almost_equal(shares.values, wanted_values)

    # example of getting shares over two dimensions
    shares = space_animals.get_shares_over(dim_letters=("p", "a"))
    wanted_values = np.einsum("pta,t->pta", animal_values, 1 / np.sum(animal_values, axis=(0, 2)))
    assert_array_almost_equal(shares.values, wanted_values)

    # example of getting shares over all dimensions
    shares = space_animals.get_shares_over(dim_letters=("p", "t", "a"))
    assert_array_almost_equal(shares.values, animal_values / np.sum(animal_values))

    # example of getting shares over a dimension that doesn't exist
    with pytest.raises(AssertionError):
        space_animals.get_shares_over(dim_letters=("s",))


def test_maths():
    # test minimum
    minimum = space_animals.minimum(numbers)
    assert minimum.dims == dims
    assert_array_almost_equal(minimum.values, np.minimum(values, animal_values.sum(axis=2)))

    # test maximum
    maximum = space_animals.maximum(numbers)
    assert maximum.dims == dims
    assert_array_almost_equal(maximum.values, np.maximum(values, animal_values.sum(axis=2)))

    # test sum
    summed = space_animals + numbers
    assert summed.dims == dims
    assert_array_almost_equal(summed.values, animal_values.sum(axis=2) + values)

    # test minus
    subtracted = space_animals - numbers
    assert subtracted.dims == dims
    assert_array_almost_equal(subtracted.values, animal_values.sum(axis=2) - values)
    subtracted_flipped = numbers - space_animals
    assert subtracted_flipped.dims == dims
    assert_array_almost_equal(subtracted_flipped.values, values - animal_values.sum(axis=2))

    # test multiply
    multiplied = numbers * space_animals
    assert multiplied.dims == dims_incl_animals  # different from behaviour of above methods
    assert_array_almost_equal(multiplied.values[:, :, 0], values * animal_values[:, :, 0])
    assert_array_almost_equal(multiplied.values[:, :, 1], values * animal_values[:, :, 1])

    # test divide
    divided = space_animals / numbers
    assert divided.dims == dims_incl_animals
    assert_array_almost_equal(divided.values[:, :, 0], animal_values[:, :, 0] / values)
    assert_array_almost_equal(divided.values[:, :, 1], animal_values[:, :, 1] / values)
    divided_flipped = numbers / space_animals
    assert divided_flipped.dims == dims_incl_animals
    assert_array_almost_equal(divided_flipped.values[:, :, 0], values / (animal_values[:, :, 0]))
    assert_array_almost_equal(divided_flipped.values[:, :, 1], values / (animal_values[:, :, 1]))


def test_get_item():
    cats_on_the_moon = space_animals["Moon"]["cat"]
    assert isinstance(cats_on_the_moon, FlodymArray)
    assert_array_almost_equal(cats_on_the_moon.values, space_animals.values[2, :, 0])
    # note that this does not work for the time dimension (not strings)
    # and also assumes that no item appears in more than one dimension


def test_sub_array_handler():
    space_cat = space_animals["cat"]  # space cat from str
    another_space_cat = space_animals[{"a": "cat"}]  # space cat from dict
    assert_array_equal(space_cat.values, another_space_cat.values)

    space_1990 = space_animals[{"t": 1990}]  # space animals in 1990
    assert space_1990.values.shape == (4, 2)
    assert space_1990.dims.letters == ("p", "a")

    with pytest.raises(ValueError):
        space_animals[{"a": "dog"}]  # there isn't a dog in space_animals


def test_dimension_subsets():
    historic_dims_incl_animals = DimensionSet(dim_list=[places, historic_time, animals])
    historic_space_animals = FlodymArray(dims=historic_dims_incl_animals)
    historic_space_animals[...] = space_animals[{"t": historic_time}]

    assert np.min(historic_space_animals.values) > 0.0

    space_animals_copy = FlodymArray(dims=dims_incl_animals)
    space_animals_copy[{"t": historic_time}] = 1.0 * historic_space_animals
    space_animals_copy[{"t": 2010}] = 1.0 * space_animals[{"t": 2010}]

    assert_array_equal(space_animals.values, space_animals_copy.values)

    wrong_historic_time = (Dimension(name="historic time", letter="t", items=[1990, 2000]),)
    with pytest.raises(ValueError):
        space_animals[{"t": wrong_historic_time}]  # same letter in original and subset

    another_wrong_historic_time = (Dimension(name="historic time", letter="p", items=[1990, 2000]),)
    with pytest.raises(ValueError):
        space_animals[
            {"t": another_wrong_historic_time}
        ]  # same letter in other dim of original and subset

    with pytest.raises(ValueError):
        historic_space_animals[{"h": time}]  # index is not a subset


def test_multiple_dimension_subsets():
    """Test slicing with multiple Dimension objects at once.
    
    This reproduces an issue where slicing with two Dimension objects causes
    a shape mismatch error due to NumPy's advanced indexing broadcasting behavior.
    """
    # Create a larger array to make the issue more apparent
    regions = Dimension(name="region", letter="r", items=["EUR", "USA", "CHN"])
    years = Dimension(name="time", letter="t", items=list(range(2020, 2030)))  # 10 years
    materials = Dimension(name="material", letter="m", items=["concrete", "steel", "wood"])
    applications = Dimension(name="application", letter="a", items=["A1", "A2", "A3", "A4", "A5", "A6"])
    
    full_dims = DimensionSet(dim_list=[regions, years, materials, applications])
    arr = FlodymArray(dims=full_dims, values=np.arange(np.prod(full_dims.shape)).reshape(full_dims.shape).astype(float))
    
    # Create subset dimensions
    subset_years = Dimension(name="subset_years", letter="y", items=[2025, 2026, 2027, 2028, 2029])  # 5 years
    subset_apps = Dimension(name="subset_apps", letter="b", items=["A1", "A2", "A3", "A4"])  # 4 apps
    
    # Test 1: Single dimension subset works
    result1 = arr[{"m": "concrete", "t": subset_years}]
    assert result1.shape == (3, 5, 6), f"Expected (3, 5, 6), got {result1.shape}"
    
    # Test 2: Two dimension subsets - THIS IS THE FAILING CASE
    result2 = arr[{"m": "concrete", "t": subset_years, "a": subset_apps}]
    assert result2.shape == (3, 5, 4), f"Expected (3, 5, 4), got {result2.shape}"
    
    # Test 3: Verify values are correct (order matters!)
    # Manual extraction for comparison - sequential slicing should give same result
    expected = arr[{"m": "concrete"}][{"t": subset_years}][{"a": subset_apps}]
    assert_array_equal(result2.values, expected.values)
    
    # Test 4: Test with different order of keys in dict (should not matter)
    result2b = arr[{"a": subset_apps, "m": "concrete", "t": subset_years}]
    assert_array_equal(result2b.values, expected.values)
    
    # Test 5: setitem with multiple dimension subsets
    arr2 = arr.copy()
    new_values = FlodymArray(
        dims=DimensionSet(dim_list=[regions, subset_years, subset_apps]),
        values=np.ones((3, 5, 4)) * 999
    )
    arr2[{"m": "concrete", "t": subset_years, "a": subset_apps}] = new_values
    
    # Verify values were set correctly
    subset = arr2[{"m": "concrete", "t": subset_years, "a": subset_apps}]
    assert np.allclose(subset.values, 999), "setitem with multiple dimension subsets failed"
    
    # Test 6: Verify other values were NOT changed
    other_material = arr2[{"m": "steel"}]
    expected_other = arr[{"m": "steel"}]
    assert_array_equal(other_material.values, expected_other.values)


def test_setitem():
    array_1d = FlodymArray(dims=dims["t",])
    array_1d[...] = 1
    array_1d[1990] = 2
    assert_array_equal(array_1d.values, np.array([2, 1, 1]))

    array_2d = FlodymArray(dims=dims["p", "t"])
    array_2d[...] = 1
    array_2d[1990] = 2
    assert_array_equal(
        array_2d.values,
        np.array([[2, 1, 1]] * 4),
    )


def test_to_df():
    fda = deepcopy(space_animals)
    fda.values[...] = 0.0
    fda.values[0, 1, 0] = 1.0

    df = fda.to_df()

    assert df.shape == (24, 1)
    assert df.loc[("Earth", 2000, "cat"), "value"] == 1.0
    assert df.loc[("Earth", 2000, "mouse"), "value"] == 0.0

    df = fda.to_df(sparse=True)

    assert df.shape == (1, 1)
    assert df.loc[("Earth", 2000, "cat"), "value"] == 1.0
    with pytest.raises(KeyError):
        df.loc[("Earth", 2000, "mouse"), "value"]


if __name__ == "__main__":
    test_to_df()
