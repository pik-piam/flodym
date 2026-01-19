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


class TestFlodymArrayIndexing:
    """Tests for getitem and setitem, including with advanced indexing separated by slices."""

    # 80 years
    years = Dimension(name="year", letter="t", items=list(range(2020, 2100)))
    # 12 regions
    regions = Dimension(name="region", letter="r", items=[f"R{i}" for i in range(1, 13)])
    # 4 sectors
    sectors = Dimension(name="sector", letter="s", items=["S1", "S2", "S3", "S4"])
    # 7 products
    products = Dimension(
        name="product", letter="p", items=["P1", "P2", "P3", "P4", "P5", "P6", "P7"]
    )
    # 2 materials
    materials = Dimension(name="material", letter="m", items=["M1", "M2"])
    full_dims = DimensionSet(dim_list=[years, regions, sectors, products, materials])
    arr = FlodymArray(
        dims=full_dims,
        values=np.arange(np.prod(full_dims.shape)).reshape(full_dims.shape).astype(float),
    )
    # 5 regions in subset
    subset_regions = Dimension(name="sub_regions", letter="x", items=["R1", "R2", "R3", "R4", "R5"])
    # 3 products in subset
    subset_products = Dimension(name="sub_products", letter="y", items=["P1", "P2", "P3"])
    # mask includes one material, subset regions and subset products
    mask = {"m": "M1", "r": subset_regions, "p": subset_products}

    def test_get_item(self):
        cats_on_the_moon = space_animals["Moon"]["cat"]
        assert isinstance(cats_on_the_moon, FlodymArray)
        assert_array_almost_equal(cats_on_the_moon.values, space_animals.values[2, :, 0])
        # note that this does not work for the time dimension (not strings)
        # and also assumes that no item appears in more than one dimension

    def test_getitem_indexing_with_slice(self):
        """Test getitem with advanced indices separated by slices (NumPy dimension reordering)."""
        result = self.arr[self.mask]
        expected_shape = (80, 5, 4, 3)
        assert (
            result.shape == expected_shape
        ), f"getitem: Expected {expected_shape}, got {result.shape}"
        expected_values = self.arr[{"m": "M1"}][{"r": self.subset_regions}][
            {"p": self.subset_products}
        ]
        assert_array_equal(result.values, expected_values.values)

    def test_setitem(self):
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

    def test_setitem_indexing_with_slice(self):
        """Test setitem with advanced indices separated by slices (NumPy dimension reordering)."""
        arr_copy = self.arr.copy()

        slice_dims = DimensionSet(
            dim_list=[self.years, self.subset_regions, self.sectors, self.subset_products]
        )
        new_values = FlodymArray.full(slice_dims, fill_value=123.0)
        arr_copy[self.mask] = new_values
        result = arr_copy[self.mask]
        assert np.allclose(result.values, 123), "setitem failed to update values correctly"


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


class TestFlodymArrayFull:
    """Tests for FlodymArray.full() class method with various fill value types."""

    def test_full_with_int(self):
        filled = FlodymArray.full(dims, fill_value=3, name="int_filled")
        assert filled.name == "int_filled"
        assert filled.dims == dims
        assert filled.dims is not dims  # ensure dims is copied
        assert np.all(filled.values == 3)
        assert filled.values.dtype == np.int_

    def test_full_with_float(self):
        filled = FlodymArray.full(dims, fill_value=3.5, name="float_filled")
        assert filled.name == "float_filled"
        assert filled.dims == dims
        assert np.all(filled.values == 3.5)
        assert filled.values.dtype == np.float64

    def test_full_with_np_array(self):
        """Test full with a numpy array that broadcasts across dimensions."""
        # Create a 1D array that broadcasts along the time dimension
        time_values = np.array([1.0, 2.0, 3.0])  # shape (3,) for 3 time steps
        filled = FlodymArray.full(dims, fill_value=time_values, name="np_array_filled")
        assert filled.name == "np_array_filled"
        assert filled.values.shape == dims.shape  # (4, 3)
        # Each row should have [1.0, 2.0, 3.0]
        assert_array_equal(filled.values[0], time_values)
        assert_array_equal(filled.values[1], time_values)
        assert_array_equal(filled.values[3], time_values)

    def test_full_with_2d_np_array(self):
        """Test full with a 2D numpy array matching the full shape."""
        fill_array = np.arange(12).reshape(4, 3).astype(np.float64)
        filled = FlodymArray.full(dims, fill_value=fill_array, name="2d_array_filled")
        assert filled.name == "2d_array_filled"
        assert_array_equal(filled.values, fill_array)
        assert filled.values.dtype == np.float64

    def test_full_dims_not_modified(self):
        """Ensure the original dims is not modified when creating a full array."""
        original_dims = DimensionSet(dim_list=base_dim_list)
        filled = FlodymArray.full(original_dims, fill_value=1.0)
        # Modify the filled array's dims
        filled.dims.dim_list[0] = animals
        # Original should be unchanged
        assert original_dims[0] == places


class TestFlodymArrayFullLike:
    """Tests for FlodymArray.full_like() class method with various fill value types."""

    def test_full_like_with_int(self):
        template = FlodymArray.full(dims, fill_value=2.0, name="template")
        filled_like = FlodymArray.full_like(template, fill_value=5, name="int_filled")
        assert filled_like.name == "int_filled"
        assert filled_like.dims == template.dims
        assert filled_like.dims is not template.dims  # ensure dims is copied
        assert np.all(filled_like.values == 5)
        assert filled_like.values.dtype == np.int_

    def test_full_like_with_float(self):
        template = FlodymArray.full(dims, fill_value=2.0, name="template")
        filled_like = FlodymArray.full_like(template, fill_value=-1.5, name="float_filled")
        assert filled_like.name == "float_filled"
        assert np.all(filled_like.values == -1.5)
        assert filled_like.values.dtype == np.float64

    def test_full_like_with_np_array(self):
        """Test full_like with a numpy array that broadcasts across dimensions."""
        template = FlodymArray.full(dims, fill_value=2.0)
        # Create a 1D array that broadcasts along the time dimension
        time_values = np.array([10.0, 20.0, 30.0])  # shape (3,) for 3 time steps
        filled_like = FlodymArray.full_like(template, fill_value=time_values)
        assert filled_like.values.shape == dims.shape  # (4, 3)
        # Each row should have [10.0, 20.0, 30.0]
        assert_array_equal(filled_like.values[0], time_values)
        assert_array_equal(filled_like.values[1], time_values)
        assert_array_equal(filled_like.values[2], time_values)

    def test_full_like_with_2d_np_array(self):
        """Test full_like with a 2D numpy array matching the full shape."""
        template = FlodymArray.full(dims, fill_value=2.0)
        fill_array = np.arange(12).reshape(4, 3).astype(np.int32)
        filled_like = FlodymArray.full_like(template, fill_value=fill_array)
        assert_array_equal(filled_like.values, fill_array)
        assert filled_like.values.dtype == np.int32

    def test_full_like_with_explicit_dtype(self):
        """Test that explicit dtype parameter overrides fill_value dtype."""
        template = FlodymArray.full(dims, fill_value=2.0)
        filled_like = FlodymArray.full_like(template, fill_value=5, dtype=np.float64)
        assert np.all(filled_like.values == 5.0)
        assert filled_like.values.dtype == np.float64

    def test_full_like_template_not_modified(self):
        """Ensure the template array is not modified."""
        template = FlodymArray.full(dims, fill_value=2.0, name="template")
        original_values = template.values.copy()
        filled_like = FlodymArray.full_like(template, fill_value=-1.0)
        # Modify filled_like
        filled_like.values[...] = 999
        filled_like.dims.dim_list[0] = animals
        # Template should be unchanged
        assert_array_equal(template.values, original_values)
        assert template.dims[0] == places


class TestFlodymArrayScalar:
    """Tests for FlodymArray.scalar() class method with various value types."""

    def test_scalar_with_int(self):
        scalar = FlodymArray.scalar(42, name="int_scalar")
        assert scalar.name == "int_scalar"
        assert len(scalar.dims) == 0
        assert scalar.dims.letters == ()
        assert scalar.shape == ()
        assert scalar.size == 1
        assert scalar.values.shape == ()
        assert scalar.values.item() == 42

    def test_scalar_with_float(self):
        scalar = FlodymArray.scalar(4.2, name="float_scalar")
        assert scalar.name == "float_scalar"
        assert scalar.values.item() == pytest.approx(4.2)

    def test_scalar_with_np_float(self):
        scalar = FlodymArray.scalar(np.float32(2.718), name="np_float_scalar")
        assert scalar.values.item() == pytest.approx(2.718, rel=1e-5)
        assert scalar.values.dtype == np.float32


class TestFlodymArrayCopy:
    """Tests for FlodymArray.copy() method ensuring deep copy behavior."""

    def test_copy_returns_new_object(self):
        array = FlodymArray.full(dims, fill_value=4, name="original")
        array_copy = array.copy()
        assert array_copy is not array

    def test_copy_dims_equal_but_not_same_object(self):
        """Ensure dims is deep copied, not just referenced."""
        array = FlodymArray.full(dims, fill_value=4, name="original")
        array_copy = array.copy()
        assert array_copy.dims == array.dims
        assert array_copy.dims is not array.dims

    def test_copy_dims_internal_dimensions_not_shared(self):
        """Ensure individual Dimension objects within dims are also copied."""
        array = FlodymArray.full(dims, fill_value=4, name="original")
        array_copy = array.copy()
        # Check that internal dimension list is not the same object
        assert array_copy.dims.dim_list is not array.dims.dim_list

    def test_copy_values_equal_but_not_same_object(self):
        """Ensure values array is deep copied, not just referenced."""
        array = FlodymArray.full(dims, fill_value=4, name="original")
        array_copy = array.copy()
        assert_array_equal(array_copy.values, array.values)
        assert array_copy.values is not array.values

    def test_copy_name_preserved(self):
        """Ensure name is copied correctly."""
        array = FlodymArray.full(dims, fill_value=4, name="my_array")
        array_copy = array.copy()
        assert array_copy.name == array.name
        assert array_copy.name == "my_array"

    def test_copy_values_modification_independent(self):
        """Modifying copied values should not affect original."""
        array = FlodymArray.full(dims, fill_value=4, name="original")
        array_copy = array.copy()
        array_copy.values[...] = 999
        assert np.all(array.values == 4)
        assert np.all(array_copy.values == 999)

    def test_copy_dims_modification_independent(self):
        """Modifying copied dims should not affect original."""
        array = FlodymArray.full(dims, fill_value=4, name="original")
        array_copy = array.copy()
        # Try to modify the copy's dims
        array_copy.dims.dim_list[0] = animals
        # Original should be unchanged
        assert array.dims[0] == places


if __name__ == "__main__":
    t = TestFlodymArrayIndexing()
    t.test_getitem_indexing_with_slice()
