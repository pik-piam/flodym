from pydantic_core import ValidationError
import pytest

from flodym import Dimension, DimensionSet


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


def test_intersection_operator():
    """Test the intersection operator (&) for DimensionSets."""
    dims1 = [
        {"name": "time", "letter": "t", "items": [1990, 2000, 2010]},
        {"name": "place", "letter": "p", "items": ["World"]},
        {"name": "material", "letter": "m", "items": ["steel", "aluminum"]},
    ]
    dims2 = [
        {"name": "place", "letter": "p", "items": ["World"]},
        {"name": "material", "letter": "m", "items": ["steel", "aluminum"]},
        {"name": "product", "letter": "r", "items": ["car", "bike"]},
    ]
    set1 = DimensionSet(dim_list=dims1)
    set2 = DimensionSet(dim_list=dims2)

    # Test intersection using & operator
    intersection = set1 & set2
    assert intersection.letters == ("p", "m")
    assert intersection.names == ("place", "material")

    # Test that it's equivalent to intersect_with method
    assert intersection == set1.intersect_with(set2)

    # Test intersection with no common dimensions
    dims3 = [{"name": "animal", "letter": "a", "items": ["cat", "dog"]}]
    set3 = DimensionSet(dim_list=dims3)
    empty_intersection = set1 & set3
    assert len(empty_intersection.dim_list) == 0

    # Test intersection with same set
    same_intersection = set1 & set1
    assert same_intersection == set1


def test_union_operator():
    """Test the union operator (|) for DimensionSets."""
    dims1 = [
        {"name": "time", "letter": "t", "items": [1990, 2000, 2010]},
        {"name": "place", "letter": "p", "items": ["World"]},
    ]
    dims2 = [
        {"name": "place", "letter": "p", "items": ["World"]},
        {"name": "material", "letter": "m", "items": ["steel", "aluminum"]},
    ]
    set1 = DimensionSet(dim_list=dims1)
    set2 = DimensionSet(dim_list=dims2)

    # Test union using | operator
    union = set1 | set2
    assert union.letters == ("t", "p", "m")
    assert union.names == ("time", "place", "material")

    # Test that it's equivalent to union_with method
    assert union == set1.union_with(set2)

    # Test union with disjoint sets
    dims3 = [{"name": "animal", "letter": "a", "items": ["cat", "dog"]}]
    set3 = DimensionSet(dim_list=dims3)
    disjoint_union = set1 | set3
    assert disjoint_union.letters == ("t", "p", "a")
    assert disjoint_union.names == ("time", "place", "animal")

    # Test union with same set
    same_union = set1 | set1
    assert same_union == set1


def test_difference_operator():
    """Test the difference operator (-) for DimensionSets."""
    dims1 = [
        {"name": "time", "letter": "t", "items": [1990, 2000, 2010]},
        {"name": "place", "letter": "p", "items": ["World"]},
        {"name": "material", "letter": "m", "items": ["steel", "aluminum"]},
    ]
    dims2 = [
        {"name": "place", "letter": "p", "items": ["World"]},
        {"name": "product", "letter": "r", "items": ["car", "bike"]},
    ]
    set1 = DimensionSet(dim_list=dims1)
    set2 = DimensionSet(dim_list=dims2)

    # Test difference using - operator
    difference = set1 - set2
    assert difference.letters == ("t", "m")
    assert difference.names == ("time", "material")

    # Test that it's equivalent to difference_with method
    assert difference == set1.difference_with(set2)

    # Test difference with disjoint sets
    dims3 = [{"name": "animal", "letter": "a", "items": ["cat", "dog"]}]
    set3 = DimensionSet(dim_list=dims3)
    disjoint_difference = set1 - set3
    assert disjoint_difference == set1

    # Test difference with same set
    same_difference = set1 - set1
    assert len(same_difference.dim_list) == 0


def test_symmetric_difference_operator():
    """Test the symmetric difference operator (^) for DimensionSets."""
    dims1 = [
        {"name": "time", "letter": "t", "items": [1990, 2000, 2010]},
        {"name": "place", "letter": "p", "items": ["World"]},
        {"name": "material", "letter": "m", "items": ["steel", "aluminum"]},
    ]
    dims2 = [
        {"name": "place", "letter": "p", "items": ["World"]},
        {"name": "product", "letter": "r", "items": ["car", "bike"]},
    ]
    set1 = DimensionSet(dim_list=dims1)
    set2 = DimensionSet(dim_list=dims2)

    # Test symmetric difference using ^ operator
    sym_diff = set1 ^ set2
    assert set(sym_diff.letters) == {"t", "m", "r"}
    assert set(sym_diff.names) == {"time", "material", "product"}

    # Test that it's equivalent to (set1 - set2) | (set2 - set1)
    expected = (set1 - set2) | (set2 - set1)
    assert sym_diff.letters == expected.letters

    # Test symmetric difference with same set
    same_sym_diff = set1 ^ set1
    assert len(same_sym_diff.dim_list) == 0

    # Test symmetric difference with disjoint sets
    dims3 = [{"name": "animal", "letter": "a", "items": ["cat", "dog"]}]
    set3 = DimensionSet(dim_list=dims3)
    disjoint_sym_diff = set1 ^ set3
    # Should contain all dimensions from both sets
    assert set(disjoint_sym_diff.letters) == {"t", "p", "m", "a"}


def test_len_dunder():
    """Test the __len__ dunder method for DimensionSets."""
    # Empty dimension set
    empty_set = DimensionSet(dim_list=[])
    assert len(empty_set) == 0

    # Single dimension
    dims1 = [{"name": "time", "letter": "t", "items": [1990, 2000, 2010]}]
    set1 = DimensionSet(dim_list=dims1)
    assert len(set1) == 1

    # Multiple dimensions
    dims2 = [
        {"name": "time", "letter": "t", "items": [1990, 2000, 2010]},
        {"name": "place", "letter": "p", "items": ["World"]},
        {"name": "material", "letter": "m", "items": ["steel", "aluminum"]},
    ]
    set2 = DimensionSet(dim_list=dims2)
    assert len(set2) == 3

    # Test that len is equivalent to ndim property
    assert len(set2) == set2.ndim


def test_bool_dunder():
    """Test the __bool__ dunder method for DimensionSets."""
    # Empty dimension set should be falsy
    empty_set = DimensionSet(dim_list=[])
    assert not empty_set
    assert bool(empty_set) is False

    # Non-empty dimension set should be truthy
    dims = [{"name": "time", "letter": "t", "items": [1990, 2000, 2010]}]
    non_empty_set = DimensionSet(dim_list=dims)
    assert non_empty_set
    assert bool(non_empty_set) is True

    # Test with multiple dimensions
    dims2 = [
        {"name": "time", "letter": "t", "items": [1990, 2000, 2010]},
        {"name": "place", "letter": "p", "items": ["World"]},
    ]
    set2 = DimensionSet(dim_list=dims2)
    assert set2
    assert bool(set2) is True


def test_ndim_property():
    """Test that the ndim property still works correctly after being moved."""
    # Empty dimension set
    empty_set = DimensionSet(dim_list=[])
    assert empty_set.ndim == 0

    # Single dimension
    dims1 = [{"name": "time", "letter": "t", "items": [1990, 2000, 2010]}]
    set1 = DimensionSet(dim_list=dims1)
    assert set1.ndim == 1

    # Multiple dimensions
    dims2 = [
        {"name": "time", "letter": "t", "items": [1990, 2000, 2010]},
        {"name": "place", "letter": "p", "items": ["World"]},
        {"name": "material", "letter": "m", "items": ["steel", "aluminum"]},
    ]
    set2 = DimensionSet(dim_list=dims2)
    assert set2.ndim == 3


def test_dimensionset_empty():
    """DimensionSet.empty should yield a reusable scalar-friendly container."""
    empty = DimensionSet.empty()
    assert isinstance(empty, DimensionSet)
    assert len(empty) == 0
    assert empty.shape == ()
    assert empty.letters == ()
    assert DimensionSet.empty() is not empty


def test_dimensionset_total_size():
    """total_size should equal the product of dimension lengths."""
    dims = [
        {"name": "time", "letter": "t", "items": [1990, 2000, 2010]},
        {"name": "place", "letter": "p", "items": ["Earth", "Moon"]},
    ]
    dim_set = DimensionSet(dim_list=dims)
    assert dim_set.shape == (3, 2)
    assert dim_set.total_size == 6


def test_dimension_as_dimset():
    """Test that as_dimset() converts a Dimension to a DimensionSet."""
    dim = Dimension(name="time", letter="t", items=[1990, 2000, 2010])

    # Convert to DimensionSet
    dimset = dim.as_dimset()

    # Verify it's a DimensionSet
    assert isinstance(dimset, DimensionSet)

    # Verify it contains only the one dimension
    assert len(dimset) == 1
    assert dimset.letters == ("t",)
    assert dimset.names == ("time",)

    # Verify the dimension is the same
    assert dimset[0] == dim


def test_dimension_add():
    """Test the __add__ operator for Dimension objects."""
    dim1 = Dimension(name="time", letter="t", items=[1990, 2000, 2010])
    dim2 = Dimension(name="place", letter="p", items=["World", "Moon"])

    # Test Dimension + Dimension
    dimset = dim1 + dim2
    assert isinstance(dimset, DimensionSet)
    assert len(dimset) == 2
    assert dimset.letters == ("t", "p")
    assert dimset.names == ("time", "place")

    # Test Dimension + DimensionSet
    dim3 = Dimension(name="material", letter="m", items=["steel", "aluminum"])
    dimset2 = dim1 + dim2
    dimset3 = dim3 + dimset2

    assert isinstance(dimset3, DimensionSet)
    assert len(dimset3) == 3
    assert dimset3.letters == ("m", "t", "p")
    assert dimset3.names == ("material", "time", "place")

    # Test adding invalid type
    with pytest.raises(TypeError):
        dim1 + "invalid"


def test_dimensionset_append():
    """Test the append() method for DimensionSet."""
    dims = [
        {"name": "time", "letter": "t", "items": [1990, 2000, 2010]},
        {"name": "place", "letter": "p", "items": ["World"]},
    ]
    dimset = DimensionSet(dim_list=dims)
    new_dim = Dimension(name="material", letter="m", items=["steel", "aluminum"])

    # Test append with inplace=False (returns new DimensionSet)
    new_dimset = dimset.append(new_dim, inplace=False)
    assert new_dimset is not None
    assert len(new_dimset) == 3
    assert new_dimset.letters == ("t", "p", "m")

    # Original should be unchanged
    assert len(dimset) == 2
    assert dimset.letters == ("t", "p")

    # Test append with inplace=True (modifies in place)
    dimset2 = DimensionSet(dim_list=dims)
    result = dimset2.append(new_dim, inplace=True)
    assert result is None
    assert len(dimset2) == 3
    assert dimset2.letters == ("t", "p", "m")

    # Test appending dimension with duplicate letter
    dup_dim = Dimension(name="other_time", letter="t", items=[2020, 2030])
    with pytest.raises(ValueError):
        dimset.append(dup_dim)

    # Test appending non-Dimension object
    with pytest.raises(TypeError):
        dimset.append("invalid")


def test_dimensionset_prepend():
    """Test the prepend() method for DimensionSet."""
    dims = [
        {"name": "time", "letter": "t", "items": [1990, 2000, 2010]},
        {"name": "place", "letter": "p", "items": ["World"]},
    ]
    dimset = DimensionSet(dim_list=dims)
    new_dim = Dimension(name="material", letter="m", items=["steel", "aluminum"])

    # Test prepend with inplace=False (returns new DimensionSet)
    new_dimset = dimset.prepend(new_dim, inplace=False)
    assert new_dimset is not None
    assert len(new_dimset) == 3
    assert new_dimset.letters == ("m", "t", "p")

    # Original should be unchanged
    assert len(dimset) == 2
    assert dimset.letters == ("t", "p")

    # Test prepend with inplace=True (modifies in place)
    dimset2 = DimensionSet(dim_list=dims)
    result = dimset2.prepend(new_dim, inplace=True)
    assert result is None
    assert len(dimset2) == 3
    assert dimset2.letters == ("m", "t", "p")

    # Test prepending dimension with duplicate letter
    dup_dim = Dimension(name="other_time", letter="t", items=[2020, 2030])
    with pytest.raises(ValueError):
        dimset.prepend(dup_dim)

    # Test prepending non-Dimension object
    with pytest.raises(TypeError):
        dimset.prepend("invalid")


def test_dimensionset_remove():
    """Test the remove() method (alias for drop()) for DimensionSet."""
    dims = [
        {"name": "time", "letter": "t", "items": [1990, 2000, 2010]},
        {"name": "place", "letter": "p", "items": ["World"]},
        {"name": "material", "letter": "m", "items": ["steel", "aluminum"]},
    ]
    dimset = DimensionSet(dim_list=dims)

    # Test remove with inplace=False (returns new DimensionSet)
    new_dimset = dimset.remove("m", inplace=False)
    assert new_dimset is not None
    assert len(new_dimset) == 2
    assert new_dimset.letters == ("t", "p")

    # Original should be unchanged
    assert len(dimset) == 3
    assert dimset.letters == ("t", "p", "m")

    # Test remove with inplace=True (modifies in place)
    dimset2 = DimensionSet(dim_list=dims)
    result = dimset2.remove("m", inplace=True)
    assert result is None
    assert len(dimset2) == 2
    assert dimset2.letters == ("t", "p")

    # Test removing with name instead of letter
    dimset3 = DimensionSet(dim_list=dims)
    new_dimset3 = dimset3.remove("time", inplace=False)
    assert len(new_dimset3) == 2
    assert new_dimset3.letters == ("p", "m")


def test_dimensionset_insert():
    """Test the insert() method for DimensionSet."""
    dims = [
        {"name": "time", "letter": "t", "items": [1990, 2000, 2010]},
        {"name": "place", "letter": "p", "items": ["World"]},
    ]
    dimset = DimensionSet(dim_list=dims)
    new_dim = Dimension(name="material", letter="m", items=["steel", "aluminum"])

    # Test insert at beginning (index 0)
    new_dimset = dimset.insert(0, new_dim, inplace=False)
    assert new_dimset is not None
    assert len(new_dimset) == 3
    assert new_dimset.letters == ("m", "t", "p")

    # Test insert at middle (index 1)
    new_dimset2 = dimset.insert(1, new_dim, inplace=False)
    assert len(new_dimset2) == 3
    assert new_dimset2.letters == ("t", "m", "p")

    # Test insert at end (index 2)
    new_dimset3 = dimset.insert(2, new_dim, inplace=False)
    assert len(new_dimset3) == 3
    assert new_dimset3.letters == ("t", "p", "m")

    # Original should be unchanged
    assert len(dimset) == 2
    assert dimset.letters == ("t", "p")

    # Test insert with inplace=True
    dimset2 = DimensionSet(dim_list=dims)
    result = dimset2.insert(1, new_dim, inplace=True)
    assert result is None
    assert len(dimset2) == 3
    assert dimset2.letters == ("t", "m", "p")

    # Test inserting dimension with duplicate letter
    dup_dim = Dimension(name="other_time", letter="t", items=[2020, 2030])
    with pytest.raises(ValueError):
        dimset.insert(0, dup_dim)

    # Test inserting non-Dimension object
    with pytest.raises(TypeError):
        dimset.insert(0, "invalid")


def test_dimensionset_add():
    """Test the __add__ operator for DimensionSet."""
    dims1 = [
        {"name": "time", "letter": "t", "items": [1990, 2000, 2010]},
        {"name": "place", "letter": "p", "items": ["World"]},
    ]
    dims2 = [
        {"name": "material", "letter": "m", "items": ["steel", "aluminum"]},
        {"name": "product", "letter": "r", "items": ["car", "bike"]},
    ]
    dimset1 = DimensionSet(dim_list=dims1)
    dimset2 = DimensionSet(dim_list=dims2)

    # Test DimensionSet + DimensionSet with no overlap
    result = dimset1 + dimset2
    assert isinstance(result, DimensionSet)
    assert len(result) == 4
    assert result.letters == ("t", "p", "m", "r")

    # Test DimensionSet + Dimension
    new_dim = Dimension(name="animal", letter="a", items=["cat", "dog"])
    result2 = dimset1 + new_dim
    assert isinstance(result2, DimensionSet)
    assert len(result2) == 3
    assert result2.letters == ("t", "p", "a")

    # Test that adding overlapping dimensions raises ValueError
    dims3 = [
        {"name": "place", "letter": "p", "items": ["Mars"]},  # 'p' overlaps
        {"name": "product", "letter": "r", "items": ["car", "bike"]},
    ]
    dimset3 = DimensionSet(dim_list=dims3)

    with pytest.raises(ValueError) as error_msg:
        dimset1 + dimset3
    assert "overlap" in str(error_msg.value).lower()

    # Test commutativity for disjoint sets
    result_forward = dimset1 + dimset2
    result_backward = dimset2 + dimset1
    assert set(result_forward.letters) == set(result_backward.letters)


def test_copy_dimension_set():
    """Test the copy method of DimensionSet."""
    dimensions = [
        Dimension(name="time", letter="t", items=[1990, 2000, 2010]),
        Dimension(name="place", letter="p", items=["World"]),
    ]
    original = DimensionSet(dim_list=dimensions)

    # Create a copy
    copied = original.copy()

    # Ensure the copied object is not the same as the original
    assert copied is not original

    # Ensure modifying the copy does not affect the original
    assert copied.dim_list is not original.dim_list

    # Ensure the contained Dimension objects remain the same
    for o_dim, c_dim in zip(original, copied):
        assert o_dim is c_dim
