import numpy as np
import pytest

import flodym as fd
from flodym.data_reader import CSVDimensionReader
from flodym.mfa_definition import DimensionDefinition, MFADefinition


@pytest.fixture
def building():
    return fd.Dimension(name="Building type", letter="b", items=["Com", "Res"], dtype=str)


@pytest.fixture
def residential():
    return fd.Dimension(name="Residential type", letter="r", items=["Sin", "Mul"], dtype=str)


@pytest.fixture
def sub_type(building, residential):
    return building.split(name="Building sub-type", letter="s", splitting={"Res": residential})


@pytest.fixture
def time():
    return fd.Dimension(name="Time", letter="t", items=[2020, 2030], dtype=int)


@pytest.fixture
def production_s(sub_type, time):
    values = np.array([[1.0, 2.0], [8.0, 8.0], [2.0, 2.0]])
    return fd.FlodymArray(dims=sub_type + time, values=values, name="production")


# Dimension -------------------------------------------------------------------------------------


def test_split_items(sub_type, building, residential):
    assert sub_type.items == ["Com", "Res|Sin", "Res|Mul"]
    assert sub_type.parent_items == ["Com", "Res", "Res"]
    assert sub_type.parent is building
    assert sub_type.sub_dims == {"Res": residential}
    assert sub_type.dtype is str


def test_split_errors(building, residential):
    with pytest.raises(ValueError, match="not items"):
        building.split(name="Sub-type", letter="s", splitting={"Ind": residential})
    piped = fd.Dimension(name="Piped", letter="p", items=["a|b"])
    split = building.split(name="Sub-type", letter="s", splitting={"Res": piped})
    assert split.items == ["Com", "Res|a|b"]
    clashing = fd.Dimension(name="Clashing", letter="c", items=["Res", "Res|a"])
    with pytest.raises(ValueError, match="not unique"):
        clashing.split(
            name="Sub-type",
            letter="s",
            splitting={"Res": fd.Dimension(name="Sub-items", letter="a", items=["a"])},
        )


def test_nested_split(building, residential):
    infra = fd.Dimension(name="Infrastructure", letter="i", items=["Infr", "Build"])
    level_1 = infra.split(name="Level 1", letter="l", splitting={"Build": building})
    level_2 = level_1.split(name="Level 2", letter="m", splitting={"Build|Res": residential})
    assert level_2.items == ["Infr", "Build|Com", "Build|Res|Sin", "Build|Res|Mul"]
    assert level_2.ancestors == [level_1, infra]
    assert level_2.ancestor_items(infra) == ["Infr", "Build", "Build", "Build"]
    assert level_2.is_descendant_of(infra)
    assert not infra.is_descendant_of(level_2)


def test_mapping_validation():
    regions = fd.Dimension(name="Region", letter="w", items=["EUR", "USA"])
    with pytest.raises(ValueError, match="not assigned"):
        fd.Dimension.from_mapping("Country", "c", regions, {"DEU": "EUR"})
    with pytest.raises(ValueError, match="not items of the parent"):
        fd.Dimension.from_mapping("Country", "c", regions, {"DEU": "EUR", "USA": "XXX"})
    with pytest.raises(ValueError, match="same length"):
        fd.Dimension(name="Country", letter="c", items=["DEU"], parent=regions, parent_items=[])
    with pytest.raises(ValueError, match="same letter"):
        fd.Dimension.from_mapping("Country", "w", regions, {"DEU": "EUR", "USA": "USA"})


def test_common_ancestor(building, residential, sub_type):
    other_split = building.split(name="Other", letter="o", splitting={"Com": residential})
    assert sub_type.common_ancestor(building) is building
    assert building.common_ancestor(sub_type) is building
    assert sub_type.common_ancestor(other_split) == building
    assert sub_type.common_ancestor(residential) is None


# Aggregation and disaggregation ----------------------------------------------------------------


def test_sum_to_ancestor(production_s, building):
    expected = np.array([[1.0, 2.0], [10.0, 10.0]])
    for key in ("b", "Building type", building):
        aggregated = production_s.sum_to((key, "t"))
        assert aggregated.dims.letters == ("b", "t")
        np.testing.assert_allclose(aggregated.values, expected)
    np.testing.assert_allclose(production_s.sum_values_to(("b",)), [3.0, 20.0])


def test_disaggregate_with_shares_and_aggregate_round_trip(production_s, sub_type, building):
    production_b = production_s.aggregate_to(building)
    shares = fd.FlodymArray(dims=sub_type.as_dimset(), values=np.array([1.0, 0.8, 0.2]))
    split = production_b.disaggregate_to(sub_type, weights=shares)
    assert split.dims.letters == ("s", "t")
    np.testing.assert_allclose(split.values, production_s.values)
    np.testing.assert_allclose(split.aggregate_to(building).values, production_b.values)


def test_disaggregate_with_unnormalized_and_dict_weights(production_s, sub_type, building):
    production_b = production_s.aggregate_to(building)
    unnormalized = fd.FlodymArray(dims=sub_type.as_dimset(), values=np.array([7.0, 4.0, 1.0]))
    from_dict = {
        "Res": fd.FlodymArray(dims=sub_type.sub_dims["Res"].as_dimset(), values=np.array([4, 1]))
    }
    for weights in (unnormalized, from_dict):
        split = production_b.disaggregate_to(sub_type, weights=weights)
        np.testing.assert_allclose(split.values, production_s.values)


def test_disaggregate_without_weights_copies(building, sub_type):
    intensity_b = fd.FlodymArray(dims=building.as_dimset(), values=np.array([5.0, 10.0]))
    np.testing.assert_allclose(intensity_b.disaggregate_to(sub_type).values, [5.0, 10.0, 10.0])


def test_weighted_mean(production_s, sub_type, building):
    intensity_s = fd.FlodymArray(dims=sub_type.as_dimset(), values=np.array([5.0, 10.0, 20.0]))
    intensity_b = intensity_s.aggregate_to(building, weights=production_s)
    assert set(intensity_b.dims.letters) == {"b", "t"}
    np.testing.assert_allclose(intensity_b[{"t": 2020}].values, [5.0, 12.0])


def test_zero_weights_give_equal_shares(building, sub_type):
    total = fd.FlodymArray(dims=building.as_dimset(), values=np.array([1.0, 10.0]))
    zero_weights = fd.FlodymArray(dims=sub_type.as_dimset(), values=np.zeros(3))
    split = total.disaggregate_to(sub_type, weights=zero_weights)
    np.testing.assert_allclose(split.values, [1.0, 5.0, 5.0])


def test_multi_level_aggregation(building, residential):
    infra = fd.Dimension(name="Infrastructure", letter="i", items=["Infr", "Build"])
    level_1 = infra.split(name="Level 1", letter="l", splitting={"Build": building})
    level_2 = level_1.split(name="Level 2", letter="m", splitting={"Build|Res": residential})
    flow = fd.FlodymArray(dims=level_2.as_dimset(), values=np.array([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_allclose(flow.aggregate_to(infra).values, [1.0, 9.0])
    np.testing.assert_allclose(flow.aggregate_to(level_1).values, [1.0, 2.0, 7.0])
    shares = fd.FlodymArray(dims=level_2.as_dimset(), values=np.array([1.0, 2.0, 3.0, 4.0]))
    total = fd.FlodymArray(dims=infra.as_dimset(), values=np.array([1.0, 9.0]))
    np.testing.assert_allclose(total.disaggregate_to(level_2, weights=shares).values, flow.values)


def test_assignment_across_several_levels(building, residential, time):
    infra = fd.Dimension(name="Infrastructure", letter="i", items=["Infr", "Build"])
    level_1 = infra.split(name="Level 1", letter="l", splitting={"Build": building})
    level_2 = level_1.split(name="Level 2", letter="m", splitting={"Build|Res": residential})
    flow = fd.FlodymArray(dims=level_2 + time, values=np.arange(1.0, 9.0).reshape(4, 2))
    target_i = fd.FlodymArray(dims=time + infra)
    target_i[...] = flow
    np.testing.assert_allclose(target_i.values, [[1.0, 15.0], [2.0, 18.0]])
    target_l = fd.FlodymArray(dims=level_1.as_dimset())
    target_l[...] = flow
    np.testing.assert_allclose(target_l.values, [3.0, 7.0, 26.0])
    target_slice = fd.FlodymArray(dims=infra + time)
    target_slice[{"t": 2030}] = flow[{"t": 2030}]
    np.testing.assert_allclose(target_slice.values, [[0.0, 2.0], [0.0, 18.0]])
    reordered_infra = fd.Dimension(name="Infrastructure", letter="i", items=["Build", "Infr"])
    target_reordered = fd.FlodymArray(dims=reordered_infra.as_dimset())
    target_reordered[...] = flow
    np.testing.assert_allclose(target_reordered.values, [33.0, 3.0])
    with pytest.raises(ValueError, match="does not have"):
        flow[...] = target_i


def test_aggregate_errors(production_s, building, residential):
    with pytest.raises(ValueError, match="no dimension"):
        production_s.aggregate_to(residential)
    production_b = production_s.aggregate_to(building)
    with pytest.raises(ValueError, match="already has"):
        production_b.aggregate_to(building)
    with pytest.raises(ValueError, match="dictionary"):
        production_b.disaggregate_to(production_s.dims["s"], weights={"Com": production_b})


def test_country_to_region_with_reordered_items():
    regions = fd.Dimension(name="Region", letter="w", items=["EUR", "USA"])
    countries = fd.Dimension.from_mapping(
        "Country", "c", regions, {"USA": "USA", "DEU": "EUR", "FRA": "EUR"}
    )
    population = fd.FlodymArray(dims=countries.as_dimset(), values=np.array([3.0, 1.0, 2.0]))
    reordered_regions = fd.Dimension(name="Region", letter="w", items=["USA", "EUR"])
    np.testing.assert_allclose(population.aggregate_to(reordered_regions).values, [3.0, 3.0])


# Automatic handling in operations --------------------------------------------------------------


def test_addition_aggregates(production_s, building):
    production_b = production_s.aggregate_to(building)
    for result in (production_b + production_s, production_s + production_b):
        assert set(result.dims.letters) == {"b", "t"}
        np.testing.assert_allclose(result.sum_to(("b", "t")).values, 2 * production_b.values)
    difference = production_s - production_b
    np.testing.assert_allclose(difference.values, 0.0)


def test_addition_of_siblings_aggregates_to_common_ancestor(building, residential, sub_type):
    other_split = building.split(name="Other", letter="o", splitting={"Com": residential})
    first = fd.FlodymArray(dims=sub_type.as_dimset(), values=np.array([1.0, 2.0, 3.0]))
    second = fd.FlodymArray(dims=other_split.as_dimset(), values=np.array([1.0, 1.0, 1.0]))
    result = first + second
    assert result.dims.letters == ("b",)
    np.testing.assert_allclose(result.values, [3.0, 6.0])
    with pytest.raises(ValueError, match="different sub-dimensions"):
        first * second


def test_multiplication_refines(production_s, building):
    intensity_b = fd.FlodymArray(dims=building.as_dimset(), values=np.array([5.0, 10.0]))
    for result in (intensity_b * production_s, production_s * intensity_b):
        assert set(result.dims.letters) == {"s", "t"}
        np.testing.assert_allclose(
            result.sum_to(("s", "t")).values, [[5.0, 10.0], [80.0, 80.0], [20.0, 20.0]]
        )
    quotient = production_s / intensity_b
    np.testing.assert_allclose(quotient[{"t": 2020}].values, [0.2, 0.8, 0.2])


def test_power_and_minimum(production_s, building):
    exponent = fd.FlodymArray(dims=building.as_dimset(), values=np.array([1.0, 2.0]))
    np.testing.assert_allclose((production_s**exponent)[{"t": 2020}].values, [1.0, 64.0, 4.0])
    limit = fd.FlodymArray(dims=building.as_dimset(), values=np.array([2.0, 25.0]))
    minimum = production_s.minimum(limit)
    assert minimum.dims.letters == ("b",)
    np.testing.assert_allclose(minimum.values, [2.0, 20.0])


def test_cast_to_refines(building, sub_type, time):
    intensity_b = fd.FlodymArray(dims=building.as_dimset(), values=np.array([5.0, 10.0]))
    cast = intensity_b.cast_to(sub_type + time)
    assert cast.dims.letters == ("s", "t")
    np.testing.assert_allclose(cast.values, [[5.0, 5.0], [10.0, 10.0], [10.0, 10.0]])


def test_assignment(production_s, building, time):
    target = fd.FlodymArray(dims=building + time)
    target[...] = production_s
    np.testing.assert_allclose(target.values, [[1.0, 2.0], [10.0, 10.0]])
    with pytest.raises(ValueError, match="does not have"):
        production_s[...] = target


def test_arrays_without_sub_dims_unchanged(time):
    region = fd.Dimension(name="Region", letter="r", items=["A", "B"])
    first = fd.FlodymArray(dims=region + time, values=np.ones((2, 2)))
    second = fd.FlodymArray(dims=time.as_dimset(), values=np.array([1.0, 2.0]))
    np.testing.assert_allclose((first + second).values, [3.0, 4.0])
    np.testing.assert_allclose((first * second).values, [[1.0, 2.0], [1.0, 2.0]])


# Slicing by ancestor items ---------------------------------------------------------------------


def test_setitem_by_parent_item(production_s):
    production_s[{"s": "Res"}] = 0.0
    np.testing.assert_allclose(production_s.values, [[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]])
    production_s["Res", 2030] = np.array([3.0, 4.0])
    np.testing.assert_allclose(production_s[{"t": 2030}].values, [2.0, 3.0, 4.0])
    production_s[{"s": ["Com", "Res"], "t": 2020}] = 7.0
    np.testing.assert_allclose(production_s[{"t": 2020}].values, [7.0, 7.0, 7.0])


def test_setitem_by_higher_ancestor_item(building, residential):
    infra = fd.Dimension(name="Infrastructure", letter="i", items=["Infr", "Build"])
    level_1 = infra.split(name="Level 1", letter="l", splitting={"Build": building})
    level_2 = level_1.split(name="Level 2", letter="m", splitting={"Build|Res": residential})
    array = fd.FlodymArray(dims=level_2.as_dimset(), values=np.zeros(4))
    array[{"m": "Build"}] = 1.0
    np.testing.assert_allclose(array.values, [0.0, 1.0, 1.0, 1.0])
    array[{"m": "Build|Res"}] = 2.0
    np.testing.assert_allclose(array.values, [0.0, 1.0, 2.0, 2.0])


def test_getitem_by_split_item(production_s, residential):
    slicing_dict = {"s": "Res"}
    for keys in (slicing_dict, "Res"):
        sliced = production_s[keys]
        assert sliced.dims.letters == ("r", "t")
        assert sliced.dims["r"] == residential
        np.testing.assert_allclose(sliced.values, [[8.0, 8.0], [2.0, 2.0]])
    assert slicing_dict == {"s": "Res"}
    np.testing.assert_allclose(production_s["Res", 2030].values, [8.0, 2.0])


def test_setitem_by_split_item_with_sub_dim_array(production_s, residential):
    production_s["Res"] = production_s["Res"] * 2.0
    np.testing.assert_allclose(production_s[{"t": 2020}].values, [1.0, 16.0, 4.0])
    values = fd.FlodymArray(dims=residential.as_dimset(), values=np.array([3.0, 5.0]))
    production_s[{"s": "Res", "t": 2030}] = values
    np.testing.assert_allclose(production_s[{"t": 2030}].values, [2.0, 3.0, 5.0])


def test_getitem_by_other_ancestor_items_fails(production_s, building, residential):
    with pytest.raises(ValueError, match="only be used to set values"):
        production_s[{"s": ["Com", "Res"]}]
    infra = fd.Dimension(name="Infrastructure", letter="i", items=["Infr", "Build"])
    level_1 = infra.split(name="Level 1", letter="l", splitting={"Build": building})
    level_2 = level_1.split(name="Level 2", letter="m", splitting={"Build|Res": residential})
    array = fd.FlodymArray(dims=level_2.as_dimset(), values=np.arange(4.0))
    np.testing.assert_allclose(array["Build|Res"].values, [2.0, 3.0])
    with pytest.raises(ValueError, match="only be used to set values"):
        array["Build"]
    regions = fd.Dimension(name="Region", letter="w", items=["EUR", "USA"])
    countries = fd.Dimension.from_mapping("Country", "c", regions, {"DEU": "EUR", "USA": "USA"})
    population = fd.FlodymArray(dims=countries.as_dimset(), values=np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="only be used to set values"):
        population["EUR"]


@pytest.fixture
def nested_with_ancestor_sub_dims(building, residential, sub_type):
    infra = fd.Dimension(name="Infrastructure", letter="i", items=["Infr", "Build"])
    level_1 = infra.split(name="Level 1", letter="l", splitting={"Build": building})
    level_2 = level_1.split(
        name="Level 2",
        letter="m",
        splitting={"Build|Res": residential},
        ancestor_sub_dims={"Build": sub_type},
    )
    return infra, level_1, level_2


def test_ancestor_sub_dims_slicing(nested_with_ancestor_sub_dims, sub_type, residential):
    _, level_1, level_2 = nested_with_ancestor_sub_dims
    assert level_2.sub_dims == {"Build|Res": residential, "Build": sub_type}
    array = fd.FlodymArray(dims=level_2.as_dimset(), values=np.array([1.0, 2.0, 3.0, 4.0]))
    by_build = array["Build"]
    assert by_build.dims.letters == ("s",)
    np.testing.assert_allclose(by_build.values, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(array["Build|Res"].values, [3.0, 4.0])
    np.testing.assert_allclose(array.aggregate_to(level_1).values, [1.0, 2.0, 7.0])
    np.testing.assert_allclose(by_build.aggregate_to(sub_type.parent).values, [2.0, 7.0])
    array["Build"] = by_build * 10.0
    np.testing.assert_allclose(array.values, [1.0, 20.0, 30.0, 40.0])


def test_ancestor_sub_dims_errors(building, residential):
    infra = fd.Dimension(name="Infrastructure", letter="i", items=["Infr", "Build"])
    level_1 = infra.split(name="Level 1", letter="l", splitting={"Build": building})
    reordered = fd.Dimension(name="Reordered", letter="s", items=["Res|Sin", "Com", "Res|Mul"])
    with pytest.raises(ValueError, match="must correspond"):
        level_1.split(
            name="Level 2",
            letter="m",
            splitting={"Build|Res": residential},
            ancestor_sub_dims={"Build": reordered},
        )
    with pytest.raises(ValueError, match="must correspond"):
        level_1.split(
            name="Level 2",
            letter="m",
            splitting={"Build|Res": residential},
            ancestor_sub_dims={"Build|Com": residential},
        )


def test_dict_weights_ignore_ancestor_sub_dims(nested_with_ancestor_sub_dims, residential):
    _, level_1, level_2 = nested_with_ancestor_sub_dims
    total = fd.FlodymArray(dims=level_1.as_dimset(), values=np.array([1.0, 2.0, 10.0]))
    weights = {"Build|Res": fd.FlodymArray(dims=residential.as_dimset(), values=np.array([4, 1]))}
    split = total.disaggregate_to(level_2, weights=weights)
    np.testing.assert_allclose(split.values, [1.0, 2.0, 8.0, 2.0])


# Extensivity -----------------------------------------------------------------------------------


@pytest.fixture
def strict():
    """Raise on any extensivity report."""
    with fd.config.override(error_behaviors={"extensivity": "error"}):
        yield


@pytest.fixture
def flow_s(production_s):
    return production_s.to_class(fd.FlodymArray, extensive=True)


@pytest.fixture
def intensity_s(sub_type):
    values = np.array([5.0, 10.0, 20.0])
    return fd.FlodymArray(dims=sub_type.as_dimset(), values=values, extensive=False)


@pytest.fixture
def intensity_b(building):
    return fd.FlodymArray(dims=building.as_dimset(), values=np.array([5.0, 10.0]), extensive=False)


def test_correct_usage_is_not_reported(
    strict, flow_s, intensity_s, intensity_b, building, sub_type, time
):
    flow_b = flow_s.aggregate_to(building)
    assert flow_b.extensive is True
    shares = fd.FlodymArray(
        dims=sub_type.as_dimset(), values=np.array([1.0, 0.8, 0.2]), extensive=False
    )
    split = flow_b.disaggregate_to(sub_type, weights=shares)
    assert split.extensive is True
    np.testing.assert_allclose(split.values, flow_s.values)
    mean = intensity_s.aggregate_to(building, weights=flow_s)
    assert mean.extensive is False
    copied = intensity_b.disaggregate_to(sub_type)
    assert copied.extensive is False
    assert (flow_s + flow_b).extensive is True
    assert (intensity_b * flow_s).extensive is True
    assert flow_s.sum_to(("b", "t")).extensive is True
    target_b = fd.FlodymArray(dims=building + time, extensive=True)
    target_b[...] = flow_s
    np.testing.assert_allclose(target_b.values, flow_b.values)


@pytest.mark.parametrize(
    "operation, message",
    [
        (lambda arrays: arrays["intensity_s"].aggregate_to(arrays["b"]), "summed"),
        (lambda arrays: arrays["intensity_s"].sum_to(("b",)), "summed"),
        (lambda arrays: arrays["flow_b"].disaggregate_to(arrays["s"]), "copied"),
        (
            lambda arrays: arrays["flow_s"].aggregate_to(arrays["b"], weights=arrays["flow_s"]),
            "averaged with weights",
        ),
        (
            lambda arrays: arrays["intensity_b"].disaggregate_to(
                arrays["s"], weights=arrays["flow_s"]
            ),
            "split by weights",
        ),
    ],
)
def test_wrong_usage_is_reported(
    strict, operation, message, flow_s, intensity_s, intensity_b, building, sub_type
):
    arrays = {
        "flow_s": flow_s,
        "flow_b": flow_s.aggregate_to(building),
        "intensity_s": intensity_s,
        "intensity_b": intensity_b,
        "b": building,
        "s": sub_type,
    }
    with pytest.raises(ValueError, match=message):
        operation(arrays)


def test_check_extensivity_false_skips_reports(strict, intensity_s, building):
    summed = intensity_s.aggregate_to(building, check_extensivity=False)
    np.testing.assert_allclose(summed.values, [5.0, 30.0])


def test_addition_of_intensive_arrays_refines(strict, intensity_s, intensity_b):
    for result in (intensity_b + intensity_s, intensity_s + intensity_b):
        assert result.dims.letters == ("s",)
        assert result.extensive is False
        np.testing.assert_allclose(result.values, [10.0, 20.0, 30.0])
    minimum = intensity_s.minimum(intensity_b)
    np.testing.assert_allclose(minimum.values, [5.0, 10.0, 10.0])


def test_assignment_of_intensive_coarse_array_copies(strict, intensity_b, sub_type):
    target = fd.FlodymArray(dims=sub_type.as_dimset(), extensive=False)
    target[...] = intensity_b
    np.testing.assert_allclose(target.values, [5.0, 10.0, 10.0])
    unknown_target = fd.FlodymArray(dims=sub_type.as_dimset())
    unknown_target[{"s": "Res"}] = intensity_b[{"b": "Res"}]
    np.testing.assert_allclose(unknown_target.values, [0.0, 10.0, 10.0])
    assert unknown_target.extensive is False


def test_assignment_of_extensive_coarse_array_fails(flow_s, building, sub_type, time):
    flow_b = flow_s.aggregate_to(building)
    target = fd.FlodymArray(dims=sub_type + time, extensive=True)
    with pytest.raises(ValueError, match="disaggregate_to"):
        target[...] = flow_b


def test_assignment_of_intensive_fine_array_is_reported(strict, intensity_s, building):
    target = fd.FlodymArray(dims=building.as_dimset(), extensive=False)
    with pytest.raises(ValueError, match="summed"):
        target[...] = intensity_s


# Read-in ---------------------------------------------------------------------------------------


def test_read_mapping_and_split_dimensions(tmp_path):
    (tmp_path / "regions.csv").write_text("EUR\nUSA\n")
    (tmp_path / "countries.csv").write_text("DEU,EUR\nFRA,EUR\nUSA,USA\n")
    (tmp_path / "buildings.csv").write_text("Com\nRes\n")
    (tmp_path / "residential.csv").write_text("Sin\nMul\n")
    reader = CSVDimensionReader(
        dimension_files={
            "Region": tmp_path / "regions.csv",
            "Country": tmp_path / "countries.csv",
            "Building type": tmp_path / "buildings.csv",
            "Residential type": tmp_path / "residential.csv",
        }
    )
    definitions = [
        DimensionDefinition(name="Country", letter="c", dtype=str, parent="w"),
        DimensionDefinition(
            name="Building sub-type", letter="s", dtype=str, parent="b", splitting={"Res": "r"}
        ),
        DimensionDefinition(name="Region", letter="w", dtype=str),
        DimensionDefinition(name="Building type", letter="b", dtype=str),
        DimensionDefinition(name="Residential type", letter="r", dtype=str),
    ]
    dims = reader.read_dimensions(definitions)
    assert dims.letters == ("c", "s", "w", "b", "r")
    assert dims["c"].parent == dims["w"]
    assert dims["c"].parent_items == ["EUR", "EUR", "USA"]
    assert dims["s"].items == ["Com", "Res|Sin", "Res|Mul"]
    assert dims["s"].parent == dims["b"]


def test_read_mapping_with_single_column_fails(tmp_path):
    (tmp_path / "regions.csv").write_text("EUR\nUSA\n")
    reader = CSVDimensionReader(
        dimension_files={"Region": tmp_path / "regions.csv", "Country": tmp_path / "regions.csv"}
    )
    definitions = [
        DimensionDefinition(name="Region", letter="w", dtype=str),
        DimensionDefinition(name="Country", letter="c", dtype=str, parent="w"),
    ]
    with pytest.raises(ValueError, match="two columns"):
        reader.read_dimensions(definitions)


def test_definition_checks():
    with pytest.raises(ValueError, match="no parent"):
        DimensionDefinition(name="Sub", letter="s", dtype=str, splitting={"Res": "r"})

    def make_definition(dimensions):
        return MFADefinition(dimensions=dimensions, processes=[], flows=[])

    with pytest.raises(ValueError, match="not defined"):
        make_definition([DimensionDefinition(name="Country", letter="c", dtype=str, parent="w")])
    with pytest.raises(ValueError, match="Cyclic"):
        make_definition(
            [
                DimensionDefinition(name="First", letter="a", dtype=str, parent="b"),
                DimensionDefinition(name="Second", letter="b", dtype=str, parent="a"),
            ]
        )
