import numpy as np
import pytest

from flodym.data_reader import (
    CSVDimensionReader,
    CSVParameterReader,
    ExcelDimensionReader,
    ExcelParameterReader,
)
from flodym.mfa_definition import DimensionDefinition, MFADefinition, ParameterDefinition
from flodym.mfa_system import MFASystem

CSV_DIMENSION_FILES = {
    "animals": "tests/tests_data/dimension_animals_horizontal.csv",
    "time": "tests/tests_data/dimension_time_vertical.csv",
    "region": "tests/tests_data/dimension_region_single.csv",
}
EXCEL_DIMENSION_FILE = "tests/tests_data/dimensions.xlsx"
EXCEL_DIMENSION_FILES = {a: EXCEL_DIMENSION_FILE for a in CSV_DIMENSION_FILES}
DIMENSION_SHEET_NAMES = {a: a for a in CSV_DIMENSION_FILES}
DIMENSION_DEFINITIONS = [
    DimensionDefinition(name="animals", letter="a", dtype=str),
    DimensionDefinition(name="time", letter="t", dtype=int),
    DimensionDefinition(name="region", letter="r", dtype=str),
]
DIMENSION_ITEMS_EXPECTED = {
    "animals": ["dog", "cat"],
    "time": [1990, 2000, 2010],
    "region": ["world"],
}
PARAMETER_DEFINITIONS = [
    ParameterDefinition(name="p1", dim_letters=["t", "r"]),
    ParameterDefinition(name="p2", dim_letters=["t", "r"]),
    ParameterDefinition(name="p3", dim_letters=["t", "r"]),
    ParameterDefinition(name="p4", dim_letters=["t", "r"]),
    ParameterDefinition(name="p5", dim_letters=["t", "r", "a"]),
    ParameterDefinition(name="p6", dim_letters=["t", "r", "a"]),
]
CSV_PARAMETER_FILES = {
    a.name: "tests/tests_data/parameter_" + a.name + ".csv" for a in PARAMETER_DEFINITIONS
}
EXCEL_PARAMETER_FILE = "tests/tests_data/parameters.xlsx"
EXCEL_PARAMETER_FILES = {a.name: EXCEL_PARAMETER_FILE for a in PARAMETER_DEFINITIONS}
PARAMETER_SHEET_NAMES = {a.name: a.name for a in PARAMETER_DEFINITIONS}


WRONG_PARAMETER_DEFINITIONS = [
    # missing row
    ParameterDefinition(name="e1", dim_letters=["t", "a"]),
    # wrong dim item
    ParameterDefinition(name="e2", dim_letters=["t"]),
    # additional dim item
    ParameterDefinition(name="e3", dim_letters=["t"]),
    # empty cell
    ParameterDefinition(name="e4", dim_letters=["t"]),
    # missing dim
    ParameterDefinition(name="e5", dim_letters=["t", "a"]),
]

WRONG_CSV_PARAMETER_FILES = {
    a.name: "tests/tests_data/parameter_" + a.name + ".csv" for a in WRONG_PARAMETER_DEFINITIONS
}
WRONG_EXCEL_PARAMETER_FILE = "tests/tests_data/parameters.xlsx"
WRONG_EXCEL_PARAMETER_FILES = {
    a.name: WRONG_EXCEL_PARAMETER_FILE for a in WRONG_PARAMETER_DEFINITIONS
}
WRONG_SHEET_NAMES = {a.name: a.name for a in WRONG_PARAMETER_DEFINITIONS}


def test_dimensions():

    csv_reader = CSVDimensionReader(CSV_DIMENSION_FILES)
    excel_reader = ExcelDimensionReader(EXCEL_DIMENSION_FILES, DIMENSION_SHEET_NAMES)

    for reader in [csv_reader, excel_reader]:
        dimension_set = reader.read_dimensions(DIMENSION_DEFINITIONS)

        for dim in dimension_set:
            assert dim.items == DIMENSION_ITEMS_EXPECTED[dim.name]


def test_valid_parameter_reader():

    dims = CSVDimensionReader(CSV_DIMENSION_FILES).read_dimensions(DIMENSION_DEFINITIONS)

    csv_reader = CSVParameterReader(CSV_PARAMETER_FILES)
    excel_reader = ExcelParameterReader(EXCEL_PARAMETER_FILES, PARAMETER_SHEET_NAMES)

    for reader in [csv_reader, excel_reader]:
        # for reader in [excel_reader]:
        parameters = reader.read_parameters(PARAMETER_DEFINITIONS, dims)
        for pn, p in parameters.items():
            if pn in ["p1", "p2", "p3", "p4"]:
                assert np.array_equal(p.values, [[1], [2], [3]])
            elif pn in ["p5", "p6"]:
                assert np.array_equal(p.values, np.array([[[1, 4]], [[2, 5]], [[3, 6]]]))


def test_wrong_parameter_reader():

    wrong_csv_reader = CSVParameterReader(WRONG_CSV_PARAMETER_FILES)
    wrong_excel_reader = ExcelParameterReader(WRONG_EXCEL_PARAMETER_FILES, WRONG_SHEET_NAMES)

    dims = CSVDimensionReader(CSV_DIMENSION_FILES).read_dimensions(DIMENSION_DEFINITIONS)

    for reader in [wrong_csv_reader, wrong_excel_reader]:
        for prm_def in WRONG_PARAMETER_DEFINITIONS:
            with pytest.raises(ValueError):
                reader.read_parameter_values(prm_def.name, dims.get_subset(prm_def.dim_letters))


def test_allow_incomplete_data():
    incomplete_parameter_definitions = [
        # missing row
        ParameterDefinition(name="e1", dim_letters=["t", "a"]),
        # empty cell
        ParameterDefinition(name="e4", dim_letters=["t"]),
    ]

    incomplete_csv_reader = CSVParameterReader(WRONG_CSV_PARAMETER_FILES, allow_missing_values=True)
    incomplete_excel_reader = ExcelParameterReader(
        WRONG_EXCEL_PARAMETER_FILES, WRONG_SHEET_NAMES, allow_missing_values=True
    )

    dims = CSVDimensionReader(CSV_DIMENSION_FILES).read_dimensions(DIMENSION_DEFINITIONS)

    for reader in [incomplete_csv_reader, incomplete_excel_reader]:
        parameters = reader.read_parameters(incomplete_parameter_definitions, dims)
        assert np.array_equal(parameters["e1"].values, [[4, 1], [5, 0], [6, 3]])
        assert np.array_equal(parameters["e4"].values, [0, 2, 3])


def test_allow_extra_data():
    extra_parameter_definitions = [
        # wrong dim item
        ParameterDefinition(name="e2", dim_letters=["t"]),
        # additional dim item
        ParameterDefinition(name="e3", dim_letters=["t"]),
    ]

    extra_csv_reader = CSVParameterReader(
        WRONG_CSV_PARAMETER_FILES, allow_missing_values=True, allow_extra_values=True
    )
    extra_excel_reader = ExcelParameterReader(
        WRONG_EXCEL_PARAMETER_FILES,
        WRONG_SHEET_NAMES,
        allow_missing_values=True,
        allow_extra_values=True,
    )

    dims = CSVDimensionReader(CSV_DIMENSION_FILES).read_dimensions(DIMENSION_DEFINITIONS)

    for reader in [extra_csv_reader, extra_excel_reader]:
        parameters = reader.read_parameters(extra_parameter_definitions, dims)
        assert np.array_equal(parameters["e2"].values, [1, 0, 3])
        assert np.array_equal(parameters["e3"].values, [1, 2, 3])


def test_build_mfa_system():

    definition = MFADefinition(
        dimensions=DIMENSION_DEFINITIONS,
        processes=[],
        stocks=[],
        flows=[],
        parameters=PARAMETER_DEFINITIONS,
    )

    class MinimalMFASystem(MFASystem):
        def compute(self):
            pass

    MinimalMFASystem.from_csv(
        definition,
        CSV_DIMENSION_FILES,
        CSV_PARAMETER_FILES,
    )

    MinimalMFASystem.from_excel(
        definition,
        EXCEL_DIMENSION_FILES,
        EXCEL_PARAMETER_FILES,
        dimension_sheets=DIMENSION_SHEET_NAMES,
        parameter_sheets=PARAMETER_SHEET_NAMES,
    )


def test_dimension_letters_as_column_names():
    """Test that dimension letters can be used as column names in CSV files."""
    dims = CSVDimensionReader(CSV_DIMENSION_FILES).read_dimensions(DIMENSION_DEFINITIONS)

    # Test with all letters as column names
    csv_files_letters = {
        "p1": "tests/tests_data/parameter_p1_letters.csv",
        "p5": "tests/tests_data/parameter_p5_letters.csv",
    }
    reader_letters = CSVParameterReader(csv_files_letters)

    # Test parameter with 2 dimensions (t, r)
    param_def_p1 = ParameterDefinition(name="p1", dim_letters=["t", "r"])
    params_p1 = reader_letters.read_parameters([param_def_p1], dims)
    assert np.array_equal(params_p1["p1"].values, [[1], [2], [3]])

    # Test parameter with 3 dimensions (t, r, a)
    param_def_p5 = ParameterDefinition(name="p5", dim_letters=["t", "r", "a"])
    params_p5 = reader_letters.read_parameters([param_def_p5], dims)
    assert np.array_equal(params_p5["p5"].values, np.array([[[1, 4]], [[2, 5]], [[3, 6]]]))


def test_mixed_dimension_letters_and_names():
    """Test that dimension letters and names can be mixed as column names."""
    dims = CSVDimensionReader(CSV_DIMENSION_FILES).read_dimensions(DIMENSION_DEFINITIONS)

    # Test with mixed letters and names
    csv_files_mixed = {
        "p1": "tests/tests_data/parameter_p1_mixed.csv",
    }
    reader_mixed = CSVParameterReader(csv_files_mixed)

    param_def_p1 = ParameterDefinition(name="p1", dim_letters=["t", "r"])
    params_p1 = reader_mixed.read_parameters([param_def_p1], dims)
    assert np.array_equal(params_p1["p1"].values, [[1], [2], [3]])


def test_excel_readers_use_first_sheet_without_sheet_names(tmp_path):
    import pandas as pd

    from flodym import DimensionSet

    def write_two_sheets(path, first, second, header):
        with pd.ExcelWriter(path) as writer:
            first.to_excel(writer, sheet_name="first", index=False, header=header)
            second.to_excel(writer, sheet_name="second", index=False, header=header)

    dimension_path = tmp_path / "dimensions.xlsx"
    write_two_sheets(dimension_path, pd.DataFrame([1990, 2000]), pd.DataFrame([1, 2, 3]), False)
    parameter_path = tmp_path / "parameters.xlsx"
    write_two_sheets(
        parameter_path,
        pd.DataFrame({"time": [1990, 2000], "value": [1.0, 2.0]}),
        pd.DataFrame({"time": [1990, 2000], "value": [3.0, 4.0]}),
        True,
    )
    time_definition = DimensionDefinition(name="time", letter="t", dtype=int)

    dimension = ExcelDimensionReader({"time": dimension_path}).read_dimension(time_definition)
    assert dimension.items == [1990, 2000]

    dims = DimensionSet(dim_list=[dimension])
    parameter = ExcelParameterReader({"p": parameter_path}).read_parameter_values("p", dims)
    assert np.array_equal(parameter.values, [1.0, 2.0])


def test_data_reader_is_abstract():
    from flodym import DataReader

    with pytest.raises(TypeError):
        DataReader()  # ty: ignore[abstract-class-instantiation]


def test_parameter_filenames_is_deprecated():
    reader = CSVParameterReader(CSV_PARAMETER_FILES)
    with pytest.warns(DeprecationWarning, match="parameter_files"):
        assert reader.parameter_filenames is reader.parameter_files
    with pytest.warns(DeprecationWarning, match="parameter_files"):
        reader.parameter_filenames = {}
    assert reader.parameter_files == {}
