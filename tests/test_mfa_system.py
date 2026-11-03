import pytest

from flodym import (
    DataReader,
    Dimension,
    DimensionDefinition,
    FlowDefinition,
    MFADefinition,
    MFASystem,
)


class _SingleDimDataReader(DataReader):
    """Minimal reader for tests with one dimension and no parameters."""

    def __init__(self, items_by_dimension_name: dict[str, list]):
        self.items_by_dimension_name = items_by_dimension_name

    def read_dimension(self, dimension_definition: DimensionDefinition) -> Dimension:
        return Dimension(
            name=dimension_definition.name,
            letter=dimension_definition.letter,
            items=self.items_by_dimension_name[dimension_definition.name],
            dtype=dimension_definition.dtype,
        )

    def read_parameter_values(self, parameter_name, dims):
        raise AssertionError(f"No parameters expected, but got request for {parameter_name}.")


class _MinimalMFASystem(MFASystem):
    def compute(self):
        pass


def test_mfa_system_from_definition_without_stocks():
    definition = MFADefinition(
        dimensions=[DimensionDefinition(name="time", letter="t", dtype=int)],
        processes=["sysenv"],
        flows=[],
    )
    data_reader = _SingleDimDataReader(items_by_dimension_name={"time": [2020]})

    mfa = _MinimalMFASystem.from_data_reader(definition=definition, data_reader=data_reader)

    assert mfa.stocks == {}
    assert mfa.flows == {}
    assert mfa.parameters == {}
    assert list(mfa.processes.keys()) == ["sysenv"]


def test_mass_balance_check_without_stocks_for_simple_system():
    definition = MFADefinition(
        dimensions=[DimensionDefinition(name="time", letter="t", dtype=int)],
        processes=["sysenv", "process"],
        flows=[
            FlowDefinition(
                from_process_name="sysenv",
                to_process_name="process",
                dim_letters=("t",),
            ),
            FlowDefinition(
                from_process_name="process",
                to_process_name="sysenv",
                dim_letters=("t",),
            ),
        ],
    )
    data_reader = _SingleDimDataReader(items_by_dimension_name={"time": [2020]})

    mfa = _MinimalMFASystem.from_data_reader(definition=definition, data_reader=data_reader)

    mfa.flows["sysenv => process"][...] = 2.0
    mfa.flows["process => sysenv"][...] = 2.0
    mfa.check_mass_balance(raise_error=True)

    mfa.flows["process => sysenv"][...] = 1.0
    with pytest.raises(ValueError, match="Mass balance check failed"):
        mfa.check_mass_balance(tolerance=1e-12, raise_error=True)
