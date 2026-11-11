import importlib
import pkgutil

import numpy as np
import pydantic
import pytest

import flodym
from flodym import Dimension, FlowDefinition, StockDefinition
from flodym._base_model import FlodymBaseModel, _describe_value
from flodym.example_objects import ExampleMFA, get_example_array, get_example_mfa
from flodym.export.array_plotter import PlotlyArrayPlotter
from flodym.stocks import SimpleFlowDrivenStock


def _flodym_pydantic_models() -> list[type[pydantic.BaseModel]]:
    for module_info in pkgutil.walk_packages(flodym.__path__, "flodym."):
        importlib.import_module(module_info.name)
    models = []
    classes_to_visit = [pydantic.BaseModel]
    while classes_to_visit:
        for subclass in classes_to_visit.pop().__subclasses__():
            classes_to_visit.append(subclass)
            if subclass.__module__.startswith("flodym.") and subclass not in models:
                models.append(subclass)
    return sorted(models, key=lambda model: f"{model.__module__}.{model.__qualname__}")


@pytest.mark.parametrize("model", _flodym_pydantic_models(), ids=lambda model: model.__qualname__)
def test_models_use_flodym_base_model(model: type[pydantic.BaseModel]):
    assert issubclass(model, FlodymBaseModel)
    assert model.model_config.get("extra") == "forbid"
    assert model.model_config.get("arbitrary_types_allowed") is True


def test_unknown_argument_raises_for_dimension():
    with pytest.raises(flodym.FlodymValidationError, match="typo"):
        Dimension(name="Time", letter="t", items=[2020], typo=1)  # ty: ignore[unknown-argument]


def test_unknown_argument_raises_for_definition():
    with pytest.raises(flodym.FlodymValidationError, match="name_overide"):
        FlowDefinition(
            from_process_name="sysenv",
            to_process_name="process",
            dim_letters=("t",),
            name_overide="flow",  # ty: ignore[unknown-argument]
        )


def test_stock_definition_accepts_process_alias_and_name():
    by_alias = StockDefinition(dim_letters=("t",), subclass=SimpleFlowDrivenStock, process="use")
    by_name = StockDefinition(
        dim_letters=("t",), subclass=SimpleFlowDrivenStock, process_name="use"
    )
    assert by_alias.process_name == by_name.process_name == "use"


def test_unknown_argument_and_attribute_raise_for_mfa_system():
    mfa = get_example_mfa()
    with pytest.raises(flodym.FlodymValidationError, match="typo"):
        ExampleMFA(
            dims=mfa.dims,
            parameters=mfa.parameters,
            processes=mfa.processes,
            flows=mfa.flows,
            stocks=mfa.stocks,
            typo=1,  # ty: ignore[unknown-argument]
        )
    with pytest.raises(ValueError, match="no field"):
        mfa.undeclared_attribute = 1


def test_unknown_argument_raises_for_array_plotter():
    array = get_example_array()
    with pytest.raises(flodym.FlodymValidationError, match="linecolor_dimm"):
        PlotlyArrayPlotter(
            array=array,
            intra_line_dim=array.dims.letters[0],
            linecolor_dimm=array.dims.letters[-1],  # ty: ignore[unknown-argument]
        )


def _time_dims() -> flodym.DimensionSet:
    return flodym.DimensionSet(dim_list=[Dimension(name="time", letter="t", items=[1, 2, 3])])


def test_validation_error_is_value_error_and_keeps_pydantic_error():
    with pytest.raises(ValueError) as error:
        Dimension(name="t", letter="t", items=[1])
    assert isinstance(error.value, flodym.FlodymValidationError)
    assert isinstance(error.value.pydantic_error, pydantic.ValidationError)
    assert error.value.__suppress_context__


def test_validation_error_message_leaves_out_pydantic_details():
    with pytest.raises(flodym.FlodymValidationError) as error:
        Dimension(name="t", letter="tt", items=[1])
    message = str(error.value)
    assert message.startswith("Invalid arguments for Dimension:\n")
    assert "- name: String should have at least 2 characters. Given: 't' (type str)." in message
    assert "- letter: " in message
    for pydantic_detail in ("errors.pydantic.dev", "type=", "input_value", "Value error,"):
        assert pydantic_detail not in message


def test_unknown_argument_message_suggests_closest_argument():
    with pytest.raises(flodym.FlodymValidationError) as error:
        flodym.FlodymArray(dims=_time_dims(), valuse=1)  # ty: ignore[unknown-argument]
    assert "- 'valuse' is not a valid argument. Did you mean 'values'?" in str(error.value)
    assert "Valid arguments: dims, values, name, extensive." in str(error.value)


def test_unknown_argument_message_includes_aliases():
    with pytest.raises(flodym.FlodymValidationError) as error:
        FlowDefinition(
            from_proces="sysenv",  # ty: ignore[unknown-argument]
            to_process_name="process",
            dim_letters=("t",),
        )
    assert "Did you mean 'from_process'?" in str(error.value)


def test_missing_argument_message():
    with pytest.raises(flodym.FlodymValidationError, match="- Missing required argument 'dims'."):
        flodym.FlodymArray()


def test_message_of_flodym_check_is_kept_without_prefix():
    with pytest.raises(flodym.FlodymValidationError) as error:
        flodym.FlodymArray(dims=_time_dims(), values=np.ones(4))
    lines = str(error.value).splitlines()
    assert lines[0] == "Invalid arguments for FlodymArray:"
    assert lines[1].startswith("- Values passed to FlodymArray must have the same shape")
    # further lines of the message are indented to stay within the item
    assert lines[2].startswith("  ")


def test_union_type_errors_are_combined():
    with pytest.raises(flodym.FlodymValidationError) as error:
        flodym.FixedLifetime(dims=_time_dims(), mean="a")
    assert str(error.value) == (
        "Invalid arguments for FixedLifetime:\n"
        "- mean: must be a FlodymArray, a numpy array or a number (type SupportsFloat). "
        "Given: 'a' (type str)."
    )


def test_union_errors_other_than_type_mismatch_are_kept():
    dims = _time_dims()
    with pytest.raises(flodym.FlodymValidationError) as error:
        flodym.FixedLifetime(dims=dims, mean={"dims": dims, "values": np.ones(4)})
    lines = str(error.value).splitlines()
    assert lines[1] == (
        "- mean: must be a FlodymArray, a numpy array or a number (type SupportsFloat). "
        "Given: dict with keys 'dims', 'values'."
    )
    assert lines[2].startswith("  - as FlodymArray: Values passed to FlodymArray must have")
    assert lines[3].startswith("    ")
    with pytest.raises(flodym.FlodymValidationError) as error:
        flodym.FixedLifetime(dims=dims, mean={"values": np.ones(3)})
    lines = str(error.value).splitlines()
    assert lines[2] == "  - as FlodymArray: Missing required argument 'dims'."


def test_nested_errors_refer_to_nested_model():
    with pytest.raises(flodym.FlodymValidationError) as error:
        flodym.MFADefinition(
            dimensions=[{"name": "time", "letter": "t"}],
            processes=["sysenv"],
            flows=[
                {
                    "from_process": "sysenv",
                    "to_process": "sysenv",
                    "dim_letters": ("t",),
                    "name_overide": "flow",
                }
            ],
            stocks=[],
            parameters=[],
        )
    message = str(error.value)
    assert "- dimensions[0] (DimensionDefinition): Missing required argument 'dtype'." in message
    assert (
        "- 'name_overide' is not a valid argument of flows[0] (FlowDefinition). "
        "Did you mean 'name_override'?"
    ) in message
    assert message.count("Invalid arguments for") == 1


def test_model_validate_raises_readable_error():
    with pytest.raises(flodym.FlodymValidationError, match="Invalid arguments for Dimension"):
        Dimension.model_validate({"name": "t", "letter": "t", "items": [1]})


def test_assignment_errors_are_readable():
    with pytest.raises(flodym.FlodymValidationError) as error:
        flodym.config.checks.mass_balance_after_compute = "abc"
    assert str(error.value).startswith("Invalid value assigned to Checks:\n")
    lifetime_model = flodym.FixedLifetime(dims=_time_dims(), mean=np.ones(3))
    with pytest.raises(flodym.FlodymValidationError, match="Invalid value assigned to"):
        lifetime_model.inflow_at = "begin"


def test_config_override_raises_readable_error():
    with (
        pytest.raises(flodym.FlodymValidationError) as error,
        flodym.config.override(error_behaviors={"unset_array_used": "loud"}),
    ):
        pass
    assert str(error.value).count("Invalid arguments for ErrorBehaviors") == 1
    assert flodym.config.error_behaviors.unset_array_used != "loud"


def test_pydantic_does_not_call_init_for_nested_models():
    # pydantic calls a custom __init__ also to validate nested models given as dictionaries,
    # which would nest the error messages; see FlodymBaseModel.__init__
    assert not FlodymBaseModel.__pydantic_custom_init__
    assert not flodym.FlodymArray.__pydantic_custom_init__


@pytest.mark.parametrize("model", _flodym_pydantic_models(), ids=lambda model: model.__qualname__)
def test_assignment_validation_uses_readable_errors(model: type[pydantic.BaseModel]):
    if model.model_config.get("validate_assignment"):
        assert getattr(model.__setattr__, "_readable_errors", False)
    else:
        assert not getattr(model.__setattr__, "_readable_errors", False)


def test_pydantic_message_is_used_if_formatting_fails(monkeypatch: pytest.MonkeyPatch):
    def failing_problem(*args, **kwargs):
        raise KeyError("unforeseen error details")

    monkeypatch.setattr(flodym._base_model, "_problem", failing_problem)
    with pytest.raises(flodym.FlodymValidationError) as error:
        Dimension(name="t", letter="t", items=[1])
    assert str(error.value) == str(error.value.pydantic_error)


def test_positional_arguments_raise_readable_type_error():
    with pytest.raises(TypeError) as error:
        flodym.FlodymArray(_time_dims())
    assert str(error.value) == (
        "FlodymArray accepts only keyword arguments, e.g. "
        "SomeClass(argument_1=value_1), not SomeClass(value_1).\n"
        "Given: 1 argument without name. Valid arguments: dims, values, name, extensive."
    )
    with pytest.raises(TypeError, match="Given: 3 arguments without name"):
        Dimension("time", "t", [1])


def test_errors_gives_pydantic_error_details():
    with pytest.raises(flodym.FlodymValidationError) as error:
        Dimension(name="t", letter="tt", items=[1])
    errors = error.value.errors(include_url=False)
    assert [(e["type"], e["loc"]) for e in errors] == [
        ("string_too_short", ("name",)),
        ("string_too_long", ("letter",)),
    ]


def test_single_failed_check_is_cause_and_keeps_its_traceback():
    with pytest.raises(flodym.FlodymValidationError) as error:
        flodym.FlodymArray(dims=_time_dims(), values=np.ones(4))
    cause = error.value.__cause__
    assert type(cause) is ValueError
    assert str(cause).startswith("Values passed to FlodymArray must have the same shape")
    assert cause.__traceback__ is not None
    assert error.value.__suppress_context__


def test_pydantic_errors_have_no_cause():
    with pytest.raises(flodym.FlodymValidationError) as error:
        Dimension(name="t", letter="t", items=[1])
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__


def test_exception_type_of_failed_check_is_named_unless_value_error():
    class CheckedModel(FlodymBaseModel):
        value: int

        @pydantic.field_validator("value")
        @classmethod
        def _check_value(cls, value: int) -> int:
            assert value > 0, "value must be positive"
            return value

    with pytest.raises(flodym.FlodymValidationError) as error:
        CheckedModel(value=0)
    assert "- value: AssertionError: value must be positive" in str(error.value)
    assert isinstance(error.value.__cause__, AssertionError)


@pytest.mark.parametrize(
    ("value", "description"),
    [
        (np.ones((2, 3)), "numpy.ndarray with shape (2, 3) and dtype float64"),
        (
            flodym.FlodymArray(dims=_time_dims(), name="demand"),
            "flodym.FlodymArray 'demand' with dims (t) and shape (3,)",
        ),
        (_time_dims(), "flodym.DimensionSet with dims (t)"),
        ({"a": 1, "b": 2}, "dict with keys 'a', 'b'"),
        ({}, "empty dict"),
        ("a", "'a' (type str)"),
        (
            list(range(100)),
            "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16... (type list, length 100)",
        ),
    ],
)
def test_describe_value(value, description):
    assert _describe_value(value) == description


def test_describe_value_of_other_array_like_gives_shape_in_one_line():
    pandas = pytest.importorskip("pandas")
    assert _describe_value(pandas.DataFrame({"a": [1, 2]})) == "pandas.DataFrame with shape (2, 1)"
