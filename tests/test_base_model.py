import importlib
import pkgutil

import pydantic
import pytest

import flodym
from flodym import Dimension, FlowDefinition, StockDefinition
from flodym._base_model import FlodymBaseModel
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
    with pytest.raises(pydantic.ValidationError, match="typo"):
        Dimension(name="Time", letter="t", items=[2020], typo=1)  # ty: ignore[unknown-argument]


def test_unknown_argument_raises_for_definition():
    with pytest.raises(pydantic.ValidationError, match="name_overide"):
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
    with pytest.raises(pydantic.ValidationError, match="typo"):
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
    with pytest.raises(pydantic.ValidationError, match="linecolor_dimm"):
        PlotlyArrayPlotter(
            array=array,
            intra_line_dim=array.dims.letters[0],
            linecolor_dimm=array.dims.letters[-1],  # ty: ignore[unknown-argument]
        )
