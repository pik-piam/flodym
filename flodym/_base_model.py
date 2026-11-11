"""Common base class of all flodym pydantic models, and readable validation errors."""

import difflib
import sys
import types
import typing
from collections.abc import Mapping
from typing import Any, SupportsFloat

import numpy as np
from pydantic import AliasChoices, ConfigDict, ValidationError
from pydantic import BaseModel as PydanticBaseModel
from pydantic.fields import FieldInfo
from pydantic_core import ErrorDetails
from typing_extensions import Self

_MAX_INPUT_LENGTH = 60
"""Maximum length of the representation of a given value in an error message."""

_TYPE_NAMES: dict[Any, str] = {
    np.ndarray: "a numpy array",
    SupportsFloat: "a number (type SupportsFloat)",
}
"""Readable names of types in error messages. Other classes are named by their class name."""


class FlodymValidationError(ValueError):
    """Raised if arguments of a flodym class or values assigned to its attributes are invalid.

    The message lists each problem on one line. For handling errors in code, use
    :py:meth:`errors`, which gives e.g. the error type of each problem, independent of the
    wording of the message. The original pydantic error is available as :py:attr:`pydantic_error`.

    If exactly one check of flodym failed, e.g. a shape check, the exception raised by this check
    is the cause of this error, so that the traceback also shows where the check failed.

    It is a subclass of ``ValueError``, so that ``except ValueError`` also catches it.
    """

    def __init__(self, message: str, pydantic_error: ValidationError) -> None:
        super().__init__(message)
        self.pydantic_error = pydantic_error
        """The original pydantic error."""

    def errors(self, **kwargs: Any) -> list[ErrorDetails]:
        """Details of each problem, as returned by ``pydantic.ValidationError.errors()``.

        Each entry contains e.g. the error type, such as ``string_too_short``, which does not
        change with the wording of the message, and the location of the problem as a tuple.

        Args:
            **kwargs: Passed on to ``pydantic.ValidationError.errors()``, e.g.
                ``include_url=False``.
        """
        return self.pydantic_error.errors(**kwargs)


FlodymValidationError.__module__ = "flodym"


class FlodymBaseModel(PydanticBaseModel):
    """Base class of all flodym pydantic models, with a shared configuration.

    Unknown arguments raise an error instead of being ignored, so that typos in argument
    names are caught. Fields may have arbitrary types, e.g. numpy arrays.

    Validation errors are raised as :py:class:`flodym.FlodymValidationError`, with a message
    that is easier to read than the pydantic one.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def __init__(self, /, *args: Any, **data: Any) -> None:
        if args:
            # Python's own error would name FlodymBaseModel and count self as an argument
            raise TypeError(_positional_arguments_message(type(self), len(args))) from None
        try:
            super().__init__(**data)
        except ValidationError as error:
            readable = _readable_error(type(self), error, "Invalid arguments for")
            raise readable from readable.__cause__

    # Tell pydantic that this __init__ behaves like its own. Otherwise, pydantic would call it
    # also to validate nested models given as dictionaries, which is slower, and would nest
    # the error messages of the nested models. The nested errors are made readable by the
    # __init__ of the outermost model instead.
    __init__.__pydantic_base_init__ = True

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Make the errors of assignments readable in models that validate assignments.

        Only these models can raise a ValidationError on assignment. Other models are left
        unchanged, since a wrapper would slow down every assignment.
        """
        super().__pydantic_init_subclass__(**kwargs)
        if cls.model_config.get("validate_assignment") and not getattr(
            cls.__setattr__, "_readable_errors", False
        ):
            cls.__setattr__ = _with_readable_assignment_errors(cls.__setattr__)

    @classmethod
    def model_validate(cls, obj: Any, *args: Any, **kwargs: Any) -> Self:
        try:
            return super().model_validate(obj, *args, **kwargs)
        except ValidationError as error:
            readable = _readable_error(cls, error, "Invalid arguments for")
            raise readable from readable.__cause__


def _with_readable_assignment_errors(setattr_method: Any) -> Any:
    """Wrap ``__setattr__`` such that a ValidationError is raised as FlodymValidationError."""

    def __setattr__(self: PydanticBaseModel, name: str, value: Any) -> None:
        try:
            setattr_method(self, name, value)
        except ValidationError as error:
            readable = _readable_error(type(self), error, "Invalid value assigned to")
            raise readable from readable.__cause__

    __setattr__._readable_errors = True
    return __setattr__


def _positional_arguments_message(model: type[PydanticBaseModel], n_args: int) -> str:
    """Message for arguments given without name, which flodym classes do not accept."""
    given = "1 argument" if n_args == 1 else f"{n_args} arguments"
    return (
        f"{model.__name__} accepts only keyword arguments, e.g. "
        "SomeClass(argument_1=value_1), "
        "not SomeClass(value_1).\n"
        f"Given: {given} without name. "
        f"Valid arguments: {', '.join(_argument_names(model))}."
    )


def _readable_error(
    model: type[PydanticBaseModel], error: ValidationError, title: str
) -> FlodymValidationError:
    """Turn a pydantic validation error into a FlodymValidationError with a readable message.

    If the message cannot be created, e.g. for a kind of error not foreseen here, the pydantic
    message is used, so that the cause of the error is never hidden.

    If exactly one check of flodym (or of a subclass) failed, its exception is set as the cause
    of the returned error. Raising the returned error ``from`` its cause then keeps the traceback
    of the failed check. Otherwise, the cause is None, since the traceback of pydantic's
    validation does not help to find the problem.
    """
    try:
        message = _readable_message(model, error, title)
    except Exception:  # noqa: BLE001 - any failure falls back to the pydantic message
        message = str(error)
    readable = FlodymValidationError(message, pydantic_error=error)
    check_errors = [_check_error(details) for details in error.errors()]
    check_errors = [check_error for check_error in check_errors if check_error is not None]
    readable.__cause__ = check_errors[0] if len(check_errors) == 1 else None
    return readable


def _check_error(details: Any) -> BaseException | None:
    """The exception raised by a check in a validator, if the error comes from one."""
    if details["type"] in ("value_error", "assertion_error"):
        return details.get("ctx", {}).get("error")
    return None


def _readable_message(model: type[PydanticBaseModel], error: ValidationError, title: str) -> str:
    """Message of a pydantic validation error, with one line per problem."""
    errors = error.errors(include_url=False)
    problems: list[str] = []
    union_fields_done: set[str] = set()
    for details in errors:
        field = _union_field(model, details, errors)
        if field is None:
            problems.append(_problem(model, details))
        elif field not in union_fields_done:
            union_fields_done.add(field)
            field_errors = [e for e in errors if e["loc"][:1] == (field,)]
            problems.append(_union_problem(model, field, field_errors))
    return f"{title} {model.__name__}:\n" + "\n".join(problems)


def _problem(model: Any, details: Any) -> str:
    """Describe a single pydantic error as one item of the error message.

    Args:
        model: The type to which the location of the error refers.
        details: The error, as given by ``ValidationError.errors()``.
    """
    error_type = details["type"]
    loc = details["loc"]
    if error_type in ("extra_forbidden", "missing") and loc:
        # the error concerns an argument of the model at the location without the last item
        parent_location, parent = _walk(model, loc[:-1])
        if error_type == "extra_forbidden":
            return _unknown_argument_problem(parent, parent_location, str(loc[-1]))
        message = f"Missing required argument '{loc[-1]}'."
        parent_label = _model_label(parent_location, parent)
        return f"- {parent_label}: {message}" if parent_label else f"- {message}"
    location, _ = _walk(model, loc)
    check_error = _check_error(details)
    if check_error is not None:
        # an error raised by a flodym check: its own message is the most informative
        message = str(check_error).strip()
        if type(check_error) is not ValueError:
            message = f"{type(check_error).__name__}: {message}"
        message = _indent(message)
    else:
        message = f"{details['msg']}. Given: {_describe_value(details['input'])}."
    return f"- {location}: {message}" if location else f"- {message}"


def _model_label(location: str, model: Any) -> str:
    """Location of a nested model with its class name, e.g. ``flows[0] (FlowDefinition)``."""
    if location and _is_model(model):
        return f"{location} ({model.__name__})"
    return location


def _unknown_argument_problem(parent: Any, parent_location: str, argument: str) -> str:
    """Describe an unknown argument, with the closest valid argument name as suggestion.

    Args:
        parent: The model which does not accept the argument, or None if it is not known.
        parent_location: Location of this model within the outermost model, or "" if it is the
            outermost model itself.
        argument: The name of the unknown argument.
    """
    message = f"- '{argument}' is not a valid argument"
    message += f" of {_model_label(parent_location, parent)}." if parent_location else "."
    if not _is_model(parent):
        return message
    valid = _argument_names(parent)
    suggestions = difflib.get_close_matches(argument, valid, n=1)
    if suggestions:
        message += f" Did you mean '{suggestions[0]}'?"
    return message + f"\n  Valid arguments: {', '.join(valid)}."


def _union_problem(model: type[PydanticBaseModel], field: str, field_errors: list[Any]) -> str:
    """Describe a value that is valid as none of the types allowed for a field.

    Pydantic tries each type of the union and reports the errors of each try, with a label of
    the type as second location item. If a try failed only because the value has another type,
    it is summarized in the list of allowed types. Other errors, e.g. a failed check of a
    FlodymArray created from a dictionary, are listed below, with the type tried.
    """
    union_args = _union_args(model, field)
    allowed = [_TYPE_NAMES.get(t, f"a {getattr(t, '__name__', t)}") for t in union_args]
    allowed_text = (
        ", ".join(allowed[:-1]) + " or " + allowed[-1] if len(allowed) > 1 else allowed[0]
    )
    message = f"- {field}: must be {allowed_text}."
    given = [e["input"] for e in field_errors if len(e["loc"]) == 2]
    if given:
        message += f" Given: {_describe_value(given[0])}."

    errors_by_try: dict[Any, list[Any]] = {}
    for details in field_errors:
        errors_by_try.setdefault(details["loc"][1], []).append(details)
    # pydantic tries the types in the order of the annotation
    tried_types = union_args if len(errors_by_try) == len(union_args) else errors_by_try
    for tried_type, (label, errors) in zip(tried_types, errors_by_try.items(), strict=False):
        if all(_is_type_mismatch(details) for details in errors):
            continue
        type_name = getattr(tried_type, "__name__", label)
        for details in errors:
            problem = _problem(tried_type, {**details, "loc": details["loc"][2:]})
            message += f"\n  - as {type_name}: " + _indent(problem.removeprefix("- "))
    return message


def _union_field(model: type[PydanticBaseModel], details: Any, errors: list[Any]) -> str | None:
    """The field of an error if its errors are pydantic's errors for the types of a union.

    Pydantic reports the errors for each type of the union, with the type as second location
    item.
    """
    loc = details["loc"]
    if not loc or not isinstance(loc[0], str) or len(_union_args(model, loc[0])) < 2:
        return None
    field_errors = [e for e in errors if e["loc"][:1] == loc[:1]]
    if len(field_errors) > 1 and all(len(e["loc"]) >= 2 for e in field_errors):
        return loc[0]
    return None


def _is_type_mismatch(details: Any) -> bool:
    """Whether an error only says that the value has another type than the one tried."""
    error_type = details["type"]
    return len(details["loc"]) == 2 and (
        error_type in ("is_instance_of", "is_subclass_of", "model_type", "model_attributes_type")
        or error_type.endswith(("_type", "_parsing"))
    )


def _union_args(model: type[PydanticBaseModel], field: str) -> list[Any]:
    """Types allowed for a field, if its annotation is a union; else an empty list.

    None is left out, since pydantic does not report an error for it.
    """
    if field not in model.model_fields:
        return []
    annotation = model.model_fields[field].annotation
    if typing.get_origin(annotation) not in (typing.Union, types.UnionType):
        return []
    return [arg for arg in typing.get_args(annotation) if arg is not type(None)]


def _argument_names(model: type[PydanticBaseModel]) -> list[str]:
    """Names of all arguments accepted by a model, including aliases."""
    return list(_fields_by_argument(model))


def _fields_by_argument(model: type[PydanticBaseModel]) -> dict[str, FieldInfo]:
    """Fields of a model by the argument names which set them, including aliases."""
    fields: dict[str, FieldInfo] = {}
    for name, field in model.model_fields.items():
        fields[name] = field
        if isinstance(field.alias, str):
            fields[field.alias] = field
        if isinstance(field.validation_alias, str):
            fields[field.validation_alias] = field
        elif isinstance(field.validation_alias, AliasChoices):
            fields.update({c: field for c in field.validation_alias.choices if isinstance(c, str)})
    return fields


def _walk(model: type[PydanticBaseModel], loc: tuple[int | str, ...]) -> tuple[str, Any]:
    """Follow the location of an error through the types of the fields.

    Pydantic's internal labels in the location, such as ``function-after[...]``, are left out.

    Returns:
        The location as text, e.g. ``flows[0].dim_letters`` or ``parameters['yield']``, and the
        type at this location, or None if it is not known.
    """
    text = ""
    current: Any = model
    for item in loc:
        if isinstance(item, str) and ("[" in item or "(" in item):
            continue
        current = _unwrap(current)
        if _is_model(current) and isinstance(item, str):
            text += f".{item}" if text else item
            field = _fields_by_argument(current).get(item)
            current = field.annotation if field is not None else None
        else:
            text += f"[{item}]" if isinstance(item, int) else f"[{item!r}]"
            current = _item_type(current)
    return text, _unwrap(current)


def _unwrap(annotation: Any) -> Any:
    """Remove ``Annotated`` and ``None`` from a type annotation.

    Of a union of several types, the first model or container type is kept, since only these
    can contain nested errors.
    """
    if typing.get_origin(annotation) is typing.Annotated:
        return _unwrap(typing.get_args(annotation)[0])
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        nested = [arg for arg in args if _is_model(arg) or typing.get_args(arg)]
        if len(args) == 1 or nested:
            return _unwrap((nested or args)[0])
    return annotation


def _item_type(annotation: Any) -> Any:
    """Type of the items of a container type, e.g. of the values of a dictionary."""
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if not args:
        return None
    if isinstance(origin, type) and issubclass(origin, Mapping):
        return args[-1]
    return args[0]


def _is_model(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, PydanticBaseModel)


def _describe_value(value: Any) -> str:
    """Short description of a given value, with its type.

    Arrays, dimension sets and dictionaries are described by their structure, which is shorter
    and more informative than a shortened representation.
    """
    # imported here, since these modules import this one
    from flodym.dimensions import DimensionSet
    from flodym.flodym_arrays import FlodymArray

    type_name = _qualified_name(type(value))
    if isinstance(value, np.ndarray):
        return f"{type_name} with shape {value.shape} and dtype {value.dtype}"
    if isinstance(value, FlodymArray):
        letters = ", ".join(value.dims.letters)
        return f"{type_name} '{value.name}' with dims ({letters}) and shape {value.shape}"
    if isinstance(value, DimensionSet):
        return f"{type_name} with dims ({', '.join(value.letters)})"
    if isinstance(getattr(value, "shape", None), tuple):
        # other array-like objects, e.g. pandas.DataFrame
        return f"{type_name} with shape {value.shape}"
    if isinstance(value, Mapping):
        if not value:
            return f"empty {type_name}"
        return f"{type_name} with keys {_shorten(', '.join(repr(key) for key in value))}"
    # in one line, so that the item of the error message is not broken up
    text = _shorten(" ".join(repr(value).split()))
    if text.endswith("...") and hasattr(value, "__len__"):
        return f"{text} (type {type_name}, length {len(value)})"
    return f"{text} (type {type_name})"


def _shorten(text: str) -> str:
    if len(text) > _MAX_INPUT_LENGTH:
        return text[: _MAX_INPUT_LENGTH - 3] + "..."
    return text


def _qualified_name(cls: type) -> str:
    """Name of a class with its module, e.g. ``pandas.DataFrame``; built-in classes without.

    The top-level package is used as module if the class is available there, as for most
    public classes of e.g. pandas, numpy and flodym.
    """
    module = cls.__module__
    if module == "builtins":
        return cls.__qualname__
    package = module.split(".")[0]
    if getattr(sys.modules.get(package), cls.__qualname__, None) is cls:
        return f"{package}.{cls.__qualname__}"
    return f"{module}.{cls.__qualname__}"


def _indent(text: str) -> str:
    """Indent all lines but the first, so that multi-line messages stay within their item."""
    return text.replace("\n", "\n  ")
