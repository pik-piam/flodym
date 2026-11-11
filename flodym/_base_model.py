"""Common base class of all flodym pydantic models, and readable validation errors."""

import difflib
import types
import typing
from collections.abc import Mapping
from typing import Any, SupportsFloat

import numpy as np
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, ValidationError
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
        """Details of each problem, as returned by ``pydantic.ValidationError.errors(**kwargs)``."""
        return self.pydantic_error.errors(**kwargs)


FlodymValidationError.__module__ = "flodym"


class FlodymBaseModel(PydanticBaseModel):
    """Base class of all flodym pydantic models, with a shared configuration.

    Unknown arguments raise an error instead of being ignored, so that typos in argument
    names are caught. Fields may have arbitrary types, e.g. numpy arrays.

    Validation errors are raised as :py:class:`flodym.FlodymValidationError`, with a message
    that is easier to read than the pydantic one. Since this class defines ``__init__``, pydantic
    also calls it to validate nested models given as dictionaries, so that nested models format
    their own errors.
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

    @classmethod
    def model_validate(cls, obj: Any, *args: Any, **kwargs: Any) -> Self:
        try:
            return super().model_validate(obj, *args, **kwargs)
        except ValidationError as error:
            readable = _readable_error(cls, error, "Invalid arguments for")
            raise readable from readable.__cause__


class ValidatedAssignmentModel(FlodymBaseModel):
    """Base class of flodym models which also validate values assigned to their fields."""

    model_config = ConfigDict(validate_assignment=True)

    def __setattr__(self, name: str, value: Any) -> None:
        try:
            super().__setattr__(name, value)
        except ValidationError as error:
            readable = _readable_error(type(self), error, "Invalid value assigned to")
            raise readable from readable.__cause__


def _positional_arguments_message(model: type[PydanticBaseModel], n_args: int) -> str:
    """Message for arguments given without name, which flodym classes do not accept."""
    given = "1 argument" if n_args == 1 else f"{n_args} arguments"
    return (
        f"{model.__name__} accepts only keyword arguments, e.g. "
        "SomeClass(argument_1=value_1), "
        "not SomeClass(value_1).\n"
        f"Given: {given} without name. "
        f"Valid arguments: {', '.join(model.model_fields)}."
    )


def _readable_error(
    model: type[PydanticBaseModel], error: ValidationError, title: str
) -> FlodymValidationError:
    """Turn a pydantic validation error into a FlodymValidationError with a readable message.

    If the message cannot be created, the pydantic message is used, so that the cause of the
    error is never hidden. If exactly one check failed, its exception is set as the cause, so
    that raising the returned error ``from`` its cause keeps the traceback of the check.
    """
    errors = error.errors()
    check_errors = [e for e in map(_check_error, errors) if e is not None]
    if len(errors) == 1 and not errors[0]["loc"] and check_errors:
        inner = check_errors[0]
        # raised by __init__ of the same model, which pydantic calls in model_validate
        if (
            isinstance(inner, FlodymValidationError)
            and inner.pydantic_error.title == model.__name__
        ):
            return inner
    try:
        message = _readable_message(model, error, title)
    except Exception:  # noqa: BLE001 - any failure falls back to the pydantic message
        message = str(error)
    readable = FlodymValidationError(message, pydantic_error=error)
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


def _problem(model: type[PydanticBaseModel] | None, details: Any) -> str:
    """Describe a single pydantic error as one item of the error message.

    Args:
        model: The model to which the location of the error refers, if known. Used to suggest
            valid argument names.
        details: The error, as given by ``ValidationError.errors()``.
    """
    error_type = details["type"]
    loc = details["loc"]
    if error_type in ("extra_forbidden", "missing") and loc:
        parent = _location(loc[:-1])
        if error_type == "extra_forbidden":
            valid_names = list(model.model_fields) if model is not None and not parent else []
            return _unknown_argument_problem(str(loc[-1]), parent, valid_names)
        message = f"Missing required argument '{loc[-1]}'."
        return f"- {parent}: {message}" if parent else f"- {message}"
    location = _location(loc)
    check_error = _check_error(details)
    if check_error is not None:
        # an error raised by a check, e.g. of a nested model: its own message is the most precise
        message = str(check_error).strip()
        if type(check_error) not in (ValueError, FlodymValidationError):
            message = f"{type(check_error).__name__}: {message}"
        message = _indent(message)
    else:
        message = f"{details['msg']}. Given: {_describe_value(details['input'])}."
    return f"- {location}: {message}" if location else f"- {message}"


def _unknown_argument_problem(argument: str, parent: str, valid_names: list[str]) -> str:
    """Describe an unknown argument, with the closest valid argument name, if known."""
    message = f"- '{argument}' is not a valid argument" + (f" of {parent}." if parent else ".")
    if not valid_names:
        return message
    suggestions = difflib.get_close_matches(argument, valid_names, n=1)
    if suggestions:
        message += f" Did you mean '{suggestions[0]}'?"
    return message + f"\n  Valid arguments: {', '.join(valid_names)}."


def _union_problem(model: type[PydanticBaseModel], field: str, field_errors: list[Any]) -> str:
    """Describe a value that is valid as none of the types allowed for a field.

    Pydantic reports the errors of trying each type, with a label of the type as second location
    item. Errors which only say that the value has another type are summarized in the list of
    allowed types. Other errors, e.g. of a nested model created from a dictionary, are listed.
    """
    allowed = [_TYPE_NAMES.get(t, f"a {t.__name__}") for t in _union_args(model, field)]
    allowed_text = ", ".join(allowed[:-1]) + " or " + allowed[-1]
    message = f"- {field}: must be {allowed_text}."
    given = [e["input"] for e in field_errors if len(e["loc"]) == 2]
    if given:
        message += f" Given: {_describe_value(given[0])}."
    for details in field_errors:
        if not _is_type_mismatch(details):
            problem = _problem(None, {**details, "loc": details["loc"][2:]})
            message += "\n  " + _indent(problem)
    return message


def _union_field(model: type[PydanticBaseModel], details: Any, errors: list[Any]) -> str | None:
    """The field of an error, if the errors of the field are those of the types of a union."""
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
    """Types allowed for a field other than None, if its annotation is a union, else []."""
    if field not in model.model_fields:
        return []
    annotation = model.model_fields[field].annotation
    if typing.get_origin(annotation) not in (typing.Union, types.UnionType):
        return []
    return [arg for arg in typing.get_args(annotation) if arg is not type(None)]


def _location(loc: tuple[int | str, ...]) -> str:
    """Location of an error, e.g. ``flows[0].dim_letters``, without pydantic's type labels."""
    text = ""
    for item in loc:
        if isinstance(item, int):
            text += f"[{item}]"
        elif "[" not in item and "(" not in item:
            text += f".{item}" if text else item
    return text


def _describe_value(value: Any) -> str:
    """Short description of a given value in one line, with its type."""
    module, type_name = type(value).__module__, type(value).__qualname__
    if module != "builtins":
        type_name = f"{module}.{type_name}"
    if isinstance(value, np.ndarray):
        return f"{type_name} with shape {value.shape} and dtype {value.dtype}"
    if isinstance(getattr(value, "shape", None), tuple):
        # other array-like objects, e.g. FlodymArray or pandas.DataFrame
        return f"{type_name} with shape {value.shape}"
    if isinstance(value, Mapping):
        if not value:
            return f"empty {type_name}"
        return f"{type_name} with keys {_shorten(', '.join(repr(key) for key in value))}"
    text = _shorten(" ".join(repr(value).split()))
    if text.endswith("...") and hasattr(value, "__len__"):
        return f"{text} (type {type_name}, length {len(value)})"
    return f"{text} (type {type_name})"


def _shorten(text: str) -> str:
    return text[: _MAX_INPUT_LENGTH - 3] + "..." if len(text) > _MAX_INPUT_LENGTH else text


def _indent(text: str) -> str:
    """Indent all lines but the first, so that multi-line messages stay within their item."""
    return text.replace("\n", "\n  ")
