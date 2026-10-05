"""Shared type variables and helpers for generic classes."""

from typing import Any

from pydantic import BaseModel
from pydantic.functional_validators import ModelWrapValidatorHandler
from typing_extensions import TypeVar

DimLetterT = TypeVar("DimLetterT", bound=str, default=Any)
"""The dimension letters that various flodym classes can be indexed with.

For example, in a ``MFASystem[Literal["t", "e", "r"]]``, type checkers report
``self.flows["a => b"].sum_over(("x",))`` as an error.
If dimensions are also addressed by name, the names have to be included as well.
"""

OtherDimLetterT = TypeVar("OtherDimLetterT", bound=str)
"""The dimension letters of a second operand, e.g. of ``b`` in ``a * b``."""


def keep_unparametrized_instances(
    cls: type[BaseModel], value: Any, handler: ModelWrapValidatorHandler[Any]
) -> Any:
    """Wrap model validator that accepts instances of the unparametrized class unchanged.

    Where a parametrized class such as ``Flow[Literal["t", "e"]]`` is expected, e.g. in the fields
    of an ``MFASystem[Literal["t", "e"]]``, pydantic would otherwise re-create each passed ``Flow``
    as a new ``Flow[Literal["t", "e"]]`` object. That replaces the passed objects by copies, drops
    subclasses (abstract ones like ``Stock`` even fail), and breaks pickling.
    The dimension letters are only meant for type checkers, so the objects are kept as they are.
    """
    origin = cls.__pydantic_generic_metadata__["origin"]
    if origin is not None and isinstance(value, origin):
        return value
    return handler(value)
