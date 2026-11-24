"""Common base class of all flodym pydantic models."""

from copy import copy, deepcopy
from typing import Any

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict
from pydantic_core import PydanticUndefined
from typing_extensions import Self


class FlodymBaseModel(PydanticBaseModel):
    """Base class of all flodym pydantic models, with a shared configuration.

    Unknown arguments raise a validation error instead of being ignored, so that typos in argument
    names are caught. Fields may have arbitrary types, e.g. numpy arrays.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        """Deep copy in which objects that refer to each other stay connected.

        This is the method of pydantic's ``BaseModel``, except that the copy is registered in
        ``memo`` before its attributes are copied, as ``copy.deepcopy`` does for other objects.
        Otherwise, an object which is reached again through its own attributes, such as a process
        through its flows, is copied a second time.
        """
        if memo is None:
            memo = {}
        new = type(self).__new__(type(self))
        memo[id(self)] = new
        private = getattr(self, "__pydantic_private__", None)
        if private is not None:
            private = deepcopy(
                {name: value for name, value in private.items() if value is not PydanticUndefined},
                memo,
            )
        object.__setattr__(new, "__dict__", deepcopy(self.__dict__, memo))
        object.__setattr__(new, "__pydantic_extra__", deepcopy(self.__pydantic_extra__, memo))
        object.__setattr__(new, "__pydantic_fields_set__", copy(self.__pydantic_fields_set__))
        object.__setattr__(new, "__pydantic_private__", private)
        return new
