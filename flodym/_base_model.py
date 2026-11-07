"""Common base class of all flodym pydantic models."""

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict


class FlodymBaseModel(PydanticBaseModel):
    """Base class of all flodym pydantic models, with a shared configuration.

    Unknown arguments raise a validation error instead of being ignored, so that typos in argument
    names are caught. Fields may have arbitrary types, e.g. numpy arrays.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
