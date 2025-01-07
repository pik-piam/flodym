"""Home to definition classes.

These are used when defining the MFA system, and can be used to check the input data
and put it into ojects with the desired properties.
"""

import numpy as np
from pydantic import (
    BaseModel as PydanticBaseModel,
    AliasChoices,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)
from typing import List, Optional


class DimensionDefinition(PydanticBaseModel):
    """Define the model dimensions.

    **Examples**

        >>> from sodym import DimensionDefinition
        >>> time_definition = DimensionDefinition(name='Time', letter='t', dtype=int)
        >>> region_definition = DimensionDefinition(name='Region', letter='r', dtype=str)
    """

    model_config = ConfigDict(protected_namespaces=())

    name: str = Field(..., min_length=2)
    letter: str = Field(
        ..., min_length=1, max_length=1, validation_alias=AliasChoices("letter", "dim_letter")
    )
    dtype: type


class DefinitionWithDimLetters(PydanticBaseModel):
    """Base class for any definition that contains dimesnsion letters."""

    model_config = ConfigDict(protected_namespaces=())

    dim_letters: tuple
    """letters of the dimensions that the object is defined on"""

    @field_validator("dim_letters", mode="before")
    def check_dimensions(cls, v):
        for letter in v:
            if (not isinstance(letter, str)) or (len(letter) != 1):
                raise ValueError("Dimensions must be defined using single digit letters")
        return v


class FlowDefinition(DefinitionWithDimLetters):
    """Define the model flows.

    **Examples**

        >>> from sodym import FlowDefinition
        >>> flow_one = FlowDefinition(from_process_name='fabrication', to_process_name='use', dim_letters=('r', 't'))
        >>> flow_two = FlowDefinition(from_process_name='use', to_process_name='end_of_life', dim_letters=('r', 't'))

    These are then used in the :py:class:MFADefinition, for creating a custom MFA System.
    """

    from_process_name: str = Field(
        validation_alias=AliasChoices("from_process_name", "from_process")
    )
    """Process from which the flow originates."""
    to_process_name: str = Field(validation_alias=AliasChoices("to_process_name", "to_process"))
    """Process to which the flow goes."""
    name_override: Optional[str] = None
    """Optional name for the flow. Will be generated from the connecting process names if not provided."""


class StockDefinition(DefinitionWithDimLetters):
    """Define the model stocks."""

    name: str = "undefined stock"
    """Name of the stock."""
    process_name: str = Field(
        default="undefined process", validation_alias=AliasChoices("process", "process_name")
    )
    """Name of the process to which the stock is connected."""


class ParameterDefinition(DefinitionWithDimLetters):
    """Define the model parameters."""

    name: str
    """Name of the parameter."""


class MFADefinition(PydanticBaseModel):
    """All the information needed to define an MFA system, compiled of lists of definition objects."""

    model_config = ConfigDict(protected_namespaces=())

    dimensions: List[DimensionDefinition]
    """List of definitions of dimensions used in the model."""
    processes: List[str]
    """List of process names used in the model."""
    flows: List[FlowDefinition]
    """List of definitions of flows used in the model."""
    stocks: List[StockDefinition]
    """List of definitions of stocks used in the model."""
    parameters: List[ParameterDefinition]
    """List of definitions of parameters used in the model."""

    @model_validator(mode="after")
    def check_dimension_letters(self):
        """Check that dimension letters used for flows, stocks and parameters are part of the defined dimensions."""
        defined_dim_letters = [dd.letter for dd in self.dimensions]
        for item in self.flows + self.stocks + self.parameters:
            correct_dims = [letter in defined_dim_letters for letter in item.dim_letters]
            if not np.all(correct_dims):
                raise ValueError(f"Undefined dimension in {item}")
        return self
