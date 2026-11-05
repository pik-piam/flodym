"""Home to definition classes.

These are used when defining the MFA system, and can be used to check the input data
and put it into ojects with the desired properties.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import (
    AliasChoices,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic import (
    BaseModel as PydanticBaseModel,
)


class DimensionDefinition(PydanticBaseModel):
    """Define the model dimensions.

    **Examples**

        >>> from flodym import DimensionDefinition
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

    dim_letters: tuple[str, ...]
    """Letters of the dimensions that the object is defined on."""

    @field_validator("dim_letters", mode="before")
    def check_dimensions(cls, v):
        for letter in v:
            if (not isinstance(letter, str)) or (len(letter) != 1):
                raise ValueError("Dimensions must be defined using single digit letters")
        return v


class FlowDefinition(DefinitionWithDimLetters):
    """Define the model flows.

    **Examples**

        >>> from flodym import FlowDefinition
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
    parameter: Optional[str] = None
    """Optional name of a parameter whose values the flow is set to, see
    :py:meth:`flodym.MFASystem.set_arrays_from_parameters`."""


class StockDefinition(DefinitionWithDimLetters):
    """Define the model stocks.

    The arrays of the stock (stock, inflow, outflow, and inflow by cohort) can be set from
    parameters, see :py:meth:`flodym.MFASystem.set_arrays_from_parameters`. For a dynamic stock
    model, at most one of stock, inflow and inflow by cohort can be given, as it determines the
    computation mode (see :py:meth:`flodym.DynamicStockModel.compute`).
    """

    name: str = "undefined stock"
    """Name of the stock."""
    process_name: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("process", "process_name")
    )
    """Name of the process to which the stock is connected."""
    time_letter: str = "t"
    """Letter of the time dimension, to ensure it's the first appearing in dim_letters."""
    subclass: type
    """type of stock. Can be any found in :py:data:`flodym.stocks`."""
    lifetime_model_class: Optional[type] = None
    """Lifetime model used for the stock. Only needed if type is not simple_flow_driven.
    Available lifetime models can be found in :py:data:`flodym.lifetime_models`.
    """
    solver: Optional[str] = "manual"
    """Deprecated and ignored. The stock-driven DSM always uses the former "manual" algorithm."""
    cohort_letter: Optional[str] = None
    """Letter of the cohort dimension of a dynamic stock model (see
    :py:attr:`flodym.DynamicStockModel.cohort_dim`). It must be a defined dimension with the same
    items as the time dimension, but not one of ``dim_letters``."""
    stock_parameter: Optional[str] = None
    """Optional name of a parameter whose values the stock is set to."""
    inflow_parameter: Optional[str] = None
    """Optional name of a parameter whose values the stock inflow is set to."""
    outflow_parameter: Optional[str] = None
    """Optional name of a parameter whose values the stock outflow is set to."""
    inflow_by_cohort_parameter: Optional[str] = None
    """Optional name of a parameter whose values the inflow by cohort of a dynamic stock model is
    set to. Requires ``cohort_letter``."""

    @property
    def array_parameters(self) -> Dict[str, str]:
        """Names of the parameters which stock arrays are set to, keyed by array name
        (``"stock"``, ``"inflow"``, ``"outflow"``, ``"inflow_by_cohort"``)."""
        parameters = {
            "stock": self.stock_parameter,
            "inflow": self.inflow_parameter,
            "outflow": self.outflow_parameter,
            "inflow_by_cohort": self.inflow_by_cohort_parameter,
        }
        return {name: p for name, p in parameters.items() if p is not None}

    @model_validator(mode="after")
    def check_array_parameters(self):
        is_dynamic = "cohort_dim" in self.subclass.model_fields
        if self.cohort_letter is not None and not is_dynamic:
            raise ValueError(
                f"Stock '{self.name}': cohort_letter is given, but subclass {self.subclass} has "
                "no cohort dimension."
            )
        if self.inflow_by_cohort_parameter is not None and self.cohort_letter is None:
            raise ValueError(
                f"Stock '{self.name}': inflow_by_cohort_parameter requires cohort_letter."
            )
        driving = [n for n in ("stock", "inflow", "inflow_by_cohort") if n in self.array_parameters]
        if is_dynamic and len(driving) > 1:
            raise ValueError(
                f"Stock '{self.name}': For a dynamic stock model, at most one of the stock, "
                f"inflow and inflow by cohort can be set from a parameter, but got {driving}."
            )
        return self

    @model_validator(mode="after")
    def init_solver(self):
        if self.solver not in ["manual", "lapack"]:
            raise ValueError("Solver must be either 'manual' or 'lapack'.")
        return self

    @model_validator(mode="after")
    def check_lifetime_model(self):
        if (
            self.lifetime_model_class is not None
            and "lifetime_model" not in self.subclass.model_fields
        ):
            raise ValueError(f"Lifetime model is given, but not used in subclass {self.subclass}.")
        elif self.lifetime_model_class is None and "lifetime_model" in self.subclass.model_fields:
            raise ValueError(
                f"Lifetime model class must be part of definition for given subclass {self.subclass}"
            )
        return self


class ParameterDefinition(DefinitionWithDimLetters):
    """Define the model parameters."""

    name: str
    """Name of the parameter."""


class ProcessDefinition(PydanticBaseModel):
    """Define a process, optionally with an explicit ID and with parameters used by
    :py:meth:`flodym.Process.compute`. In :py:attr:`MFADefinition.processes`, it can be used
    instead of a process name, see :py:func:`flodym.make_processes`.

    In- and outflows in ``inflow_shares`` and ``outflow_shares`` are given either by flow name,
    or by the name of the process they come from or go to, if only one flow connects the two
    processes.

    **Examples**

        >>> from flodym import ProcessDefinition
        >>> ProcessDefinition(name="shredder", outflow_shares={"remelting": "shredder yield"})
    """

    name: str
    """Name of the process."""
    id: Optional[int] = None
    """ID of the process. If None, it is assigned automatically, see
    :py:func:`flodym.make_processes`."""
    inflow_shares: Dict[str, str] = {}
    """Names of the parameters giving the share of an inflow in the total throughput of the
    process, keyed by flow name or by the name of the process the inflow comes from."""
    outflow_shares: Dict[str, str] = {}
    """Names of the parameters giving the share of an outflow in the total throughput of the
    process, keyed by flow name or by the name of the process the outflow goes to."""
    forward_dimension_splitter: Optional[str] = None
    """Name of the parameter by which the total throughput of the process is multiplied before
    computing outflows, to add dimensions which the total does not have, but the outflows do
    (see :py:meth:`flodym.Process.compute`). It must sum to one over the added dimensions."""
    backward_dimension_splitter: Optional[str] = None
    """Name of the parameter by which the total throughput of the process is multiplied before
    computing inflows, to add dimensions which the total does not have, but the inflows do
    (see :py:meth:`flodym.Process.compute`). It must sum to one over the added dimensions."""


class MFADefinition(PydanticBaseModel):
    """All the information needed to define an MFA system, compiled of lists of definition objects."""

    model_config = ConfigDict(protected_namespaces=())

    dimensions: List[DimensionDefinition]
    """List of definitions of dimensions used in the model."""
    processes: List[str | ProcessDefinition]
    """List of the processes used in the model, given as names or process definitions."""
    flows: List[FlowDefinition]
    """List of definitions of flows used in the model."""
    stocks: List[StockDefinition] = []
    """List of definitions of stocks used in the model."""
    parameters: List[ParameterDefinition] = []
    """List of definitions of parameters used in the model."""

    @model_validator(mode="after")
    def check_dimension_letters(self):
        """Check that dimension letters used for flows, stocks and parameters are part of the defined dimensions."""
        defined_dim_letters = [dd.letter for dd in self.dimensions]
        for item in self.flows + self.stocks + self.parameters:
            correct_dims = [letter in defined_dim_letters for letter in item.dim_letters]
            if not np.all(correct_dims):
                raise ValueError(f"Undefined dimension in {item}")
        for stock in self.stocks:
            if stock.cohort_letter is not None and stock.cohort_letter not in defined_dim_letters:
                raise ValueError(f"Undefined cohort dimension in {stock}")
        return self

    @model_validator(mode="after")
    def check_parameter_names(self):
        """Check that parameters used by flow and process definitions are defined."""
        defined_parameters = {p.name for p in self.parameters}
        used_parameters = []
        for flow in self.flows:
            if flow.parameter is not None:
                used_parameters.append((flow.parameter, f"flow definition {flow}"))
        for stock in self.stocks:
            for name in stock.array_parameters.values():
                used_parameters.append((name, f"definition of stock '{stock.name}'"))
        for process in self.processes:
            if isinstance(process, str):
                continue
            used_by = f"definition of process '{process.name}'"
            names = list(process.inflow_shares.values()) + list(process.outflow_shares.values())
            splitters = [process.forward_dimension_splitter, process.backward_dimension_splitter]
            names += [splitter for splitter in splitters if splitter is not None]
            used_parameters += [(name, used_by) for name in names]
        for name, used_by in used_parameters:
            if name not in defined_parameters:
                raise ValueError(f"Parameter '{name}' used in {used_by} is not defined.")
        return self

    def to_dfs(self) -> Dict[str, pd.DataFrame]:
        """Export definition information to pandas DataFrames.
        Column names are the field names, rows have the lists.

        Returns:
            A dictionary mapping from definition list names (such as "dimensions") to the
            DataFrames.
        """
        all_dfs = {}
        for field_name, def_list in self.model_dump().items():
            if not def_list:
                continue
            def_dfs = []
            for definition in def_list:
                if isinstance(definition, str):
                    def_dict = {"name": [definition]}
                else:
                    def_dict = {k: [v] for k, v in definition.items()}
                def_dfs.append(pd.DataFrame.from_dict(def_dict))
            df = pd.concat(def_dfs)
            all_dfs[field_name] = df
        return all_dfs
