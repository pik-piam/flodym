"""Home to definition classes.

These are used when defining the MFA system, and can be used to check the input data
and put it into objects with the desired properties.
"""

from typing import Any

import numpy as np
import pandas as pd
from pydantic import (
    AliasChoices,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from ._base_model import FlodymBaseModel


class DimensionDefinition(FlodymBaseModel):
    """Define the model dimensions.

    **Examples**

        >>> import flodym as fd
        >>> time_definition = fd.DimensionDefinition(name="Time", letter="t", dtype=int)
        >>> region_definition = fd.DimensionDefinition(name="Region", letter="r", dtype=str)
    """

    name: str = Field(..., min_length=2)
    """Name of the dimension, with at least two characters."""
    letter: str = Field(
        ..., min_length=1, max_length=1, validation_alias=AliasChoices("letter", "dim_letter")
    )
    """Single-character key of the dimension. Can also be given as ``dim_letter``."""
    dtype: type
    """Data type of the dimension items, e.g. ``int`` or ``str``."""


class _DefinitionWithDimLetters(FlodymBaseModel):
    """Base class for any definition that contains dimension letters."""

    dim_letters: tuple[str, ...]
    """Letters of the dimensions that the object is defined on."""

    @field_validator("dim_letters", mode="before")
    def _check_dimensions(cls, v: Any) -> Any:
        """Check that all dimension letters are single-character strings."""
        for letter in v:
            if (not isinstance(letter, str)) or (len(letter) != 1):
                raise ValueError("Dimensions must be defined using single digit letters")
        return v


class FlowDefinition(_DefinitionWithDimLetters):
    """Define the model flows.

    **Examples**

        >>> import flodym as fd
        >>> flow_one = fd.FlowDefinition(
        ...     from_process_name="fabrication", to_process_name="use", dim_letters=("r", "t")
        ... )
        >>> flow_two = fd.FlowDefinition(
        ...     from_process_name="use", to_process_name="end_of_life", dim_letters=("r", "t")
        ... )

    These are then used in the :py:class:`flodym.MFADefinition`, for creating a custom MFA System.
    """

    from_process_name: str = Field(
        validation_alias=AliasChoices("from_process_name", "from_process")
    )
    """Process from which the flow originates."""
    to_process_name: str = Field(validation_alias=AliasChoices("to_process_name", "to_process"))
    """Process to which the flow goes."""
    name_override: str | None = None
    """Optional name for the flow.

    Will be generated from the connecting process names if not provided.
    """
    prescribed: str | None = None
    """Optional name of a parameter whose values the flow is set to. The parameter is linked to
    the flow by :py:meth:`flodym.MFASystem.link_parameters`, and its values are copied into the
    flow by :py:meth:`flodym.MFASystem.set_arrays_from_prescribed`."""


class StockDefinition(_DefinitionWithDimLetters):
    """Define the model stocks.

    The arrays of the stock (stock, inflow, outflow, and inflow by cohort) can be prescribed by
    parameters, see :py:meth:`flodym.MFASystem.link_parameters` and
    :py:meth:`flodym.MFASystem.set_arrays_from_prescribed`. For a dynamic stock
    model, at most one of stock, inflow and inflow by cohort can be given, as it determines the
    computation mode (see :py:meth:`flodym.DynamicStockModel.compute`).
    """

    model_config = ConfigDict(validate_by_name=True)

    name: str = "undefined stock"
    """Name of the stock."""
    process_name: str | None = Field(default=None, alias="process")
    """Name of the process to which the stock is connected."""
    time_letter: str = "t"
    """Letter of the time dimension, to ensure it's the first appearing in dim_letters."""
    subclass: type
    """type of stock. Can be any found in :py:data:`flodym.stocks`."""
    lifetime_model_class: type | None = None
    """Lifetime model used for the stock. Only needed if type is not simple_flow_driven.
    Available lifetime models can be found in :py:data:`flodym.lifetime_models`.
    """
    solver: str | None = "manual"
    """Deprecated and ignored. The stock-driven DSM always uses the former "manual" algorithm."""
    cohort_letter: str | None = None
    """Letter of the cohort dimension of a dynamic stock model (see
    :py:attr:`flodym.DynamicStockModel.cohort_dim`). It must be a defined dimension with the same
    items as the time dimension, but not one of ``dim_letters``."""
    prescribed_stock: str | None = None
    """Optional name of a parameter whose values the stock is set to."""
    prescribed_inflow: str | None = None
    """Optional name of a parameter whose values the stock inflow is set to."""
    prescribed_outflow: str | None = None
    """Optional name of a parameter whose values the stock outflow is set to."""
    prescribed_inflow_by_cohort: str | None = None
    """Optional name of a parameter whose values the inflow by cohort of a dynamic stock model is
    set to. Requires ``cohort_letter``."""

    @model_validator(mode="after")
    def _check_prescribed_arrays(self) -> Self:
        """Check the consistency of cohort and prescribed array settings.

        Checks that ``cohort_letter`` and ``prescribed_inflow_by_cohort`` are only given for
        dynamic stock models, and that at most one of the arrays which determine the computation
        mode of a dynamic stock model is prescribed.
        """
        is_dynamic = "cohort_dim" in self.subclass.model_fields
        if self.cohort_letter is not None and not is_dynamic:
            raise ValueError(
                f"Stock '{self.name}': cohort_letter is given, but subclass {self.subclass} has "
                "no cohort dimension."
            )
        if self.prescribed_inflow_by_cohort is not None and self.cohort_letter is None:
            raise ValueError(
                f"Stock '{self.name}': prescribed_inflow_by_cohort requires cohort_letter."
            )
        driving = [
            n for n in ("stock", "inflow", "inflow_by_cohort") if n in self.prescribed_arrays
        ]
        if is_dynamic and len(driving) > 1:
            raise ValueError(
                f"Stock '{self.name}': For a dynamic stock model, at most one of the stock, "
                f"inflow and inflow by cohort can be prescribed, but got {driving}."
            )
        return self

    @model_validator(mode="after")
    def _init_solver(self) -> Self:
        """Check that the deprecated ``solver`` is ``"manual"`` or ``"lapack"``."""
        if self.solver not in ["manual", "lapack"]:
            raise ValueError("Solver must be either 'manual' or 'lapack'.")
        return self

    @model_validator(mode="after")
    def _check_lifetime_model(self) -> Self:
        """Check that a lifetime model class is given if and only if the subclass uses one."""
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

    @property
    def prescribed_arrays(self) -> dict[str, str]:
        """Names of the parameters which stock arrays are prescribed by.

        Keyed by array name (``"stock"``, ``"inflow"``, ``"outflow"``, ``"inflow_by_cohort"``).
        """
        parameters = {
            "stock": self.prescribed_stock,
            "inflow": self.prescribed_inflow,
            "outflow": self.prescribed_outflow,
            "inflow_by_cohort": self.prescribed_inflow_by_cohort,
        }
        return {name: p for name, p in parameters.items() if p is not None}


class ParameterDefinition(_DefinitionWithDimLetters):
    """Define the model parameters."""

    name: str
    """Name of the parameter."""
    extensive: bool | None = None
    """Whether the parameter is extensive, see :py:attr:`flodym.FlodymArray.extensive`. If
    None, it is set by :py:meth:`flodym.MFASystem.link_parameters` for parameters used as shares,
    dimension splitters or prescribed arrays."""


class ProcessDefinition(FlodymBaseModel):
    """Define a process.

    A process can be defined optionally with an explicit ID and with parameters used by
    :py:meth:`flodym.Process.compute`. In :py:attr:`MFADefinition.processes`, it can be used
    instead of a process name, see :py:func:`flodym.make_processes`.

    In- and outflows in ``inflow_shares`` and ``outflow_shares`` are given either by flow name,
    or by the name of the process they come from or go to, if only one flow connects the two
    processes.

    **Examples**

        >>> import flodym as fd
        >>> fd.ProcessDefinition(name="shredder", outflow_shares={"remelting": "shredder yield"})
    """

    name: str
    """Name of the process."""
    id: int | None = None
    """ID of the process. If None, it is assigned automatically, see
    :py:func:`flodym.make_processes`."""
    inflow_shares: dict[str, str] = {}
    """Names of the parameters giving the share of an inflow in the total throughput of the
    process, keyed by flow name or by the name of the process the inflow comes from."""
    outflow_shares: dict[str, str] = {}
    """Names of the parameters giving the share of an outflow in the total throughput of the
    process, keyed by flow name or by the name of the process the outflow goes to."""
    forward_dimension_splitter: str | None = None
    """Name of the parameter by which the total throughput of the process is multiplied before
    computing outflows, to add dimensions which the total does not have, but the outflows do
    (see :py:meth:`flodym.Process.compute`). It must sum to one over the added dimensions."""
    backward_dimension_splitter: str | None = None
    """Name of the parameter by which the total throughput of the process is multiplied before
    computing inflows, to add dimensions which the total does not have, but the inflows do
    (see :py:meth:`flodym.Process.compute`). It must sum to one over the added dimensions."""


class MFADefinition(FlodymBaseModel):
    """All the information needed to define an MFA system.

    It is compiled of lists of definition objects.
    """

    dimensions: list[DimensionDefinition]
    """List of definitions of dimensions used in the model."""
    processes: list[str | ProcessDefinition]
    """List of the processes used in the model, given as names or process definitions."""
    flows: list[FlowDefinition]
    """List of definitions of flows used in the model."""
    stocks: list[StockDefinition] = []
    """List of definitions of stocks used in the model."""
    parameters: list[ParameterDefinition] = []
    """List of definitions of parameters used in the model."""
    check_parameter_names: bool = True
    """Whether to check that all parameters used in flow, stock and process definitions are
    defined in :py:attr:`parameters`. Set it to False to use parameters which are added to
    :py:attr:`flodym.MFASystem.parameters` later, before calling
    :py:meth:`flodym.MFASystem.link_parameters`."""

    @model_validator(mode="after")
    def _check_dimension_letters(self) -> Self:
        """Check that dimension letters of flows, stocks and parameters are defined dimensions."""
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
    def _validate_parameter_names(self) -> Self:
        """Check that parameters used by flow, stock and process definitions are defined.

        Skipped if :py:attr:`check_parameter_names` is False.
        """
        if not self.check_parameter_names:
            return self
        defined_parameters = {p.name for p in self.parameters}
        used_parameters: list[tuple[str, str]] = []
        for flow in self.flows:
            if flow.prescribed is not None:
                used_parameters.append((flow.prescribed, f"flow definition {flow}"))
        for stock in self.stocks:
            for name in stock.prescribed_arrays.values():
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

    def to_dfs(self) -> dict[str, pd.DataFrame]:
        """Export definition information to pandas DataFrames.

        Column names are the field names, rows have the lists.

        Returns:
            A dictionary mapping from definition list names (such as "dimensions") to the
            DataFrames.
        """
        all_dfs: dict[str, pd.DataFrame] = {}
        for field_name, def_list in self.model_dump().items():
            # skip options such as check_parameter_names, and empty lists
            if not isinstance(def_list, list) or not def_list:
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
