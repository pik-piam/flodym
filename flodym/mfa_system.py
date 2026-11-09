"""Home to a base class for MFA systems.

Specific MFA models can be built that inherit from this class.
"""

import logging
import warnings
from os import PathLike
from typing import Any, SupportsFloat

import numpy as np
from pydantic import model_validator
from typing_extensions import Self

from ._attribute_dict import AttributeDict
from ._base_model import FlodymBaseModel
from ._checks import config_hint, report_balance_failures
from ._config import ErrorBehavior, config, handle_error
from .data_reader import (
    CompoundDataReader,
    CSVDimensionReader,
    CSVParameterReader,
    DataReader,
    ExcelDimensionReader,
    ExcelParameterReader,
)
from .dimensions import DimensionSet
from .flodym_arrays import FlodymArray, Flow, Parameter
from .flow_helper import _make_unique_name, make_empty_flows
from .mfa_definition import MFADefinition
from .processes import Process, compute_processes, make_processes
from .stock_helper import make_empty_stocks
from .stocks import Stock

logger = logging.getLogger(__name__)


class MFASystem(FlodymBaseModel):
    """Base class for the calculation of a Material Flow Analysis (MFA) system.

    An MFA system consists of a set of processes, flows, stocks defined over a set of dimensions.
    For the concrete definition of the system, a subclass of MFASystem must be implemented.

    **Example**
    Define your MFA System:

        >>> import flodym as fd
        >>> class CustomMFA(fd.MFASystem):
        ...     def compute(self):
        ...         # do some computations on the CustomMFA attributes: stocks and flows
        ...         pass

    MFA flows, stocks and parameters are defined as instances of subclasses of
    :py:class:`flodym.FlodymArray`. Dimensions are managed with the :py:class:`flodym.Dimension`
    and :py:class:`flodym.DimensionSet`.

    Parameters, processes, flows and stocks can be passed as plain dictionaries. They are converted
    to :py:class:`flodym.AttributeDict` objects, such that their entries can also be accessed as
    attributes, e.g. ``mfa.flows.sysenv__to__process_1``. Dictionaries assigned to these fields
    after initialization are not converted.
    """

    dims: DimensionSet
    """All dimensions that appear in the MFA system."""
    parameters: AttributeDict[Parameter]
    """The parameters of the MFA system,
    as a dictionary mapping the names of the MFA system parameters to the parameters themselves.
    Also accessible as attributes, see :py:class:`flodym.AttributeDict`.
    """
    processes: AttributeDict[Process]
    """The processes of the MFA system, i.e. the nodes of the MFA system graph,
    as a dictionary mapping the names of the MFA system processes to the processes themselves.
    Also accessible as attributes, see :py:class:`flodym.AttributeDict`.
    """
    flows: AttributeDict[Flow]
    """The flows of the MFA system, i.e. the edges of the MFA system graph,
    as a dictionary mapping the names of the MFA system flows to the flows themselves.
    Also accessible as attributes, e.g. ``flows["sysenv => process 1"]`` as
    ``flows.sysenv__to__process_1``, see :py:class:`flodym.AttributeDict`.
    """
    stocks: AttributeDict[Stock] = AttributeDict()
    """The stocks of the MFA system,
    as a dictionary mapping the names of the MFA system stocks to the stocks themselves.
    Also accessible as attributes, see :py:class:`flodym.AttributeDict`.
    """

    @model_validator(mode="after")
    def _link_processes(self) -> Self:
        """Link the processes to the flows and stocks at setup, see :py:meth:`relink`.

        Flow names are not changed.
        """
        self.relink(update_names=False)
        return self

    @classmethod
    def from_data_reader(
        cls, definition: MFADefinition, data_reader: DataReader, link: bool = True
    ) -> Self:
        """Define and set up the MFA system and load all required data.

        Initialises stocks and flows with all zero values.

        Parameters used in the definitions (shares and dimension splitters of processes,
        prescribed flows and stock arrays) are linked by :py:meth:`link_parameters`, unless
        ``link`` is False. Prescribed values are copied into the flows and stock arrays only by
        :py:meth:`set_arrays_from_prescribed`, which :py:meth:`compute_all_possible` calls.

        Args:
            definition: The MFA definition object.
            data_reader: The data reader for dimensions and parameters.
            link: Whether to link the parameters. Set it to False to change
                :py:attr:`parameters` before linking, e.g. to add parameters computed from the
                read ones, and call :py:meth:`link_parameters` afterwards.

        Returns:
            The MFA system.
        """
        dims = data_reader.read_dimensions(definition.dimensions)
        parameters = data_reader.read_parameters(definition.parameters, dims=dims)
        processes = make_processes(definition.processes)
        flows = make_empty_flows(processes=processes, flow_definitions=definition.flows, dims=dims)
        stocks = make_empty_stocks(
            processes=processes, stock_definitions=definition.stocks, dims=dims
        )
        mfa = cls(
            dims=dims,
            parameters=parameters,
            processes=processes,
            flows=flows,
            stocks=stocks,
        )
        if link:
            mfa.link_parameters()
        return mfa

    @classmethod
    def from_csv(
        cls,
        definition: MFADefinition,
        dimension_files: dict[str, str | PathLike[str]],
        parameter_files: dict[str, str | PathLike[str]],
        allow_missing_parameter_values: bool = False,
        allow_extra_parameter_values: bool = False,
        link: bool = True,
    ) -> Self:
        """Define and set up the MFA system and load all required data from CSV files.

        Initialises stocks and flows with all zero values.

        See :py:class:`flodym.CSVDimensionReader` and :py:class:`flodym.CSVParameterReader`, and
        :py:meth:`flodym.FlodymArray.from_df` for expected format.

        Args:
            definition: The MFA definition object.
            dimension_files: A dictionary mapping dimension names to CSV files.
            parameter_files: A dictionary mapping parameter names to CSV files.
            allow_missing_parameter_values: Whether to allow missing values in the parameter data
                (missing rows or empty value cells).
            allow_extra_parameter_values: Whether to allow extra values in the parameter data.
            link: Whether to link the parameters, see :py:meth:`from_data_reader`.

        Returns:
            The MFA system.
        """
        dimension_reader = CSVDimensionReader(
            dimension_files=dimension_files,
        )
        parameter_reader = CSVParameterReader(
            parameter_files=parameter_files,
            allow_missing_values=allow_missing_parameter_values,
            allow_extra_values=allow_extra_parameter_values,
        )
        data_reader = CompoundDataReader(
            dimension_reader=dimension_reader,
            parameter_reader=parameter_reader,
        )
        return cls.from_data_reader(definition, data_reader, link=link)

    @classmethod
    def from_excel(
        cls,
        definition: MFADefinition,
        dimension_files: dict[str, str | PathLike[str]],
        parameter_files: dict[str, str | PathLike[str]],
        dimension_sheets: dict[str, str] | None = None,
        parameter_sheets: dict[str, str] | None = None,
        allow_missing_parameter_values: bool = False,
        allow_extra_parameter_values: bool = False,
        link: bool = True,
    ) -> Self:
        """Define and set up the MFA system and load all required data from Excel files.

        Initialises stocks and flows with all zero values.
        Builds a CompoundDataReader from Excel readers, and calls the from_data_reader class method.

        See :py:class:`flodym.ExcelDimensionReader`,
        :py:class:`flodym.ExcelParameterReader`, and
        :py:meth:`flodym.FlodymArray.from_df` for expected format.

        Args:
            definition: The MFA definition object.
            dimension_files: A dictionary mapping dimension names to Excel files.
            parameter_files: A dictionary mapping parameter names to Excel files.
            dimension_sheets: A dictionary mapping dimension names to sheet names in the Excel
                files.
            parameter_sheets: A dictionary mapping parameter names to sheet names in the Excel
                files.
            allow_missing_parameter_values: Whether to allow missing values in the parameter data
                (missing rows or empty value cells).
            allow_extra_parameter_values: Whether to allow extra values in the parameter data.
            link: Whether to link the parameters, see :py:meth:`from_data_reader`.

        Returns:
            The MFA system.
        """
        dimension_reader = ExcelDimensionReader(
            dimension_files=dimension_files,
            dimension_sheets=dimension_sheets,
        )
        parameter_reader = ExcelParameterReader(
            parameter_files=parameter_files,
            parameter_sheets=parameter_sheets,
            allow_missing_values=allow_missing_parameter_values,
            allow_extra_values=allow_extra_parameter_values,
        )
        data_reader = CompoundDataReader(
            dimension_reader=dimension_reader,
            parameter_reader=parameter_reader,
        )
        return cls.from_data_reader(definition, data_reader, link=link)

    @property
    def total_stock_change(self) -> FlodymArray | float:
        """Mass change of the whole system per year.

        It is the sum of the net inflows (per year) of all stocks connected to a process, reduced
        to the dimensions common to all stocks.
        """
        return sum((s.net_inflow for s in self.stocks.values() if s.process is not None), 0.0)

    @property
    def _absolute_float_precision(self) -> float:
        """Machine epsilon times the largest absolute value of all flows and stocks."""
        precisions = [f._absolute_float_precision for f in self.flows.values()]
        precisions += [s._absolute_float_precision for s in self.stocks.values()]
        return max(precisions, default=0.0)

    @property
    def tolerance(self) -> float:
        """Absolute tolerance based on all flows and stocks of the MFA system.

        See :py:attr:`flodym.Config.relative_tolerance`. Used by :py:meth:`check_flows`.
        """
        return config.get_tolerance(self._absolute_float_precision)

    def relink(self, update_names: bool = True) -> None:
        """Link each process to its inflows, outflows and stock.

        The links are given by the flows and stocks of the MFA system (see
        :py:attr:`flodym.Process.inflows`).

        This is done automatically when the MFA system is set up, without changing flow names.
        Call it again after adding, replacing or removing entries of :py:attr:`flows` or
        :py:attr:`stocks`, or after changing the processes of a flow.
        Existing links of the processes are replaced.

        Args:
            update_names: If True (default), the names of flows that are named after the
                processes they connect, i.e. flows created by :py:func:`flodym.make_empty_flows`
                without ``name_override``, are generated anew from their current processes.
                The flow names, the keys of :py:attr:`flows` and the keys of the process shares
                (see :py:attr:`flodym.Process.inflow_shares`) are updated; their order is kept.
                Names are generated with the function given in the global config option
                :py:attr:`flodym.Config.flow_naming`. If a generated name is already used, a
                suffix ``__2``, ``__3``, ... is appended, and a warning is logged. Other flows keep
                their names and keys.

                Warning: With ``update_names=False``, flows named after processes keep their old
                names after their processes have been changed or renamed. These names are then
                misleading.

        Raises:
            ValueError: If a flow or stock refers to a process object which is not in
                :py:attr:`processes`, or if several stocks refer to the same process.
                In this case, the existing links are left unchanged.
        """
        linked_stocks: dict[str, str] = {}
        for flow in self.flows.values():
            for process in (flow.from_process, flow.to_process):
                self._check_is_own_process(process, f"flow '{flow.name}'")
        for stock in self.stocks.values():
            if stock.process is None:
                continue
            self._check_is_own_process(stock.process, f"stock '{stock.name}'")
            if stock.process.name in linked_stocks:
                raise ValueError(
                    f"Process '{stock.process.name}' cannot have both stocks "
                    f"'{linked_stocks[stock.process.name]}' and '{stock.name}'."
                )
            linked_stocks[stock.process.name] = stock.name

        if update_names:
            self._update_flow_names()
        for process in self.processes.values():
            process.clear_links()
        for flow in self.flows.values():
            flow.from_process.add_outflow(flow)
            flow.to_process.add_inflow(flow)
        for stock in self.stocks.values():
            if stock.process is not None:
                stock.process.set_stock(stock)

    def _update_flow_names(self) -> None:
        """Generate the names of flows named after their processes anew.

        The keys of :py:attr:`flows` and of the process shares are updated accordingly, keeping
        their order. See :py:meth:`relink`.
        """
        fixed_names = {key for key, flow in self.flows.items() if not flow._named_after_processes}
        renamed_flows: dict[str, Flow] = {}
        new_keys: dict[str, str] = {}
        for key, flow in self.flows.items():
            if flow._named_after_processes:
                generated_name = config.flow_naming(flow.from_process, flow.to_process)
                new_key = _make_unique_name(generated_name, fixed_names | renamed_flows.keys())
                flow.name = new_key
                if new_key != key:
                    new_keys[key] = new_key
                key = new_key
            renamed_flows[key] = flow
        # change the existing dictionary instead of replacing it, to keep references to it valid
        self.flows.clear()
        self.flows.update(renamed_flows)
        for process in self.processes.values():
            process._rename_share_keys(new_keys)

    def _check_is_own_process(self, process: Process, used_by: str) -> None:
        """Raise a ValueError if ``process`` is not the process of the MFA system.

        This is the case if it is not the object stored in :py:attr:`processes` under its name.
        ``used_by`` labels the referring flow or stock for the message.
        """
        if self.processes.get(process.name) is not process:
            raise ValueError(
                f"Process '{process.name}' of {used_by} is not a process of the MFA system."
            )

    def compute(self) -> None:
        """Perform all computations for the MFA system.

        This method must be implemented in a subclass of MFASystem.

        Raises:
            NotImplementedError: If it is not overridden in a subclass.
        """
        raise NotImplementedError(
            "The compute method must be implemented in a subclass of MFASystem if it is to be used."
        )

    def get_new_array(
        self, dim_letters: tuple[str, ...] | None = None, **kwargs: Any
    ) -> FlodymArray:
        """Get a new FlodymArray object.

        Args:
            dim_letters: Tuple of dimension letters to include in the new FlodymArray. If None,
                all dimensions are included.
            **kwargs: Keyword arguments to pass to the FlodymArray constructor.

        Returns:
            The new FlodymArray, defined over the selected dimensions of :py:attr:`dims`.
        """
        dims = self.dims.get_subset(dim_letters)
        return FlodymArray(dims=dims, **kwargs)

    def check_mass_balance(
        self,
        tolerance: SupportsFloat | None = None,
        error_behavior: ErrorBehavior | str | None = None,
        raise_error: bool | None = None,
    ) -> bool:
        """Check the mass balance of all processes and stocks of the MFA system.

        - For each process, the sum of inflows minus the sum of outflows must equal the net inflow
          of its stock, or zero if it has no stock
          (see :py:meth:`flodym.Process.check_mass_balance`). For the 'sysenv' process, it must
          equal the negative mass change of the system (see :py:attr:`total_stock_change`).
        - For each stock, inflow, outflow and stock must be consistent
          (see :py:meth:`flodym.Stock.check_mass_balance`).

        Args:
            tolerance: Absolute tolerance for all checks. If None, each process and stock is
                checked with its own tolerance (see :py:attr:`flodym.Process.tolerance` and
                :py:attr:`flodym.Stock.tolerance`), so that small processes and stocks are checked
                as strictly as large ones. The tolerance of 'sysenv' includes the in- and outflows
                of all stocks.
            error_behavior: What to do if the check fails. If None,
                ``flodym.config.error_behaviors.mass_balance`` is used (default: raise an error).
            raise_error: Deprecated, use ``error_behavior`` instead. True corresponds to
                ``"error"``, False to ``"warn"``.

        Returns:
            Whether all checks passed.
        """
        logger.info(f"Checking mass balance of {self.__class__.__name__} object...")

        if tolerance is not None:
            tolerance = float(tolerance)
        error_behavior = _resolve_error_behavior(
            error_behavior, raise_error, config.error_behaviors.mass_balance, "check_mass_balance"
        )

        failures: list[str | None] = []
        for process in self.processes.values():
            if process.id == 0:
                failures.append(
                    process._mass_balance_failure(
                        tolerance=self._sysenv_tolerance(process)
                        if tolerance is None
                        else tolerance,
                        mass_change=-self.total_stock_change,
                    )
                )
            else:
                failures.append(process._mass_balance_failure(tolerance=tolerance))
        failures += [
            stock._mass_balance_failure(tolerance=tolerance) for stock in self.stocks.values()
        ]

        return report_balance_failures(
            failures,
            error_behavior,
            hint=config_hint("mass_balance", tolerance=True),
            success_message="Success - Mass balance is consistent!",
        )

    def _sysenv_tolerance(self, sysenv: Process) -> float:
        """Tolerance for the mass balance of the 'sysenv' process.

        It includes the in- and outflows of all stocks connected to a process.
        """
        precisions = [sysenv._absolute_float_precision]
        for stock in self.stocks.values():
            if stock.process is not None:
                precisions += [
                    stock.inflow._absolute_float_precision,
                    stock.outflow._absolute_float_precision,
                ]
        return config.get_tolerance(max(precisions))

    def check_flows(
        self,
        exceptions: list[str] | None = None,
        error_behavior: ErrorBehavior | str | None = None,
        verbose: bool = False,
        raise_error: bool | None = None,
    ) -> bool:
        """Check that no flow has NaN values or negative values.

        Negative values are only reported beyond the tolerance of the MFA system, see
        :py:attr:`tolerance`.

        Args:
            exceptions: Names of flows to be excluded from the check. Flows from or to
                processes with names in this list are excluded as well.
            error_behavior: What to do for each flow that fails the check. If None,
                ``flodym.config.error_behaviors.check_flows`` is used (default: log a warning).
            verbose: If True, the message lists the items with negative values.
            raise_error: Deprecated, use ``error_behavior`` instead. True corresponds to
                ``"error"``, False to ``"warn"``.

        Returns:
            Whether all checked flows passed.
        """
        logger.info("Checking flows for NaN and negative values...")
        error_behavior = _resolve_error_behavior(
            error_behavior, raise_error, config.error_behaviors.check_flows, "check_flows"
        )

        if exceptions is None:
            exceptions = []
        flows = [f for f in self.flows.values() if f.name not in exceptions]
        flows = [
            f
            for f in flows
            if f.from_process.name not in exceptions and f.to_process.name not in exceptions
        ]

        all_good = True
        # Check for NaN values
        for flow in flows:
            if np.any(np.isnan(flow.values)):
                message = f"NaN values found in flow {flow.name}!" + config_hint("check_flows")
                handle_error(behavior=error_behavior, message=message)
                all_good = False

        # Check for negative values
        tolerance = self.tolerance
        for flow in flows:
            if np.any(flow.values < -tolerance):
                message = f"Negative value in flow {flow.name}!" + config_hint(
                    "check_flows", tolerance=True
                )
                if verbose:
                    message += "\n Items:"
                    indices = flow.items_where(lambda x: x < -tolerance)
                    for index in indices:
                        message += "\n  " + ", ".join(index)
                handle_error(behavior=error_behavior, message=message)
                all_good = False

        if all_good:
            logger.info(f"Success - No negative flows or NaN values in {self.__class__.__name__}")
        return all_good

    def mark_all_set(self) -> None:
        """Mark all stocks and flows as set."""
        for flow in self.flows.values():
            flow.mark_set()
        for stock in self.stocks.values():
            stock.mark_all_set()

    def mark_all_unset(self) -> None:
        """Mark all stocks and flows as unset."""
        for flow in self.flows.values():
            flow.mark_unset()
        for stock in self.stocks.values():
            stock.mark_all_unset()

    def link_parameters(self) -> None:
        """Link the parameters used in definitions to the objects that use them.

        These are shares and dimension splitters of processes (see
        :py:class:`flodym.ProcessDefinition`), and prescribed flows and stock arrays (see
        :py:attr:`flodym.FlowDefinition.prescribed` and :py:class:`flodym.StockDefinition`).

        When the processes, flows and stocks are created from definitions, they store the
        parameter names. This method replaces the names by the parameters of the same name in
        :py:attr:`parameters`. The parameters are stored by reference, so later changes of their
        values are taken into account. Objects which are already linked are not changed.
        Share keys given as process names are resolved to flow names.

        :py:meth:`from_data_reader` calls this method, unless ``link=False`` is given. This
        allows to change :py:attr:`parameters` before linking, for example:

            >>> mfa = MyMFA.from_data_reader(definition, data_reader, link=False)
            >>> mfa.parameters["c"] = mfa.parameters["a"] + mfa.parameters["b"]
            >>> mfa.link_parameters()

        For parameters which are not part of the definition, such as ``"c"``, set
        :py:attr:`flodym.MFADefinition.check_parameter_names` to False.

        The extensivity (see :py:attr:`flodym.FlodymArray.extensive`) of parameters which are
        linked is set from their use, if it is None: Shares and dimension splitters are
        intensive, parameters prescribing flows or stock arrays have the extensivity of these
        (extensive by default). Conflicting uses, and a given extensivity which contradicts the
        use, are handled according to ``flodym.config.error_behaviors.extensivity``.

        Raises:
            ValueError: If parameters are not found in :py:attr:`parameters` (all are listed),
                if a prescribing parameter lacks dimensions of the array it prescribes, or if a
                share key cannot be resolved to a flow of its process.
        """
        users = self._unlinked_parameter_users()
        missing = [(name, used_by) for name, used_by in users if name not in self.parameters]
        if missing:
            listed = "; ".join(f"'{name}' (used by {used_by})" for name, used_by in missing)
            raise ValueError(f"Parameters not found in MFASystem.parameters: {listed}.")
        self._set_parameter_extensivity()
        for _, array in self._prescribable_arrays():
            if isinstance(array._prescribed, str):
                array._set_prescribed(self.parameters[array._prescribed])
        for process in self.processes.values():
            process._link_parameters(self.parameters)

    def set_arrays_from_prescribed(self) -> None:
        """Copy the values of the prescribing parameters into the prescribed arrays.

        The prescribed arrays are flows and stock arrays (see :py:meth:`link_parameters`). Their
        values are overwritten and they are marked as set.
        Dimensions of a parameter which the array does not have are summed over.
        Called by :py:meth:`compute_all_possible`.

        Raises:
            ValueError: If a prescribing parameter is not linked yet.
        """
        for _, array in self._prescribable_arrays():
            array._set_arrays_from_prescribed()

    def compute_all_possible(self) -> list[str]:
        """Compute all processes which can be computed from the known values.

        Known values are the known flows, shares and stock values, see
        :py:meth:`flodym.Process.compute`.

        Prescribed flows and stock arrays are set to the values of their parameters first (see
        :py:meth:`set_arrays_from_prescribed`), overwriting their values. Then, all processes are
        tried, and each computed process triggers a new try of its neighbors, until no further
        process can be computed (see :py:func:`flodym.compute_processes`).
        Underdetermined processes are skipped.

        Returns:
            The names of the processes (except 'sysenv') which could not be computed.
        """
        self.set_arrays_from_prescribed()
        compute_processes(self.processes.values())
        not_computed = [p.name for p in self.processes.values() if p.id != 0 and not p.is_computed]
        if not_computed:
            logger.info(
                f"compute_all_possible(): Could not compute processes {', '.join(not_computed)}."
            )
        return not_computed

    def _prescribable_arrays(self) -> list[tuple[str, FlodymArray]]:
        """Flows and stock arrays, as tuples of a label for messages and the array."""
        arrays = [(f"flow '{name}'", flow) for name, flow in self.flows.items()]
        for stock_name, stock in self.stocks.items():
            for array_name in ("stock", "inflow", "outflow", "inflow_by_cohort"):
                array = getattr(stock, array_name, None)
                if array is not None:
                    arrays.append((f"{array_name} of stock '{stock_name}'", array))
        return arrays

    def _set_parameter_extensivity(self) -> None:
        """Set the extensivity of unlinked parameters from their use, see link_parameters."""
        required: dict[str, list[tuple[bool, str]]] = {}
        for label, array in self._prescribable_arrays():
            if isinstance(array._prescribed, str) and array.extensive is not None:
                required.setdefault(array._prescribed, []).append(
                    (array.extensive, f"prescribing the {label}")
                )
        for process in self.processes.values():
            for name in process._unlinked_parameter_names():
                required.setdefault(name, []).append(
                    (False, f"as share or dimension splitter of process '{process.name}'")
                )
        for name, uses in required.items():
            parameter = self.parameters[name]
            flags = {flag for flag, _ in uses}
            listed = "; ".join(
                f"{'extensive' if flag else 'intensive'} {used_by}" for flag, used_by in uses
            )
            if len(flags) > 1:
                handle_error(
                    config.error_behaviors.extensivity,
                    message=f"Parameter '{name}' is used with different extensivity: {listed}.",
                )
                continue
            (flag,) = flags
            if parameter._degree is None:
                parameter.extensive = flag
            elif parameter._degree != int(flag):
                handle_error(
                    config.error_behaviors.extensivity,
                    message=(
                        f"Parameter '{name}' has extensive={parameter.extensive}, but is used "
                        f"{listed}."
                    ),
                )

    def _unlinked_parameter_users(self) -> list[tuple[str, str]]:
        """Parameter names which are not linked yet.

        They are given as tuples of the name and a label of the object using it.
        """
        users = [
            (array._prescribed, label)
            for label, array in self._prescribable_arrays()
            if isinstance(array._prescribed, str)
        ]
        for process in self.processes.values():
            users += [
                (name, f"process '{process.name}'") for name in process._unlinked_parameter_names()
            ]
        return users


def _resolve_error_behavior(
    error_behavior: ErrorBehavior | str | bool | None,
    raise_error: bool | None,
    default: ErrorBehavior,
    method_name: str,
) -> ErrorBehavior:
    """Get the error behavior of a check, supporting the deprecated ``raise_error`` argument.

    A bool given as ``error_behavior`` stems from the deprecated positional use of
    ``raise_error`` and is treated as such.
    """
    if isinstance(error_behavior, bool):
        raise_error, error_behavior = error_behavior, None
    if raise_error is not None:
        warnings.warn(
            f"The raise_error argument of MFASystem.{method_name}() is deprecated since v2.0.0. "
            "It will be removed in v3.0.0. Use error_behavior='error' or 'warn' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        if error_behavior is not None:
            raise ValueError("Only one of error_behavior and raise_error may be given.")
        return ErrorBehavior.ERROR if raise_error else ErrorBehavior.WARN
    if error_behavior is None:
        return default
    return ErrorBehavior(error_behavior)
