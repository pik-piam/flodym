Migrating from v1 to v2
=======================

This page lists the changes in flodym v2 that can affect existing models, and how to adapt a
model to them. New features are listed briefly at the end; they are described in the HOWTOs and
the API reference.

Most models run under v2 without changes. The changes that are most likely to matter are:

- stricter checks of the model structure, which turn silent overwrites into errors
  (see :ref:`v2-structure`);
- a stricter mass balance check (see :ref:`v2-mass-balance`);
- changed results of dynamic stock models with unequal time steps (see :ref:`v2-stocks`).


.. _v2-structure:

Model structure
---------------

flodym v1 silently replaced objects with the same name. flodym v2 raises an error or renames the
objects instead.

- **Duplicate process names** in ``MFADefinition.processes`` or ``make_processes()`` raise an
  error. In v1, the later process replaced the earlier one.
- **Several flows between the same two processes** keep their own entries. If several flows
  without ``name_override`` get the same generated name, the second, third, ... flow get the
  suffix ``__2``, ``__3``, ..., and a warning is logged. In v1, only the last of these flows was kept.

  *Action:* If your model has such flows, check which flow you access under the generated name,
  or give the flows a ``name_override``.
- **A duplicate** ``name_override`` raises an error. In v1, the later flow replaced the earlier one.
- **A flow definition that refers to an undefined process** raises an error that names the
  missing process.
- **Setting up an** ``MFASystem`` raises an error if a flow or stock refers to a process object
  that is not in ``mfa.processes``, or if two stocks refer to the same process. A process can now
  have only one stock.
- **The process named** ``sysenv`` **must have ID 0.** v1 only checked that the process with ID 0
  is named ``sysenv``.


.. _v2-mass-balance:

Mass balance and flow checks
----------------------------

``MFASystem.check_mass_balance()`` is now computed per process and per stock.

Changed behavior:

- It also checks whether each stock is consistent with its own inflow and outflow, including
  stocks that are not connected to a process.
- Each process and stock is checked with its own tolerance, relative to its own values. In v1,
  the tolerance was relative to the largest value of the whole system, so an error in a small
  process could stay undetected.
- NaN values now make the check fail. In v1, NaN values made the check pass, because
  ``nan > tolerance`` is ``False``.
- The stock check raises a ``ValueError`` instead of a ``RuntimeError``. It uses the configured
  tolerance instead of the fixed thresholds 1 and 0.001.
- The stock balance accounts for the length of the time steps. The v1 method
  ``get_stock_balance()`` ignored it, which gave wrong results for unequal time steps.
- ``check_mass_balance()`` and ``check_flows()`` return ``True`` or ``False`` if they do not
  raise an error.

*Action:* Models can fail the check where they passed before. In that case, the check has most
likely found an error in the model. As an example, the check found a wrongly connected stock
inflow in example 2 of flodym.

Tolerances and error handling are set globally in ``flodym.config``, for example:

.. code-block:: python

   import flodym

   flodym.config.relative_tolerance = 1000           # default: 100 (times machine epsilon)
   flodym.config.error_behaviors.mass_balance = "warn"  # default: "error"

   # temporarily, within a block:
   with flodym.config.override(error_behaviors={"mass_balance": "warn"}):
       mfa.check_mass_balance()

Renamed arguments and methods:

.. list-table::
   :header-rows: 1

   * - v1
     - v2
   * - ``MFASystem.check_mass_balance(tolerance, raise_error)``
     - ``MFASystem.check_mass_balance(tolerance, error_behavior)``
   * - ``MFASystem.check_flows(exceptions, raise_error, verbose)``
     - ``MFASystem.check_flows(exceptions, error_behavior, verbose)``
   * - ``Stock.check_stock_balance()``
     - ``Stock.check_mass_balance()``
   * - ``Stock.get_stock_balance()`` (returns ``np.ndarray``)
     - ``Stock.get_mass_balance()`` (returns ``FlodymArray``)

``error_behavior`` takes ``"ignore"``, ``"info"``, ``"warn"`` or ``"error"``. ``raise_error=True``
corresponds to ``"error"``, ``raise_error=False`` to ``"warn"``. The old arguments and methods
still work, but emit a ``DeprecationWarning`` (see :ref:`v2-deprecations`).


.. _v2-stocks:

Stocks and lifetime models
--------------------------

Changed results:

- **Unequal time steps:** The annual outflow of dynamic stock models is now divided by the length
  of the current time step, not by that of the cohort. Only the new values fulfil the mass balance
  over the whole time step. With equal time steps, results are unchanged up to rounding errors.

Removed or changed interface:

- ``StockDrivenDSM(solver=...)`` is removed. The ``solver`` argument is ignored, and
  ``StockDefinition.solver`` is deprecated and ignored. Both solvers gave the same results; the
  former default ``"manual"`` is used.
- ``LifetimeModel.compute_survival_factor()`` is now private. Use the property
  ``lifetime_model.sf``, which is computed when needed, or ``lifetime_model.get_sf()``, which
  returns a ``FlodymArray``.
- ``get_stock_by_cohort()`` and ``get_outflow_by_cohort()`` log a warning if the stock has no
  ``cohort_dim``. They still return numpy arrays in this case. With ``cohort_dim`` (or
  ``StockDefinition.cohort_letter``), they return ``StockArray`` objects.
- ``DynamicStockModel`` is no longer abstract. Its ``compute()`` chooses the compute mode from the
  given arrays. ``InflowDrivenDSM`` and ``StockDrivenDSM`` are used as before.

Only relevant for user-defined subclasses:

- The private methods ``_compute_stock``, ``_compute_outflow``, ``_compute_cohorts_and_inflow``,
  ``_compute_inflow_manual`` and ``_compute_inflow_lapack`` of the dynamic stock models are
  removed. All compute modes use ``DynamicStockModel._compute_cohorts()``.
- ``Stock._check_needed_arrays()`` now runs automatically before ``compute()`` in every subclass.
- The private methods of lifetime models that compute the survival factor have changed
  signatures. Subclasses that only implement ``_survival_by_cohort()`` are not affected.


Arrays: set and unset values
----------------------------

``FlodymArray`` has a new flag ``is_set``. An array is set if values are given at initialization,
or assigned with ``set_values()``, ``set_values_from_df()`` or ``array[...] = ...``.

- ``FlodymArray(dims=d, values=None)`` is unset, the same as omitting ``values``.
- Writing to ``.values`` directly, e.g. ``array.values[0, ...] = 1``, does **not** mark the array
  as set.

With the default ``flodym.config.error_behaviors.unset_array_used = "ignore"``, this changes
nothing. It matters if you enable this check, or if you use the automatic process computation,
which treats unset flows as unknown. Use ``array.mark_set()``, or assign with ``array[...] = ...``,
to mark values written directly to ``.values``.


Automatic process computation
-----------------------------

This is a new feature and does not change existing models. If you use it, note:

- After a computation, processes are checked for consistency, for example whether they are
  mass-balanced. This also reports processes that are not mass-balanced by construction, such as a
  process with an inflow, but neither an outflow nor a stock.
- New options in ``flodym.config.error_behaviors`` control these checks:
  ``process_underdetermined``, ``process_overdetermined``, ``process_shares`` and
  ``unused_dimension_splitter``.
- Parameters used in definitions (shares, dimension splitters, prescribed flows and stock arrays)
  are linked by ``MFASystem.link_parameters()``, which ``from_data_reader()``, ``from_csv()`` and
  ``from_excel()`` call unless ``link=False`` is given. The parameters are stored by reference.
- The values of prescribed flows and stock arrays are copied only by
  ``MFASystem.set_arrays_from_prescribed()``, which ``compute_all_possible()`` calls first. It
  overwrites the values of these arrays. If you compute single processes with
  ``Process.compute()``, call ``set_arrays_from_prescribed()`` before.
- To use parameters that are not part of the definition, e.g. computed from read parameters, set
  ``MFADefinition.check_parameter_names`` to False, create the system with ``link=False``, add the
  parameters to ``mfa.parameters``, and call ``mfa.link_parameters()``.

See HOWTO 05 for details.


.. _v2-deprecations:

Deprecated
----------

The following still work in v2, but emit a ``DeprecationWarning`` and will be removed in v3.0.0:

- the ``raise_error`` argument of ``MFASystem.check_mass_balance()`` and
  ``MFASystem.check_flows()``;
- ``Stock.check_stock_balance()`` and ``Stock.get_stock_balance()``;
- ``StockDefinition.solver``;
- ``LifetimeModel.set_prms()`` (deprecated since v1.1.0): assign the parameters directly, e.g.
  ``lifetime_model.mean = 10``.

The ``naming`` argument of ``make_empty_flows()`` is deprecated as well and will be removed in
v3.0.0. It still works, but logs an info message instead of a warning. Set the global option
``flodym.config.flow_naming`` instead, e.g.
``flodym.config.flow_naming = flodym.flow_naming.process_ids``, or use
``with flodym.config.override(flow_naming=...):``.


New features
------------

- **Automatic process computation:** ``Process.compute()`` and
  ``MFASystem.compute_all_possible()`` compute unknown flows from known flows, shares and
  dimension splitters. Shares and splitters are given in ``ProcessDefinition``; flows and stock
  arrays can be prescribed by parameters with ``FlowDefinition.prescribed`` and
  ``StockDefinition.prescribed_stock``, ``prescribed_inflow``, ``prescribed_outflow`` and
  ``prescribed_inflow_by_cohort``; ``MFASystem.link_parameters()`` and
  ``MFASystem.set_arrays_from_prescribed()`` (HOWTO 05).
- **Process links:** ``process.inflows``, ``process.outflows`` and ``process.stock``;
  ``ProcessDefinition(id, name)`` to set process IDs explicitly; ``MFASystem.relink()``, which by
  default also renames flows named after their processes (``update_names=True``).
- **Flow naming:** the global option ``flodym.config.flow_naming`` sets the function that
  generates flow names from the connected processes.
- **Mass balance per process and stock:** ``Process.get_mass_balance()``,
  ``Stock.get_mass_balance()``, the corresponding ``check_mass_balance()`` methods, and the
  optional check after each computation (``flodym.config.checks.mass_balance_after_compute``).
- **Global configuration:** ``flodym.config`` for tolerances and error behavior.
- **Dynamic stock models:** compute modes ``compute_inflow_driven()``, ``compute_stock_driven()``
  and ``compute_inflow_by_cohort_driven()``, which can be called repeatedly and in any order;
  ``set_initial_stock()``; inflow by cohort (HOWTO 06).
- **Lifetime models:** ``WeibullMeanStdLifetime``, ``ExponentialLifetime``, ``Leaching``,
  ``TabulatedLifetime``, ``CompetingRisksLifetime``; the option ``relative_std``; period effects
  (``hazard_factor_by_year``, ``removal_share_by_year``); ``get_mean_lifetime_by_cohort()``,
  ``get_sf()``, ``get_survival_rate()`` and ``get_pdf()``.
- **Set and unset arrays:** ``FlodymArray.is_set``, ``mark_set()`` and ``mark_unset()``.
- **Attribute access:** ``mfa.flows.my_flow`` in addition to ``mfa.flows["my_flow"]``, also for
  stocks, processes and parameters (``flodym.AttributeDict``).
