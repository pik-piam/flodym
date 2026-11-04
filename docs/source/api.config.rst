Configuration
------------------------------

The global configuration object is available as ``flodym.config``.
Options can be changed by assignment, e.g.
``flodym.config.error_behaviors.unset_array_used = "warn"``,
or temporarily within a ``with flodym.config.override(...):`` block.

The configuration controls

- which checks are performed automatically (:py:class:`flodym.config.Checks`), e.g. the mass
  balance check after each stock or process computation (off by default),
- what happens if a check fails (:py:class:`flodym.config.ErrorBehaviors`): raise an error, log a
  warning or info message, or ignore it. Besides mass balance and flow checks, this concerns the
  automatic computation of processes (:py:meth:`flodym.Process.compute`): underdetermined
  processes, inconsistent shares or dimension splitters, and unused dimension splitters,
- the tolerance of mass balance and other checks (``relative_tolerance`` and
  ``absolute_tolerance`` of :py:class:`flodym.config.Config`).

.. autopydantic_model:: flodym.config::Config

.. autopydantic_model:: flodym.config::Checks

.. autopydantic_model:: flodym.config::ErrorBehaviors

.. autoclass:: flodym.config::ErrorBehavior
   :members:
   :undoc-members:
