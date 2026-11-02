Configuration
------------------------------

The global configuration object is available as ``flodym.config``.
Options can be changed by assignment, e.g.
``flodym.config.error_behaviors.unset_array_used = "warn"``,
or temporarily within a ``with flodym.config.override(...):`` block.

.. autopydantic_model:: flodym.config::Config

.. autopydantic_model:: flodym.config::ErrorBehaviors

.. autoclass:: flodym.config::ErrorBehavior
   :members:
   :undoc-members:
