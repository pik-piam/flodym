Uncertainty and Monte Carlo
------------------------------

Uncertain parameters are declared in an :py:class:`flodym.UncertaintyDefinition`, as
distributions relative to their nominal values. :py:func:`flodym.run_monte_carlo` propagates the
uncertainty through any function of the parameters; :py:func:`flodym.make_model_function` turns
an :py:class:`flodym.MFASystem` into such a function. HOWTO 9 shows how to use them.

Running
^^^^^^^

.. autofunction:: flodym.run_monte_carlo

.. autofunction:: flodym.make_model_function

.. autoclass:: flodym::MonteCarloResult
   :members:
   :exclude-members: model_config

.. autoclass:: flodym::SampledArray
   :members:
   :special-members: __getitem__
   :exclude-members: model_config

Declaration
^^^^^^^^^^^

.. autoclass:: flodym::UncertaintyDefinition
   :members:
   :exclude-members: model_config

.. autoclass:: flodym::ParameterUncertainty
   :members:
   :exclude-members: model_config

Distributions
^^^^^^^^^^^^^

.. autoclass:: flodym::Distribution
   :members:
   :exclude-members: model_config

.. autoclass:: flodym::NormalDistribution
   :members:
   :inherited-members: BaseModel
   :exclude-members: model_config

.. autoclass:: flodym::LogNormalDistribution
   :members:
   :inherited-members: BaseModel
   :exclude-members: model_config

.. autoclass:: flodym::UniformDistribution
   :members:
   :inherited-members: BaseModel
   :exclude-members: model_config

.. autoclass:: flodym::TriangularDistribution
   :members:
   :inherited-members: BaseModel
   :exclude-members: model_config
