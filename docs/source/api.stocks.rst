Stocks and Lifetime Models
------------------------------

.. autopydantic_model:: flodym::Stock
   :inherited-members: BaseModel

.. autopydantic_model:: flodym::SimpleFlowDrivenStock
   :inherited-members: BaseModel

.. autopydantic_model:: flodym::DynamicStockModel
   :inherited-members: BaseModel

.. autopydantic_model:: flodym::InflowDrivenDSM
   :inherited-members: BaseModel

.. autopydantic_model:: flodym::StockDrivenDSM
   :inherited-members: BaseModel

.. autopydantic_model:: flodym::LifetimeModel
   :inherited-members: BaseModel

.. autopydantic_model:: flodym::FixedLifetime
   :inherited-members: BaseModel

.. autopydantic_model:: flodym::NormalLifetime
   :inherited-members: BaseModel

.. autopydantic_model:: flodym::FoldedNormalLifetime
   :inherited-members: BaseModel

.. autopydantic_model:: flodym::LogNormalLifetime
   :inherited-members: BaseModel

.. autopydantic_model:: flodym::WeibullLifetime
   :inherited-members: BaseModel

.. autopydantic_model:: flodym::WeibullMeanStdLifetime
   :inherited-members: BaseModel

.. autopydantic_model:: flodym::ExponentialLifetime
   :inherited-members: BaseModel

.. autopydantic_model:: flodym::Leaching
   :inherited-members: BaseModel

.. autopydantic_model:: flodym::TabulatedLifetime
   :inherited-members: BaseModel

.. autopydantic_model:: flodym::CompetingRisksLifetime
   :inherited-members: BaseModel

.. autofunction:: flodym.make_empty_stocks

Base classes and types
^^^^^^^^^^^^^^^^^^^^^^

These appear in type hints and as base classes. They are not meant to be used directly.

.. py:class:: flodym.StandardDeviationLifetimeModel

   Base class of the lifetime models parametrized by ``mean`` and ``std``:
   :py:class:`flodym.NormalLifetime`, :py:class:`flodym.FoldedNormalLifetime`,
   :py:class:`flodym.LogNormalLifetime` and :py:class:`flodym.WeibullMeanStdLifetime`.

.. py:data:: flodym.LifetimeArrayType

   Types accepted for lifetime model parameters: a :py:class:`flodym.FlodymArray` with a subset of
   the lifetime model dimensions, a numpy array broadcastable to the lifetime model shape, or a
   number.
