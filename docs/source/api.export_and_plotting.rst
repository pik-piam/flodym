Export and plotting classes and functions
-----------------------------------------

.. autopydantic_model:: flodym.export::PlotlyArrayPlotter
   :inherited-members: BaseModel

.. autopydantic_model:: flodym.export::PyplotArrayPlotter
   :inherited-members: BaseModel

.. autopydantic_model:: flodym.export::PlotlySankeyPlotter
   :inherited-members: BaseModel

.. autopydantic_model:: flodym.export::PlotlyProcessGraphPlotter
   :inherited-members: BaseModel

.. autopydantic_model:: flodym.export::GraphvizProcessGraphPlotter
   :inherited-members: BaseModel

.. autofunction:: flodym.export.export_mfa_flows_to_csv

.. autofunction:: flodym.export.export_mfa_stocks_to_csv

.. autofunction:: flodym.export.export_mfa_to_pickle

.. autofunction:: flodym.export.convert_to_dict

Base classes
^^^^^^^^^^^^

These appear in type hints and as base classes of the plotters above. They are not meant to be
used directly.

.. py:class:: flodym.export.ArrayPlotter

   Base class of :py:class:`flodym.export.PyplotArrayPlotter` and
   :py:class:`flodym.export.PlotlyArrayPlotter`.

.. py:class:: flodym.export.ProcessGraphPlotter

   Base class of :py:class:`flodym.export.PlotlyProcessGraphPlotter` and
   :py:class:`flodym.export.GraphvizProcessGraphPlotter`.
