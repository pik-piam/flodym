import numpy as np
import pytest

from flodym.dimensions import Dimension, DimensionSet
from flodym.flodym_arrays import StockArray
from flodym.stocks import InflowDrivenDSM, StockDrivenDSM
from flodym.lifetime_models import LogNormalLifetime
from flodym.export import PlotlyArrayPlotter

dim_list = [
    Dimension(
        name="time",
        letter="t",
        items=list(range(1900, 2101)),
        dtype=int,
    ),
    Dimension(
        name="product",
        letter="a",
        items=["automotive", "construction"],
        dtype=str,
    ),
]

dims = DimensionSet(dim_list=dim_list)


def test_stocks():

    inflow_values = np.exp(-np.linspace(-2, 2, 201)**2)
    inflow_values = np.stack([inflow_values, inflow_values]).T
    inflow = StockArray(dims=dims, values=inflow_values)

    lifetime_model = LogNormalLifetime(dims=dims, mean=60, std=25)
    inflow_driven_dsm = InflowDrivenDSM(
        dims=dims,
        inflow=inflow,
        lifetime_model=lifetime_model,
        time_letter="t",
    )
    inflow_driven_dsm.compute()
    stock_fda = inflow_driven_dsm.stock
    stock_driven_dsm = StockDrivenDSM(
        dims=dims,
        stock=stock_fda,
        lifetime_model=lifetime_model,
        time_letter="t",
    )
    stock_driven_dsm.compute()
    inflow_post = stock_driven_dsm.inflow
    assert np.allclose(inflow.values, inflow_post.values)


# if __name__ == "__main__":
#     inflow, inflow_post = test_stocks()
#     plotter = PlotlyArrayPlotter(
#         array=inflow["automotive"],
#         intra_line_dim="time",
#         line_label="Pre",
#     )
#     fig = plotter.plot()
#     plotter = PlotlyArrayPlotter(
#         array=inflow_post["automotive"],
#         intra_line_dim="time",
#         line_label="Post",
#         fig=fig,
#     )
#     fig = plotter.plot()
#     fig.show()

