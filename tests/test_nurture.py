import numpy as np

from flodym import Dimension, DimensionSet
from flodym import FlodymArray, StockArray
from flodym import InflowDrivenDSM
from flodym import WeibullLifetime
from flodym.export import PlotlyArrayPlotter

EXT_FAC = 2

dim_list = [
    Dimension(
        name="time",
        letter="t",
        items=list(range(31)),
        dtype=int,
    ),
    Dimension(
        name="product",
        letter="p",
        items=["Base", "All ext", "Sudden ext", "Smooth ext"],
        dtype=str,
    ),
]

dims = DimensionSet(dim_list=dim_list)

inflow = StockArray(dims=dims)
inflow[{"t": 0}] = 1


factor = FlodymArray(dims=dims)
factor[...] = 1.0
factor["All ext"] = EXT_FAC
factor[{"p": "Sudden ext", "t": range(10, 31)}] = EXT_FAC
# blend from 5 to 14 years for product D
x_clip = np.clip((np.arange(31) - 5) / 10, 0, 1)
factor["Smooth ext"] = 1 + (EXT_FAC - 1) * x_clip

lifetime_model = WeibullLifetime(
    dims=dims,
    time_letter="t",
    weibull_scale=10,
    weibull_shape=2,
    lt_factor_by_year=factor,
)

dsm = InflowDrivenDSM(
    dims=dims,
    inflow=inflow,
    lifetime_model=lifetime_model,
    time_letter="t",
)
dsm.compute()

plotter = PlotlyArrayPlotter(
    array=factor,
    intra_line_dim="t",
    linecolor_dim="p",
)
fig = plotter.plot()
fig.show()

plotter = PlotlyArrayPlotter(
    array=dsm.stock,
    intra_line_dim="t",
    linecolor_dim="p",
)
fig = plotter.plot()
fig.show()


plotter = PlotlyArrayPlotter(
    array=dsm.outflow,
    intra_line_dim="t",
    linecolor_dim="p",
)
fig = plotter.plot()
fig.show()
