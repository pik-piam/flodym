import numpy as np
import pytest

from flodym.dimensions import Dimension, DimensionSet
from flodym.stocks import Stock
from flodym.lifetime_models import NormalLifetime

dim_list = [
    Dimension(
        name="time",
        letter="t",
        items=list(range(2000, 2101)),
        dtype=str,
    ),
    Dimension(
        name="product",
        letter="a",
        items=["construction", "automotive"],
        dtype=str,
    ),
]

dims = DimensionSet(dim_list=dim_list)


def test_stocks():
    pass


def test_wrong_parameter_reader():

    with pytest.raises(ValueError):
        pass
