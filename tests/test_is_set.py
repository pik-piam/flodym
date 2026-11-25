import logging

import numpy as np
import pytest

from flodym import (
    Dimension,
    DimensionSet,
    FixedLifetime,
    FlodymArray,
    InflowDrivenDSM,
    Parameter,
    SimpleFlowDrivenStock,
    StockArray,
    config,
)

DIMS = DimensionSet(
    dim_list=[
        Dimension(name="time", letter="t", items=[2000, 2001, 2002]),
        Dimension(name="place", letter="p", items=["Earth", "Moon"]),
    ]
)
VALUES = np.arange(6.0).reshape(3, 2) + 1.0


@pytest.fixture
def unset_array_error():
    with config.override(error_behaviors={"unset_array_used": "error"}):
        yield


class TestIsSetInference:
    def test_values_given(self):
        assert FlodymArray(dims=DIMS, values=VALUES).is_set

    def test_values_omitted(self):
        assert not FlodymArray(dims=DIMS).is_set

    def test_values_none(self):
        assert not FlodymArray(dims=DIMS, values=None).is_set

    def test_mark_set_and_unset(self):
        arr = FlodymArray(dims=DIMS, values=VALUES)
        assert arr.mark_unset() is arr
        assert not arr.is_set
        np.testing.assert_array_equal(arr.values, VALUES)  # values are kept
        assert FlodymArray(dims=DIMS).mark_set().is_set

    def test_set_values(self):
        arr = FlodymArray(dims=DIMS)
        arr.set_values(VALUES)
        assert arr.is_set

    def test_setitem(self):
        arr = FlodymArray(dims=DIMS)
        arr[...] = FlodymArray(dims=DIMS, values=VALUES)
        assert arr.is_set

    def test_direct_values_access_does_not_mark_set(self):
        arr = FlodymArray(dims=DIMS)
        arr.values[0, ...] = 1.0
        assert not arr.is_set

    def test_operation_result_is_set(self):
        assert (2 * FlodymArray(dims=DIMS)).is_set

    def test_conversions_keep_flag(self):
        unset = FlodymArray(dims=DIMS)
        assert not unset.to_class(Parameter).is_set
        assert not unset.cast_to(DIMS).is_set
        assert not unset.copy().is_set
        assert FlodymArray(dims=DIMS, values=VALUES).to_class(Parameter).is_set


class TestUnsetArrayUsed:
    def test_default_ignore(self, caplog):
        with caplog.at_level(logging.WARNING):
            2 * FlodymArray(dims=DIMS)
        assert not caplog.records

    def test_warn(self, caplog):
        with (
            config.override(error_behaviors={"unset_array_used": "warn"}),
            caplog.at_level(logging.WARNING),
        ):
            2 * FlodymArray(dims=DIMS, name="foo")
        assert "'foo'" in caplog.text

    @pytest.mark.usefixtures("unset_array_error")
    @pytest.mark.parametrize(
        "operation",
        [
            lambda a: 2 * a,
            lambda a: a + FlodymArray(dims=DIMS, values=VALUES),
            lambda a: -a,
            lambda a: 1 / a,
            lambda a: a.sum_over("p"),
            lambda a: a.sum_to("t"),
            lambda a: a.cast_to(DIMS),
            lambda a: a.abs(),
            lambda a: a[{"p": "Moon"}],
        ],
    )
    def test_error(self, operation):
        with pytest.raises(ValueError, match="not set"):
            operation(FlodymArray(dims=DIMS))

    @pytest.mark.usefixtures("unset_array_error")
    def test_error_on_assignment(self):
        target = FlodymArray(dims=DIMS)
        with pytest.raises(ValueError, match="not set"):
            target[...] = FlodymArray(dims=DIMS)

    @pytest.mark.usefixtures("unset_array_error")
    def test_no_error_for_direct_access_and_export(self):
        arr = FlodymArray(dims=DIMS)
        arr.values[0, ...] = 1.0
        arr.sum_values_to(("t",))
        arr.to_df()

    def test_override_restores_config(self):
        with config.override(error_behaviors={"unset_array_used": "error"}):
            assert config.error_behaviors.unset_array_used == "error"
        assert config.error_behaviors.unset_array_used == "ignore"

    def test_override_validates(self):
        with (
            pytest.raises(ValueError),
            config.override(error_behaviors={"unset_array_used": "nonsense"}),
        ):
            pass
        assert config.error_behaviors.unset_array_used == "ignore"


class TestStocks:
    def test_compute_marks_set(self):
        stock = SimpleFlowDrivenStock(dims=DIMS, inflow=StockArray(dims=DIMS, values=VALUES))
        assert not stock.stock.is_set
        stock.compute()
        assert stock.stock.is_set and stock.inflow.is_set and stock.outflow.is_set

    def test_compute_keeps_docstring(self):
        assert "stocks and outflows" in InflowDrivenDSM.compute.__doc__

    def test_zero_warning_kept(self, caplog):
        stock = InflowDrivenDSM(dims=DIMS, lifetime_model=FixedLifetime(dims=DIMS, mean=1))
        with caplog.at_level(logging.WARNING):
            stock.compute()
        assert "Inflow is zero" in caplog.text

    @pytest.mark.usefixtures("unset_array_error")
    def test_unset_inflow_error(self):
        stock = InflowDrivenDSM(dims=DIMS, lifetime_model=FixedLifetime(dims=DIMS, mean=1))
        stock.inflow.values[...] = VALUES
        with pytest.raises(ValueError, match="not set"):
            stock.compute()
        stock.inflow.mark_set()
        stock.compute()

    def test_user_subclass_is_wrapped(self):
        class MyStock(SimpleFlowDrivenStock):
            def compute(self):
                super().compute()
                self.stock.values[...] *= 2

        stock = MyStock(dims=DIMS, inflow=StockArray(dims=DIMS, values=VALUES))
        stock.compute()
        assert stock.stock.is_set
        np.testing.assert_allclose(stock.stock.values, 2 * np.cumsum(VALUES, axis=0))


def test_unset_parameter_raises_by_default():
    parameter = Parameter(dims=DIMS, name="unset parameter")
    with pytest.raises(ValueError, match="unset parameter.*set_parameter_values"):
        parameter * 2.0
    array = FlodymArray(dims=DIMS)
    array * 2.0  # other arrays follow unset_array_used, which ignores by default


def test_unset_parameter_behavior_configurable():
    parameter = Parameter(dims=DIMS, name="unset parameter")
    with config.override(error_behaviors={"unset_parameter": "ignore"}):
        result = parameter * 2.0
    assert np.all(result.values == 0.0)
