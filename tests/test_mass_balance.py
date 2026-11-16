import logging

import numpy as np
import pytest

from flodym import (
    Dimension,
    DimensionSet,
    ErrorBehavior,
    Flow,
    InflowDrivenDSM,
    MFASystem,
    NormalLifetime,
    Process,
    SimpleFlowDrivenStock,
    StockArray,
    config,
    make_processes,
)
from flodym._checks import summed_float_precision
from flodym._config import handle_error
from flodym.example_objects import get_example_mfa


def _dims(time_items: list[int] | None = None) -> DimensionSet:
    if time_items is None:
        time_items = list(range(2000, 2010))
    return DimensionSet(
        dim_list=[
            Dimension(letter="t", name="Time", dtype=int, items=time_items),
            Dimension(letter="e", name="Element", dtype=str, items=["Fe", "Cu"]),
        ]
    )


def _simple_stock(dims: DimensionSet | None = None) -> SimpleFlowDrivenStock:
    dims = _dims() if dims is None else dims
    rng = np.random.default_rng(0)
    inflow = StockArray(dims=dims, values=rng.random(dims.shape))
    outflow = StockArray(dims=dims, values=0.5 * rng.random(dims.shape))
    return SimpleFlowDrivenStock(dims=dims, inflow=inflow, outflow=outflow, name="my_stock")


def _computed_example_mfa():
    mfa = get_example_mfa()
    mfa.compute()
    return mfa


# --- config -------------------------------------------------------------------------------------


def test_config_defaults():
    assert config.checks.mass_balance_after_compute is False
    assert config.error_behaviors.mass_balance == ErrorBehavior.ERROR
    assert config.error_behaviors.check_flows == ErrorBehavior.WARN
    assert config.relative_tolerance == 100.0
    assert config.absolute_tolerance is None


def test_config_get_tolerance():
    assert config.get_tolerance(1e-15) == pytest.approx(100 * 1e-15)
    with config.override(relative_tolerance=2.0):
        assert config.get_tolerance(1e-15) == pytest.approx(2e-15)
    with config.override(absolute_tolerance=0.5):
        assert config.get_tolerance(1e-15) == 0.5
    assert config.get_tolerance(1e-15) == pytest.approx(100 * 1e-15)


def test_config_validation():
    with pytest.raises(ValueError):
        config.relative_tolerance = -1.0
    with pytest.raises(ValueError):
        config.error_behaviors.mass_balance = "explode"
    assert config.relative_tolerance == 100.0
    assert config.error_behaviors.mass_balance == ErrorBehavior.ERROR


def test_handle_error(caplog):
    with pytest.raises(ValueError, match="msg"):
        handle_error("error", message="msg")
    with pytest.raises(KeyError):
        handle_error("error", error=KeyError("key"))
    with caplog.at_level(logging.INFO):
        handle_error("warn", message="warn msg")
        handle_error("info", message="info msg")
        handle_error("ignore", message="ignore msg")
    levels = {r.getMessage(): r.levelno for r in caplog.records}
    assert levels == {"warn msg": logging.WARNING, "info msg": logging.INFO}
    with pytest.raises(ValueError, match="Exactly one"):
        handle_error("warn")


# --- stocks -------------------------------------------------------------------------------------


def test_stock_mass_balance_after_compute():
    stock = _simple_stock()
    stock.compute()
    np.testing.assert_allclose(stock.get_mass_balance().values, 0.0, atol=1e-12)
    assert stock.get_mass_balance().dims.letters == stock.dims.letters
    np.testing.assert_allclose(stock.net_inflow.values, stock.inflow.values - stock.outflow.values)
    np.testing.assert_allclose(stock.stock_change.values[0], stock.stock.values[0])
    np.testing.assert_allclose(stock.stock_change.values[1:], np.diff(stock.stock.values, axis=0))
    assert stock.check_mass_balance() is True


def test_stock_mass_balance_uneven_time_steps():
    """Inflows and outflows are annual, so the stock change of a time step is the net inflow
    times the length of the time step."""
    stock = _simple_stock(_dims([2000, 2001, 2005, 2010, 2020]))
    stock.compute()
    assert stock.check_mass_balance(error_behavior="error")
    np.testing.assert_allclose(stock.get_mass_balance().values, 0.0, atol=1e-12)


def test_stock_mass_balance_dsm():
    dims = _dims()
    inflow = StockArray(dims=dims, values=np.ones(dims.shape))
    dsm = InflowDrivenDSM(
        dims=dims,
        inflow=inflow,
        lifetime_model=NormalLifetime(dims=dims, mean=5.0, std=2.0),
    )
    dsm.compute()
    assert dsm.check_mass_balance(error_behavior="error")


def test_stock_mass_balance_failure():
    stock = _simple_stock()
    stock.compute()
    stock.stock.values[3, 0] += 1.0
    np.testing.assert_allclose(np.max(np.abs(stock.get_mass_balance().values)), 1.0)
    with pytest.raises(ValueError, match="Mass balance check failed for stock 'my_stock'"):
        stock.check_mass_balance()
    assert stock.check_mass_balance(error_behavior="ignore") is False
    assert stock.check_mass_balance(tolerance=2.0) is True
    with config.override(absolute_tolerance=2.0):
        assert stock.check_mass_balance() is True
    with config.override(error_behaviors={"mass_balance": "warn"}):
        assert stock.check_mass_balance() is False


@pytest.mark.parametrize(
    "time_items",
    [
        [2020],
        [2020, 2021],
        ["2020", "2021", "2022"],
        [2022, 2021, 2020],
        [2020, 2022, 2021],
        [2020, 2021, 2021],
        list(np.array([2022, 2021, 2020], dtype=np.uint16)),
        [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
    ],
)
def test_stock_mass_balance_needs_time_steps(time_items):
    """Time items from which no time steps can be derived give a clear error."""
    dims = DimensionSet(dim_list=[Dimension(letter="t", name="Time", items=time_items)])
    stock = SimpleFlowDrivenStock(dims=dims, name="my_stock")
    with pytest.raises(ValueError, match="time steps cannot be derived .* 'Time'"):
        stock.get_mass_balance()
    with pytest.raises(ValueError, match="at least three items"):
        stock.check_mass_balance(error_behavior="ignore")


def test_stock_tolerance():
    stock = _simple_stock()
    stock.compute()
    max_value = max(
        np.max(np.abs(stock.inflow.values)),
        np.max(np.abs(stock.outflow.values)),
        np.max(np.abs(stock.stock.values)),
    )
    expected = config.relative_tolerance * np.finfo(np.float64).eps * max_value
    assert stock.tolerance == pytest.approx(expected)


class _InconsistentStock(SimpleFlowDrivenStock):
    def compute(self):
        super().compute()
        self.stock.values[...] *= 2


def _inconsistent_stock() -> _InconsistentStock:
    return _InconsistentStock(
        dims=_dims(), inflow=StockArray(dims=_dims(), values=np.ones(_dims().shape))
    )


def test_automatic_stock_check_off_by_default(caplog):
    stock = _inconsistent_stock()
    with caplog.at_level(logging.INFO):
        stock.compute()
    assert not any("Mass balance" in r.getMessage() for r in caplog.records)
    assert stock.check_mass_balance(error_behavior="ignore") is False


def test_automatic_stock_check_enabled():
    stock = _inconsistent_stock()
    with config.override(checks={"mass_balance_after_compute": True}):
        with pytest.raises(ValueError, match="Mass balance check failed for stock"):
            stock.compute()
        _simple_stock().compute()


def test_automatic_stock_check_error_behavior(caplog):
    stock = _inconsistent_stock()
    with (
        config.override(
            checks={"mass_balance_after_compute": True}, error_behaviors={"mass_balance": "warn"}
        ),
        caplog.at_level(logging.WARNING),
    ):
        stock.compute()
    assert any("Mass balance check failed for stock" in r.getMessage() for r in caplog.records)


def test_automatic_stock_check_runs_once_for_nested_compute(monkeypatch):
    """The check runs only after the outermost compute call, e.g. not after a
    compute_inflow_driven() called by compute()."""
    dims = _dims()
    dsm = InflowDrivenDSM(
        dims=dims,
        inflow=StockArray(dims=dims, values=np.ones(dims.shape)),
        lifetime_model=NormalLifetime(dims=dims, mean=5.0, std=2.0),
    )
    calls = []
    original = InflowDrivenDSM._check_mass_balance
    monkeypatch.setattr(
        InflowDrivenDSM,
        "_check_mass_balance",
        lambda self, *args, **kwargs: (
            calls.append(kwargs["error_behavior"]) or original(self, *args, **kwargs)
        ),
    )
    dsm.compute()
    assert calls == []
    with config.override(checks={"mass_balance_after_compute": True}):
        dsm.compute()
    assert calls == [ErrorBehavior.ERROR]


def test_stock_deprecated_methods():
    stock = _simple_stock()
    stock.compute()
    with pytest.warns(DeprecationWarning, match="check_stock_balance is deprecated"):
        assert stock.check_stock_balance() is True
    with pytest.warns(DeprecationWarning, match="get_stock_balance is deprecated"):
        balance = stock.get_stock_balance()
    assert isinstance(balance, np.ndarray)
    np.testing.assert_allclose(balance, stock.get_mass_balance().values)


# --- processes ----------------------------------------------------------------------------------


def test_process_mass_balance():
    mfa = _computed_example_mfa()
    remelting = mfa.processes["remelting"]
    expected_net_inflow = (
        mfa.flows["shredder => remelting"]
        + mfa.flows["demolition => remelting"]
        - mfa.flows["remelting => sysenv"]
        - mfa.flows["remelting => slag piles"]
    )
    np.testing.assert_allclose(
        remelting.get_net_inflow().values, expected_net_inflow.values, atol=1e-10
    )
    np.testing.assert_allclose(remelting.get_mass_balance().values, 0.0, atol=1e-10)
    assert remelting.check_mass_balance() is True


def test_process_mass_balance_with_stock():
    mfa = _computed_example_mfa()
    landfills = mfa.processes["landfills"]
    np.testing.assert_allclose(
        landfills.get_net_inflow().values, mfa.stocks["landfills"].net_inflow.values
    )
    assert landfills.check_mass_balance() is True

    mfa.stocks["landfills"].outflow.values[...] = 1.0
    with pytest.raises(ValueError, match="Mass balance check failed for process 'landfills'"):
        landfills.check_mass_balance()
    assert landfills.check_mass_balance(error_behavior="ignore") is False


def test_process_mass_balance_sysenv():
    mfa = _computed_example_mfa()
    sysenv = mfa.processes["sysenv"]
    with pytest.raises(ValueError, match="mass_change"):
        sysenv.get_mass_balance()
    assert sysenv.check_mass_balance(mass_change=-mfa.total_stock_change) is True


def test_process_without_flows():
    mfa = _computed_example_mfa()
    process = Process(name="idle", id=len(mfa.processes))
    mfa.processes["idle"] = process
    assert mfa.check_mass_balance() is True
    assert process.get_sum_inflows() == 0.0
    assert process.get_sum_outflows() == 0.0
    assert process.tolerance == 0.0
    assert process.check_mass_balance() is True


def _market_mfa(n_items: int, seed: int = 0) -> MFASystem:
    """Balanced MFA system whose process "market" sums many in- and outflow values.

    The inflow is given by origin, the outflow by destination, as total times share.
    """
    dims = DimensionSet(
        dim_list=[
            Dimension(letter="t", name="Time", dtype=int, items=[2000, 2001, 2002]),
            Dimension(letter="o", name="Origin", items=[f"o{i}" for i in range(n_items)]),
            Dimension(letter="d", name="Destination", items=[f"d{i}" for i in range(n_items)]),
        ]
    )
    processes = make_processes(["sysenv", "market"])
    flows = {
        "in": Flow(
            from_process=processes["sysenv"], to_process=processes["market"], dims=dims["t", "o"]
        ),
        "out": Flow(
            from_process=processes["market"], to_process=processes["sysenv"], dims=dims["t", "d"]
        ),
    }
    rng = np.random.default_rng(seed)
    shares = rng.lognormal(0, 0.25, size=(3, n_items))
    shares /= shares.sum(axis=1, keepdims=True)
    flows["in"][...] = StockArray(dims=dims["t", "o"], values=rng.lognormal(0, 0.25, (3, n_items)))
    flows["out"][...] = flows["in"].sum_over("o") * StockArray(dims=dims["t", "d"], values=shares)
    return MFASystem(dims=dims, parameters={}, processes=processes, flows=flows)


def test_process_tolerance_grows_with_summed_values():
    """A balanced process passes the check, however many values its mass balance sums."""
    n_items = 300
    for seed in range(10):
        assert _market_mfa(n_items, seed).check_mass_balance() is True

    mfa = _market_mfa(n_items)
    market = mfa.processes["market"]
    # each entry of the balance sums the inflows from all origins and the outflows to all
    # destinations
    summed_inflows = np.abs(mfa.flows["in"].values).sum(axis=1)
    summed_outflows = np.abs(mfa.flows["out"].values).sum(axis=1)
    largest_sum = (summed_inflows + summed_outflows).max()
    expected = config.relative_tolerance * np.finfo(np.float64).eps * largest_sum
    assert market.tolerance == pytest.approx(expected, rel=1e-9, abs=0)

    # an error of the size of a single small flow is still found
    mfa.flows["out"].values[0, 0] += 1e-9
    assert 1e-9 > market.tolerance
    assert market.check_mass_balance(error_behavior="ignore") is False
    assert mfa.check_mass_balance(error_behavior="ignore") is False


def test_process_tolerance_is_not_raised_by_many_small_values():
    """The tolerance follows the sum of the values, not their number."""
    mfa = _market_mfa(300)
    for flow in mfa.flows.values():
        flow.values[...] = 1e-6
        flow.values[:, 0] = 1.0
    expected = config.relative_tolerance * np.finfo(np.float64).eps * 2 * (1.0 + 299 * 1e-6)
    assert mfa.processes["market"].tolerance == pytest.approx(expected, rel=1e-9, abs=0)


def test_summed_float_precision():
    dims = _dims()
    eps = np.finfo(np.float64).eps
    by_element = StockArray(dims=dims, values=np.arange(20.0).reshape(dims.shape) - 5.0)
    total = StockArray(dims=dims["t",], values=np.full(10, 3.0))
    assert summed_float_precision([]) == 0.0

    def approx(expected: float):
        return pytest.approx(expected, rel=1e-9, abs=0)

    # a single array: its largest absolute value
    assert summed_float_precision([by_element]) == approx(eps * 14.0)
    # several arrays: the largest sum of absolute values in an entry of their common dimensions
    assert summed_float_precision([by_element, total]) == approx(eps * (13.0 + 14.0 + 3.0))
    by_element.values[0, 0] = np.nan
    assert summed_float_precision([by_element, total]) == approx(eps * (13.0 + 14.0 + 3.0))
    # each array counts with the machine epsilon of its own data type
    single = StockArray(dims=dims["t",], values=np.full(10, 3.0, dtype=np.float32))
    eps_single = np.finfo(np.float32).eps
    assert summed_float_precision([by_element, single]) == approx(eps * 27.0 + eps_single * 3.0)


# --- MFA system ---------------------------------------------------------------------------------


def test_mfa_system_mass_balance():
    mfa = _computed_example_mfa()
    assert mfa.check_mass_balance() is True
    total_stock_change = mfa.stocks["landfills"].net_inflow + mfa.stocks["slag piles"].net_inflow
    np.testing.assert_allclose(mfa.total_stock_change.values, total_stock_change.values)


def test_mfa_system_mass_balance_lists_all_failures():
    mfa = _computed_example_mfa()
    mfa.flows["shredder => remelting"].values[...] *= 1.1
    mfa.stocks["landfills"].stock.values[...] += 1.0
    with pytest.raises(ValueError) as error_info:
        mfa.check_mass_balance()
    message = str(error_info.value)
    assert "process 'shredder'" in message
    assert "process 'remelting'" in message
    assert "stock 'landfills'" in message
    assert "process 'landfills'" not in message
    assert "process 'sysenv'" not in message
    assert mfa.check_mass_balance(error_behavior="ignore") is False


def test_mfa_system_mass_balance_sysenv_failure():
    mfa = _computed_example_mfa()
    # stock not consistent with the flows of its process, nor with the system mass change
    mfa.stocks["landfills"].inflow.values[...] *= 2
    mfa.stocks["landfills"].compute()
    with pytest.raises(ValueError) as error_info:
        mfa.check_mass_balance()
    message = str(error_info.value)
    assert "process 'sysenv'" in message
    assert "process 'landfills'" in message
    assert "stock 'landfills'" not in message


def test_mfa_system_tolerance():
    mfa = _computed_example_mfa()
    max_value = max(
        [np.max(np.abs(f.values)) for f in mfa.flows.values()]
        + [np.max(np.abs(s.stock.values)) for s in mfa.stocks.values()]
    )
    expected = config.relative_tolerance * np.finfo(np.float64).eps * max_value
    assert mfa.tolerance == pytest.approx(expected)

    mfa.flows["shredder => remelting"].values[...] += 1e-3
    assert mfa.check_mass_balance(error_behavior="ignore") is False
    assert mfa.check_mass_balance(tolerance=1e-2) is True
    with config.override(absolute_tolerance=1e-2):
        assert mfa.tolerance == 1e-2
        assert mfa.check_mass_balance() is True


def test_check_flows(caplog):
    mfa = _computed_example_mfa()
    assert mfa.check_flows() is True

    mfa.flows["shredder => sysenv"].values[0, 0] = -1.0
    mfa.flows["demolition => landfills"].values[0, 0] = np.nan
    with caplog.at_level(logging.WARNING):
        assert mfa.check_flows(verbose=True) is False
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "Negative value in flow shredder => sysenv" in m and "1980, Fe" in m for m in messages
    )
    assert any("NaN values found in flow demolition => landfills" in m for m in messages)

    with pytest.raises(ValueError, match="NaN values"):
        mfa.check_flows(error_behavior="error")
    with config.override(error_behaviors={"check_flows": "error"}), pytest.raises(ValueError):
        mfa.check_flows()
    assert mfa.check_flows(exceptions=["shredder => sysenv", "landfills"]) is True


def test_mfa_system_deprecated_raise_error():
    mfa = _computed_example_mfa()
    mfa.flows["shredder => sysenv"].values[0, 0] = -1.0

    with pytest.warns(DeprecationWarning, match="raise_error"), pytest.raises(ValueError):
        mfa.check_flows(raise_error=True)
    with pytest.warns(DeprecationWarning, match="raise_error"):
        assert mfa.check_flows([], False) is False

    with (
        pytest.warns(DeprecationWarning, match="raise_error"),
        pytest.raises(ValueError, match="process 'shredder'"),
    ):
        mfa.check_mass_balance(None, True)
    with pytest.warns(DeprecationWarning, match="raise_error"):
        assert mfa.check_mass_balance(raise_error=False) is False
    with pytest.warns(DeprecationWarning, match="raise_error"):
        assert _computed_example_mfa().check_mass_balance(raise_error=True) is True
    with (
        pytest.warns(DeprecationWarning, match="raise_error"),
        pytest.raises(ValueError, match="Only one of"),
    ):
        mfa.check_mass_balance(error_behavior="warn", raise_error=True)


def test_mfa_system_without_stocks_tolerance():
    mfa = _computed_example_mfa()
    mfa.stocks = {}
    assert mfa.tolerance > 0.0
    assert mfa.total_stock_change == 0.0
    # the processes which had a stock now lack it in their mass balance
    assert mfa.check_mass_balance(error_behavior="ignore") is False
    assert all(process.stock is None for process in mfa.processes.values())


def test_mass_balance_nan_fails():
    mfa = _computed_example_mfa()
    tolerance_before = mfa.tolerance
    mfa.flows["shredder => remelting"].values[0, 0] = np.nan
    assert mfa.tolerance == pytest.approx(tolerance_before)
    with pytest.raises(ValueError, match="process 'shredder'"):
        mfa.check_mass_balance()
    assert mfa.processes["shredder"].check_mass_balance(error_behavior="ignore") is False

    stock = _simple_stock()
    stock.compute()
    stock.stock.values[0, 0] = np.nan
    assert stock.check_mass_balance(error_behavior="ignore") is False


def test_error_messages_name_config_options():
    mfa = _computed_example_mfa()
    mfa.flows["shredder => remelting"].values[...] *= 1.1
    with pytest.raises(ValueError, match=r"flodym\.config\.error_behaviors\.mass_balance"):
        mfa.check_mass_balance()
    with pytest.raises(ValueError, match=r"flodym\.config\.relative_tolerance"):
        mfa.processes["shredder"].check_mass_balance()
    with pytest.raises(ValueError, match=r"flodym\.config\.error_behaviors\.check_flows"):
        mfa.flows["shredder => sysenv"].values[0, 0] = -1.0
        mfa.check_flows(error_behavior="error")

    stock = _inconsistent_stock()
    with (
        config.override(checks={"mass_balance_after_compute": True}),
        pytest.raises(ValueError, match=r"flodym\.config\.checks\.mass_balance_after_compute"),
    ):
        stock.compute()


def test_mfa_system_uses_local_tolerances():
    """An error that is small compared to the largest values of the system, but large compared
    to the values of the process, is detected, consistent with the check of the process alone."""
    mfa = _computed_example_mfa()
    shredder = mfa.processes["shredder"]
    assert shredder.tolerance < mfa.tolerance / 2
    error = (shredder.tolerance + mfa.tolerance) / 2
    mfa.flows["shredder => sysenv"].values[0, 0] += error

    assert shredder.check_mass_balance(error_behavior="ignore") is False
    with pytest.raises(ValueError) as error_info:
        mfa.check_mass_balance()
    assert "process 'shredder'" in str(error_info.value)
    assert "process 'sysenv'" not in str(error_info.value)
    # with the system-wide tolerance, the error is not detected
    assert mfa.check_mass_balance(tolerance=mfa.tolerance) is True
