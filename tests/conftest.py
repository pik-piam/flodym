import sys

import matplotlib
import pytest
from process_helpers import compute_process_via_block

import flodym._process_system
import flodym.processes


def pytest_sessionstart(session: pytest.Session) -> None:
    """
    Use non-interactive matplotlib backend on Windows to avoid error
    `_tkinter.TclError: Can't find a usable init.tcl`

    https://github.com/matplotlib/matplotlib/issues/28957
    https://github.com/astral-sh/uv/issues/7036
    https://github.com/python/cpython/issues/125235
    https://github.com/actions/setup-python/issues/1102
    """
    if sys.platform.startswith("win"):
        matplotlib.use("Agg")


@pytest.fixture(params=["one at a time", "simultaneously"])
def engine(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Compute single processes one at a time, or by solving their equations as a block.

    Tests which use this fixture check that both give the same results.
    """
    if request.param == "simultaneously":
        monkeypatch.setattr(flodym._process_system, "compute_process", compute_process_via_block)
        monkeypatch.setattr(flodym.processes, "compute_process", compute_process_via_block)
    return request.param
