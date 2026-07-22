import sys

import matplotlib
import pytest


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
