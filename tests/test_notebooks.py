from pathlib import Path

import pytest

from scripts.run_notebooks import REPO_ROOT, iter_notebooks, run_notebook

NOTEBOOK_PATHS = iter_notebooks(REPO_ROOT)


@pytest.mark.parametrize(
    "notebook_path",
    NOTEBOOK_PATHS,
    ids=lambda notebook_path: str(notebook_path.relative_to(REPO_ROOT)),
)
def test_notebook_runs_with_zero_return_code(notebook_path: Path):
    result = run_notebook(notebook_path=notebook_path, repo_root=REPO_ROOT, inplace=False)

    assert result.returncode == 0, (
        f"Notebook execution returned {result.returncode}: {notebook_path}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
