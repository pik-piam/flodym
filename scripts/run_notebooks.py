from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_FOLDERS = ("examples", "howtos")


def iter_notebooks(repo_root: Path = REPO_ROOT) -> list[Path]:
    notebook_paths: list[Path] = []
    for folder_name in NOTEBOOK_FOLDERS:
        notebook_paths.extend((repo_root / folder_name).rglob("*.ipynb"))
    return sorted(notebook_paths)


def _build_execution_env(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root)
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{env['PYTHONPATH']}{os.pathsep}{existing_pythonpath}"
    return env


def run_notebook(
    notebook_path: Path,
    repo_root: Path = REPO_ROOT,
    inplace: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
    ]

    run_kwargs = dict(
        cwd=notebook_path.parent,
        capture_output=True,
        text=True,
        check=False,
        env=_build_execution_env(repo_root),
    )

    if inplace:
        command.extend(["--inplace", str(notebook_path)])
        return subprocess.run(command, **run_kwargs)

    with tempfile.TemporaryDirectory() as output_dir:
        command.extend(["--output-dir", output_dir, str(notebook_path)])
        return subprocess.run(command, **run_kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute all notebooks under examples/ and howtos/."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Execute notebooks without writing outputs back to notebook files.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first failing notebook.",
    )
    args = parser.parse_args(argv)

    inplace = not args.check
    notebook_paths = iter_notebooks(REPO_ROOT)

    if not notebook_paths:
        print("No notebooks found under examples/ and howtos/.")
        return 0

    mode = "in-place" if inplace else "check"
    print(f"Executing {len(notebook_paths)} notebooks in {mode} mode.")

    failures: list[tuple[Path, subprocess.CompletedProcess[str]]] = []
    for notebook_path in notebook_paths:
        relative_path = notebook_path.relative_to(REPO_ROOT)
        print(f"[RUN] {relative_path}")
        result = run_notebook(notebook_path, repo_root=REPO_ROOT, inplace=inplace)

        if result.returncode == 0:
            print(f"[OK]  {relative_path}")
            continue

        print(f"[FAIL] {relative_path} (return code: {result.returncode})", file=sys.stderr)
        if result.stdout:
            print("STDOUT:", file=sys.stderr)
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print("STDERR:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)

        failures.append((notebook_path, result))
        if args.stop_on_error:
            break

    if failures:
        print(f"Notebook execution failed for {len(failures)} notebook(s).", file=sys.stderr)
        return 1

    print("All notebooks executed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
