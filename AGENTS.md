# Development Commands

- Install dependencies: `uv sync`.
- Run type checker `ty`: `uv check` (it does exist but may not be in your training set).
- Run tests: `uv run pytest`

Before completing a change, run:

```bash
uv check
uv run pytest
```

CI runs both of these on Python 3.10 to 3.12, on Linux, macOS and Windows.

Never run any formatter, linter, pre-commit - except if you work on the linter configuration itself.
In that case, run `uvx pre-commit run --all-files` to check the linter configuration.

# Repository structure

- `flodym`: the library itself
- `flodym/export`: plotting, Sankey diagrams, process graphs and data writers
- `examples`, `howtos`: jupytext notebooks, executed by the test suite
- `tests`: pytest suite
- `docs/source`: Sphinx documentation
- `scripts`: helper scripts

# Notebooks

`examples/*.py` and `howtos/*.py` are jupytext sources in `py:percent` format, paired with `.ipynb`
files. Both are tracked in git, but the `.py` file is the source of truth.

- Only ever edit the `.py` file. Never edit an `.ipynb` directly; it is generated.
- Do not sync the `.ipynb` files yourself. pre-commit.ci does that on the pull request.

# Conventions

- Target Python 3.10: `X | Y` annotations are fine, PEP 695 `type` aliases and generic syntax (`def f[T](...)`) are not.
- Use type hints. Currently, `pyproject.toml` disables a number of `ty` rules globally under `[tool.ty.rules]`. Code you add or touch should not need new entries in that ignore list.
- Use descriptive variable names, not single-letter
- Use google-style docstrings (https://google.github.io/styleguide/pyguide.html)
- Add pytests for critical functionality
- Modules whose name starts with `_`, such as `flodym/_df_to_flodym_array.py`, are private.
- Assign into an existing `FlodymArray` with the ellipsis slice, `foo[...] = bar`, not `foo = bar`.
  The slice keeps the declared dimensionality of `foo` and validates the right-hand side against it.
- Adding something to the public API takes three edits: the definition, an explicit re-export in
  `flodym/__init__.py` using the `X as X` form that the rest of that file uses, and an entry in the
  matching `docs/source/api.*.rst`.
- `flodym/example_objects.py` provides a ready-made `ExampleMFA` for tests
- use [Conventional Commits](https://www.conventionalcommits.org/) restricted to the types `feat`, `fix`, `docs`, and `chore`.
