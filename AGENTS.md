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
In that case, run only the ruff hooks, `uvx pre-commit run ruff-check --all-files` and
`uvx pre-commit run ruff-format --all-files`, to check the linter configuration. Running all hooks
would also sync the notebooks (see below).

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
- Use google-style docstrings (https://google.github.io/styleguide/pyguide.html): a one-line
  summary, a blank line, then the details. Lines are at most 100 characters. Ruff checks both
  (`D` rules and `W505` in `pyproject.toml`).
- Do not repeat types in the `Args:` and `Returns:` sections of docstrings. Sphinx takes them from
  the type hints.
- `__init__` and dunder methods need no docstring. Document constructor arguments in the class
  docstring. Examples in docstrings use doctest syntax (`>>>`, and `...` for continuation lines).
- Naming: module-level constants in `ALL_CAPS`; class-level constants (`ClassVar`) in lowercase,
  e.g. `_prm_names`; `TypeVar`s and type aliases in `CapWords`.
- Types that appear in public signatures must be public as well: either export them, or write
  them inline, e.g. `Literal["forward", "backward"]`. Base classes and types that users should
  not use directly are listed under "Base classes and types" in the API docs, as `py:class` or
  `py:data` entries with a short description and without their full API.
- Order class members as follows: fields and private attributes, validators (keep their relative
  order, pydantic runs them in definition order), `__init__` and other dunder methods, alternative
  constructors (`from_*`), properties, public methods, private methods. Keep related methods
  together, e.g. a method and its alias.
- Add pytests for critical functionality
- Modules whose name starts with `_`, such as `flodym/_df_to_flodym_array.py`, are private.
  Names inside private modules need no leading underscore.
- In public modules, names that are only used inside flodym start with `_`. This includes pydantic
  validators. Such names may still be used across flodym modules: they are package-internal, not
  module-private. Examples are `Flow._set_prescribed`, `Flow._named_after_processes`,
  `MFASystem._absolute_float_precision` and `Process._inflows`. Users should not need them.
- Assign into an existing `FlodymArray` with the ellipsis slice, `foo[...] = bar`, not `foo = bar`.
  The slice keeps the declared dimensionality of `foo` and validates the right-hand side against it.
- Adding something to the public API takes three edits: the definition, an explicit re-export in
  `flodym/__init__.py` using the `X as X` form that the rest of that file uses, and an entry in the
  matching `docs/source/api.*.rst`.
- Renaming or removing public API requires a deprecation, unless it was never released. Keep the
  old name working and emit a `DeprecationWarning` with `stacklevel=2` and the message
  `"<old> is deprecated since v<X>. It will be removed in v3.0.0. Use <new> instead."`.
- Changes that can affect existing models, including deprecations, belong in
  `docs/source/migration_v2.rst`.
- `flodym/example_objects.py` provides a ready-made `ExampleMFA` for tests
- use [Conventional Commits](https://www.conventionalcommits.org/) restricted to the types `feat`, `fix`, `docs`, and `chore`.
