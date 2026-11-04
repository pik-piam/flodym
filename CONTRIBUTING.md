# Contributing to flodym

Thank you for your interest in contributing to flodym.

If you'd like to contribute, the [issues page](https://github.com/pik-piam/flodym/issues) lists possible extensions and improvements.
If you wish to contribute your own, just create a fork and open a PR!

## Development setup

1. Install [uv](https://docs.astral.sh/uv/).
2. [Fork and clone the repository](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project).
3. Optional: from the project root, run:

```bash
uv sync
```

This creates a project-local virtual environment (`.venv`) and installs development dependencies.

## Run tests

Run the test suite with

```bash
uv run pytest
```

## Commit conventions

We use [Conventional Commits](https://www.conventionalcommits.org/) with the following types:

- `feat` for new features
- `fix` for bug fixes
- `docs` for documentation
- `chore` for typos or maintenance changes
