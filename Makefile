PYTHON ?= python

.PHONY: notebooks-run notebooks-sync notebooks-sync-and-run

notebooks-run:
	$(PYTHON) scripts/run_notebooks.py --ipynb

notebooks-sync:
	$(PYTHON) -m jupytext --to ipynb --update examples/*.py howtos/*.py

notebooks-sync-and-run: notebooks-sync notebooks-run
