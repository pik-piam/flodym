# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "sodym"
copyright = "2024, the sodym authors"
author = "the sodym authors"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx-pydantic",
    "sphinxcontrib.autodoc_pydantic",
    "myst_parser",
    "nbsphinx",
    "nbsphinx_link",
]

# templates_path = ['_templates']
exclude_patterns = []

autodoc_pydantic_model_show_json = False
autodoc_pydantic_field_list_validators = False
autodoc_typehints = "description"

suppress_warnings = ["config.cache"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_static_path = ['_static']
html_theme_options = {
    "collapse_navigation": False,
}
