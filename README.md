# flodym

![PyPI - Version](https://img.shields.io/pypi/v/flodym)
[![flodym.tests](https://github.com/pik-piam/flodym/actions/workflows/main_actions.yml/badge.svg)](https://github.com/pik-piam/flodym/actions/workflows/main_actions.yml)
![docs](https://app.readthedocs.org/projects/flodym/badge/?version=latest)
[![status](https://joss.theoj.org/papers/92b6faa2d82b8694f4ad5d394053ef32/status.svg)](https://joss.theoj.org/papers/92b6faa2d82b8694f4ad5d394053ef32)

The flodym (Flexibe Open Dynamic Material Systems Model) library provides key functionality for building material flow analysis models, including
- the class `MFASystem` acting as a template (parent class) for users to create their own material flow models
- the class `FlodymArray` handling mathematical operations between multi-dimensional arrays
- different classes representing stocks accumulation, in- and outflows based on age cohort tracking and lifetime distributions. Those can be integrated in the `MFASystem`.
- different options for data input and export, as well as visualization

flodym is based on the concepts of the Open Dynamic Material Systems Model [ODYM](https://doi.org/10.1111/jiec.12952)
(Pauliuk & Heeren, 2020). It is a re-implementation with a fundamentally overhauled API, internal dimension management, dynamic stock model integration, and other added features. As a result, flodym enables users to write customized, flexible MFAs, designed for maintainability and future extension.


# Installation

flodym is published on PyPI and can be installed with your preferred Python package installer.

For example, with [pip](https://pypi.org/project/pip/) run `python -m pip install flodym`.

To install as a developer:

1. Install [uv](https://docs.astral.sh/uv/).
2. Clone the flodym repository using git.
3. From the project root, run `uv sync` to create a virtual environment and install all development dependencies.

This workflow uses a project-local virtual environment (`.venv`) managed by uv.


# Dimension Management

MFA models mainly consist of mathematical operations on different multi-dimensional arrays.

For example, the generation of different waste types `waste` might be a 3D-array defined over the dimensions time $t$, region $r$ and waste type $w$, and might be calculated from multiplying `end_of_life_products` (defined over time, region, and product type $p$) with a `waste_share` mapping from product type to waste type.
In numpy, the according matrix multiplication can be carried out nicely with the `einsum` function, were an index string indicates the involved dimensions:

```
waste = np.einsum('trp,pw->trw', end_of_life_products, waste_share)
```

flodym uses this function under the hood, but wraps it in a data type `FlodymArray`, which stores the dimensions of the array and internally manages the dimensions of different arrays involved in mathematical operations.

With this, the above example reduces to

```
waste[...] = end_of_life_products * waste_share
```

This gives a flodym-based MFA models the following properties:

- **Flexibility:** When changing the dimensionality of any array in your code, you only have to apply the change once, where the array is defined, instead of adapting every operation involving it. This also allows, for example, to add or remove an entire dimension from your model with minimal effort.
- **Simplicity:** Since dimensions are automatically managed by the library, coding array operations becomes much easier. No knowledge about the einsum function, about the dimensions of each involved array or their order are required.
- **Versatility:** We offer different levels of flodym use: Users can choose to use the standard methods implemented for data read-in, system setup and visualization, or only use only some of the data types like `FlodymArray`, and custom methods for the rest.
- **Robustness:** Through the use of [Pydantic](https://docs.pydantic.dev/latest/), the setup of the system is type-checked, highlighting errors early-on. The data read-in performs extensive checks on data sorting and completeness.
- **Performance:** The use of numpy ndarrays ensures low model runtimes compared with dimension matching through pandas dataframes.


# How to cite

If you use this software in publications, please cite our [article in the Journal of Open Source Software](https://doi.org/10.21105/joss.10105):

```
@article{
    Duerrwaechter2026,
    author = {Dürrwächter, Jakob and Hosak, Merlin and Weiss, Bennet and Ueckerdt, Falko},
    title = {flodym: A Python package for dynamic material flow analysis},
    journal = {Journal of Open Source Software},
    year = {2026},
    doi = {10.21105/joss.10105},
    url = {https://doi.org/10.21105/joss.10105},
    publisher = {The Open Journal},
    volume = {11},
    number = {119},
    pages = {10105}
}
```

flodym is based on the concepts of ODYM.
Please consider also citing the according publication: [Pauliuk & Heeren, 2020](https://doi.org/10.1111/jiec.12952).


# Contributing

Contribution instructions and development setup is documented in [CONTRIBUTING.md](CONTRIBUTING.md).

# How to report problems and get support

If you encounter a bug or unexpected behaviour, please [open an issue](https://github.com/pik-piam/flodym/issues/new) on GitHub.

For questions and general support, use [GitHub Discussions](https://github.com/pik-piam/flodym/discussions) or contact jakob\[dot\]duerrwaechter\[at\]pik-potsdam.de.


# Acknowledgements

We thank Stefan Pauliuk and the other contributors to ODYM, which forms the conceptual basis for flodym.

We gratefully acknowledge funding from the TRANSIENCE project, grant number 101137606, funded by the European Commission within the Horizon Europe Research and Innovation Programme, from the Kopernikus-Projekt Ariadne through the German Federal Ministry of Education and Research (grant no. 03SFK5A0-2), and from the PRISMA project funded by the European Commission within the Horizon Europe Research and Innovation Programme under grant agreement No. 101081604 (PRISMA).


 <!-- stop parsing here on readthedocs -->
# Documentation

See our [readthedocs](https://flodym.readthedocs.io/en/latest/) page for documentation!

The notebooks in the [examples](examples) folder provide usage examples of the code.
