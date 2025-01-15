# flodym

The flodym (Flexibe Open Dynamic Material Systems Model) library provides key functionality for building material flow analysis models, including
- the class `MFASystem` acting as a template (parent class) for users to create their own material flow models
- the class `FlodymArray` handling mathematical operations between multi-dimensional arrays
- different classes representing stocks accumulation, in- and outflows based on age cohort tracking and lifetime distributions. Those can be integrated in the `MFASystem`.
- different options for data input and export, as well as visualization

# Thanks

flodym (flexible ODYM) is an adaptation of:

ODYM<br>
Copyright (c) 2018 Industrial Ecology<br>
author: Stefan Pauliuk, Uni Freiburg, Germany<br>
https://github.com/IndEcol/ODYM<br>

The development of flodym was conducted within the TRANSIENCE project, grant number 101137606, funded by the European Commission within the Horizon Europe Research and Innovation Programme.

# Installation

flodym dependencies are managed with [pip](https://pypi.org/project/pip/).

To install as a user: run `python -m pip install flodym`

To install as a developer:

1. Clone the flodym repository using git.
2. From the project main directory, run `pip install -e ".[tests,docs,examples]"` to obtain all the necessary
dependencies, including those for running the tests, making the documentation, and running the examples.

Note that it is advisable to do this within a virtual environment.

# Why choose flodym?

MFA models mainly consist on mathematical operations on different multi-dimensional arrays.

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

 <!-- stop parsing here on readthedocs -->
# How to contribute

If you'd like to contribute, the [issues page](https://github.com/pik-piam/flodym/issues) lists possible extensions and improvements.
If you wish to contribute your own, just create a fork and open a PR!

# Documentation

See our [readthedocs](https://flodym.readthedocs.io/en/latest/) page for documentation!

The notebooks in the [examples](examples) folder provide usage examples of the code.
