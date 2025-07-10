---
title: 'flodym: A Python package for dynamic material flow analysis'
tags:
  - Python
  - material flow analysis
authors:
  - name: Jakob Dürrwächter
    corresponding: true
    orcid: 0000-0001-8961-5340
    affiliation: 1
  - name: Merlin Hosak
    affiliation: 1
  - name: Bennet Weiß
    affiliation: 1
  - name: Falko Ueckerdt
    affiliation: 1
    orcid: 0000-0001-5585-030X
affiliations:
 - name: Potsdam Institute for Climate Impact Research, Energy Transition Lab, Potsdam, Germany
   index: 1
date: 13 May 2025
bibliography: paper.bib
---

# Summary

Dynamic material flow analysis (MFA) systematically tracks and quantifies the time-dependent mass flows of materials through a system (such as a country's society) throughout different stages of their life cycle, as well as their accumulation in stocks. Algorithmically, this boils down to simple arithmetic operations between multidimensional arrays. The dimensions of these arrays are taken from a common superset of dimensions present for the whole system, but the subset of each array can be different.

flodym (Flexible Open Dynamic Material Systems Model) is a library of objects and functions needed to build dynamic MFA. It implements the `FlodymArray` class, which internally manages operations of one or several such multi-dimensional arrays. Flows, stocks, and parameters all inherit from this class. Stocks include lifetime models for dynamic stock modelling. For dimension management, each arrays are stored in `DimensionSet` object consisting of several `Dimension` objects. The whole MFA system is realized with an abstract parent class, that users can implement a subclass of. flodym includes functionality for efficient read-in and export via `pandas`, as well as visualization routines, and sanity checks for the system.

flodym is based on on the concepts of ODYM [@odym]. It can be seen as a re-implementation with added functionality.

# Statement of need

MFA is a widespread modelling tool in industrial ecology and related research fields, which makes general, accessible MFA tools vital for a large target group in academia and beyond. There are several existing MFA software packages. In the following, they are listed together with information on what flodym adds to this.

- ODYM [@odym] is by far the most prominent one. The structure is similar to flodym, however flodym adds the following aspects:
  - Internal dimension management in operations of multi-dimensional arrays. For example, the ODYM code
    ```
    waste = np.einsum('trp,pw->trw', end_of_life_products, waste_share)
    ```
    reduces to
    ```
    waste[...] = end_of_life_products * waste_share
    ```
    This allows to write simpler code and reduces errors, as for example, dimensions of the same size could simply be switched in the `einsum` statement, which yields wrong results but goes unnoticed by the code. More importantly, it makes the code flexible (hence the name flodym) and extensible. Since the dimensions of each object are not explicitly given in the source code, but only once in the array definition, dimensions can be added, removed or re-ordered later without having to go through the whole source code. Similarly, slicing is eased in a similar way. If, for example, only the values of the `waste` array for the `C` (carbon) entry of the `element` dimension are needed, the ODYM syntax `waste.Values[:,0,:,:]` simplifies to `waste['C']`. Again, this allows for adding or removing dimensions later, or changing the position of the `C` entry in the `element` dimension.
  - The treatment of material stocks is improved. Stocks are made objects with dedicated inflow, outflow and stock values, as well as a lifetime model. These objects are  multi-dimensional and integrated into the mfa system class, such that the interaction with them is seamless. This is opposed to one-dimensional (apart from time) and separate dynamic stock model objects in ODYM, which where cumbersome to attach to an MFA, and could be slow due to looping over one-dimensional objects.
  - Data read-in and initialization as well as export are more flexible and general, through the use of `pandas`. Users can either use pre-built read-in functions, or write their own, and generate objects from data frames. On data read-in, flodym also performs checks on the data, detecting errors early on. Data read-in is performance-optimized especially for sparse arrays, since the full array size is only used after converting the input pandas data frame to a numpy array.
  - General visualization routines are implemented for pyplot and plotly visualization.
  - There are various functional extensions. For example, stock models can handle non-evenly-spaced time step vectors or TODO.
  - The whole code follows PEP8 formatting and principles of software development (such as github actions for tests, documentation building and formatting) and clean code, easing future collaboration and extension.
  - The code is extensively documented, including docstings, type hints, an API reference, howtos and examples.
- TODO

# Acknowledgements

Thank you to Stefan Pauliuk and other authors of ODYM [@odym], which forms the conceptual basis for flodym.

The development of flodym was conducted within the TRANSIENCE project, grant number 101137606, funded by the European Commission within the Horizon Europe Research and Innovation Programme.


# References
