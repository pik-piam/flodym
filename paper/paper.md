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
date: 14 July 2025
bibliography: paper.bib
---

# Summary

Dynamic material flow analysis (MFA) is one of the core methods in the field of industrial ecology. It systematically tracks and quantifies the time-dependent mass flows of materials through a system (such as a country's society) throughout different stages of their life cycle, as well as their accumulation in stocks, such as in the in-use stock representing all mass of a material contained in goods that are in use at a given point in time. Such analyses can be used for resource management, for assessing environmental impacts, and investigating circular economy measures. This assists in policy advice, in urban and regional planning, and in sustainable product design.

flodym (Flexible Open Dynamic Material Systems Model) is a library of objects and functions needed to build dynamic MFA.
It implements the `FlodymArray` class, which internally manages operations of one or several such multi-dimensional arrays. Objects representing flows, stocks, and parameters all inherit from this class. Stocks include lifetime models for dynamic stock modelling, i.e. for calculating the relation of material flows entering a stock and the mass and age structure of that stock over time. The whole MFA system is realized with an abstract parent class including sanity checks for the system, that users can implement a subclass of. flodym includes functionality for efficient read-in and export via pandas (@pandas1, @pandas2), as well as visualization routines.

flodym is based on on the concepts of the Open Dynamic Material Systems Model (ODYM) [@odym]. It can be seen as a re-implementation with added functionality.

# Statement of need

MFA is a widespread modelling tool in industrial ecology and related research fields, which makes general, accessible MFA tools vital for a large target group in academia and beyond. There are several existing open source MFA software packages.

ODYM [@odym] is, to the knowledge of the authors, the only general and flexible open-source MFA library, meaning that it is the only package allowing users to write their own MFA with the full flexibility of an own code. ODYM is therefore used widely in the industrial ecology community and beyond. One of ODYM's strengths is that it builds on an abstraction of the principles and structures of MFAs, such as:

- formalizing a system definition and establishing mass conservation checks
- formalizing dynamic stock models, such as done in [@Lauinger21]
- translating the abstract concepts of processes, stocks, flows, and parameters into a general library, without prescribing any details anout the MFA structure, such as its dimensions.

Since flodym is based on the concepts of ODYM, its structure and scope are similar to ODYM. However, flodym aims to add value in the following aspects, setting it apart from the original:

- flodym features complete internal dimension management in operations of multi-dimensional arrays. For example, the ODYM-based code
  ```
  waste = np.einsum('trp,pw->trw', end_of_life_products, waste_share)
  ```
  reduces to
  ```
  waste[...] = end_of_life_products * waste_share
  ```
  using `FlodymArray` objects. This allows to write simpler code and reduces errors. For example, dimensions of the same size could simply be switched in the `einsum` statement, which yields wrong results but goes unnoticed by the code. More importantly, it makes the code flexible (hence the name flodym) and extensible. Since the dimensions of each object are not explicitly given in the source code, but only once in the array definition, dimensions can be added, removed or re-ordered later without having to go through the whole source code.
- Slicing is eased in a similar way. If, for example, only the values of the `waste` array for the `C` (carbon) entry of the `element` dimension are needed, the ODYM syntax
  ```
  waste.Values[:,0,:,:]
  ```
  simplifies to
  ```
  waste['C']
  ```
  Again, this allows for adding or removing other dimensions later, or changing the position of the `C` entry in the `element` dimension, without having to change the code. Apart from these functionalities, `FlodymArrays` feature a large range of built-in methods for dimension manipulation, such as `sum_over`, `cast_to` or `get_shares_over`
- Data read-in and initialization as well as export are more flexible and general, through the use of pandas. Users can either use pre-built read-in functions, or write their own, and generate objects from data frames. On data read-in, flodym also performs checks on the data, detecting errors early on. Data read-in is performance-optimized especially for sparse arrays, since the full array size is only used after converting the input pandas data frame to a numpy array. Data is type-checked through the use of pydantic [@pydantic], adding robustness to the code.
- General visualization routines are implemented for pyplot [@pyplot] and plotly [@plotly] visualization.
- The treatment of material stocks is simplified and integrated with the rest of the MFA. This is realized through `Stock` objects containing `FlodymArray` objects for inflow, outflow and stock arrays, as well as a lifetime model and compute functions. Both stock and lifetime model are multi-dimensional and part of the mfa system class, such that the interaction with them is seamless.
- There are various smaller functional extensions. For example, stock models can handle non-evenly-spaced time step vectors, or sub-year lifetimes while still returning the correct stock-to-flow ratio.
- The whole code incorporates principles of software development (such as github actions for tests, documentation building and formatting) and clean code, easing future collaboration and extension. The code is extensively documented, including docstings, type hints, an API reference, howtos and examples.

Other existing open MFA packages such as OMAT [@OMAT] or STAN [@STAN] are different in scope: They are no libraries, but rather comprehensive tools, which eases their use, but limits the flexibility for using them in non-standard ways like flodym allows. The same applies to the pymfa [@pymfa2.1] and PMFA [@PMFA] packages, which are moreover focused on probabilistic MFA as an extension or special case of MFA.

# Applications

So far, flodym is used in-house for the REMIND-MFA [@remind-mfa] and the external TRANSIENCE EU MFA [@eu-mfa].
Further external applications are currently in early development stage.

# Acknowledgements

Thank you to Stefan Pauliuk and other contributors to ODYM [@odym], which forms the conceptual basis for flodym.

The development of flodym was conducted within the TRANSIENCE project, grant number 101137606, funded by the European Commission within the Horizon Europe Research and Innovation Programme.


# References
