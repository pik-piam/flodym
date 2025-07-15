---
title: 'flodym: A Python package for dynamic material flow analysis'
tags:
  - Python
  - material flow analysis (MFA)
  - substance flow analysis (SFA)
  - dynamic stock modelling
  - industrial ecology
authors:
  - name: Jakob Dürrwächter
    corresponding: true
    orcid: 0000-0001-8961-5340
    affiliation: 1
  - name: Merlin Hosak
    affiliation: 1
  - name: Bennet Weiss
    orcid: 0009-0009-9859-9683
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

Dynamic material flow analysis (MFA) is one of the core methods in the field of industrial ecology. It systematically tracks the time-dependent mass flows of materials within a system (such as a country's society) throughout different stages of their life cycle, and accounts for their accumulation in stocks, such as the in-use stock of materials contained in products, assets or infrastructure at a given point in time. Such analyses are used for resource management, for assessing environmental impacts, and investigating circular economy measures - assisting in policy advice, urban and regional planning, or sustainable product design.

*flodym* (Flexible Open Dynamic Material Systems Model) is a library of objects and functions needed to build dynamic MFA.
It implements the `FlodymArray` class, which internally manages operations of one or several such multi-dimensional arrays. Objects representing flows, stocks, and parameters all inherit from this class. Stocks include lifetime models for dynamic stock modelling, i.e. for calculating the relation of material flows entering a stock and the mass and age structure of that stock over time. The whole MFA system is realized with an abstract parent class including sanity checks for the system, that users can implement a subclass of. *flodym* includes functionality for efficient read-in and export via pandas [@pandas1], [@pandas2], as well as visualization routines.

*flodym* is based on the concepts of the Open Dynamic Material Systems Model (ODYM) [@odym]. It can be seen as a re-implementation with vastly expanded functionality and the aim of improved structuring. As a result, *flodym* enables users to effortlessly deploy a customized, flexible MFA - built for maintainability and future expansion.

# Statement of need

MFA is a widespread method in industrial ecology and related fields, which makes general and accessible MFA tools vital for a large audience in academia and beyond.

While there are several existing open source MFA software packages, ODYM [@odym] is, to the knowledge of the authors, the only general and adaptable open-source MFA library, allowing users to write their own MFA with the full range of options that custom code offers. ODYM is therefore widely used in the industrial ecology community and beyond. One of ODYM's strengths is that it builds on an abstraction of the principles and structures of MFAs, such as:

- formalizing a system definition and establishing mass conservation checks
- formalizing dynamic stock models, as described later in @Lauinger21
- translating the abstract concepts of processes, stocks, flows, and parameters into a general library, without prescribing any details about the MFA structure, such as its dimensions.

*flodym* is based on the concepts of ODYM such that its structure, scope and strengths are similar to ODYM. However, there are also aspects in which *flodym* aims to fill gaps and add value, setting it apart from the original:

- ODYM stores dimensionality information in its array objects, but does not harvest the full potential of this information. *flodym* uses dimensionality information for complete internal dimension management in operations of multi-dimensional arrays. For example, the ODYM-based code
  ```
  waste = np.einsum('trp,pw->trw', end_of_life_products, waste_share)
  ```
  reduces to
  ```
  waste[...] = end_of_life_products * waste_share
  ```
  using `FlodymArray` objects. This allows to write simpler code and reduces errors. For example, dimensions of the same size could simply be switched in the `einsum` statement, which yields wrong results but goes unnoticed by the code. More importantly, it makes the code flexible (hence the name *flodym*) for adaptation and extension. Since the dimensions of each object are not explicitly given for every array operation, but only once in the array definition, dimensions can be added, removed or re-ordered later with minimal changes to the source code.
- Slicing is eased in a similar way. If, for example, only the values of the `waste` array for the `C` (carbon) entry of the `element` dimension are needed, the ODYM syntax
  ```
  waste.Values[:,0,:,:]
  ```
  simplifies to
  ```
  waste['C']
  ```
  Again, this allows for adding or removing other dimensions later, or changing the position of the `C` entry in the `element` dimension, without having to change the code. Apart from these functionalities, which are built on Python's magic methods, `FlodymArrays` feature a large range of built-in conventional methods for dimension manipulation, such as `sum_over`, `cast_to` or `get_shares_over`.
- Data read-in and initialization in ODYM prescribes a strict format based on Excel files. There is no data export functionality. In *flodym*, data read-in and export are based on pandas, opening them to a wide range of formats. Users can either use pre-built *flodym* read-in functions, or write their own, and generate objects from data frames. On data read-in, *flodym* performs checks on the data, detecting errors early on. Data read-in is performance-optimized especially for sparse arrays, since the full array size is only used after converting the input pandas data frame to a numpy array. Data is type-checked through the use of pydantic [@pydantic], adding robustness to the code.
- ODYM contains the possibility of data export to a non-Python Sankey plotting tool, but no other visualization tools. In *flodym*, general visualization routines are implemented for pyplot [@pyplot] and plotly [@plotly] visualization, including plotting of multi-dimensional arrays, and Sankey plots of the MFA system.
- In ODYM, the class for dynamic stock models does not allow for dimensions apart from time. It also does not contain integrated methods for all required computation steps. Moreover, the stock objects which are used in the MFA system do not contain inflow, outflow, and stock, but only one of the three, distinguished by a `Type` attribute. To transfer the results of the dynamic stock model into the MFA, one has to loop over all non-time dimensions, run several sub-methods of the scalar dynamic stock model, and transfer the results into the MFA arrays. This is somewhat cumbersome and a performance bottleneck. In *flodym*, the treatment of material stocks is simplified and integrated with the rest of the MFA. This is realized through `Stock` objects containing `FlodymArray` objects for inflow, outflow and stock arrays, as well as a lifetime model and compute functions. Both stock and lifetime model are multi-dimensional and part of the mfa system class, such that the interaction with them is seamless and the performance gains of numpy array operations are leveraged.
- *flodym* features various smaller functional extensions compared to ODYM. For example, stock models can handle non-evenly-spaced time step vectors, or sub-year lifetimes.
- ODYM features several great application examples, but only a partial API reference, and the API does not always follow PEP 8 naming conventions.
The whole *flodym* code incorporates principles of software development (such as PEP 8 formatting, or GitHub actions for tests and documentation building) and clean code, easing future collaboration and extension. The code is extensively documented, including docstrings, type hints, an API reference, how-tos and examples.

Other existing open MFA packages such as OMAT [@OMAT] or STAN [@STAN] are different in scope: They are no libraries, but rather comprehensive tools, which eases their use, but limits the flexibility for using them in non-standard ways like *flodym* allows. The same applies to the pymfa [@pymfa2.1] and PMFA [@PMFA] packages, which are moreover focused on probabilistic MFA as an extension or special case of MFA.

# Applications

So far, *flodym* is used in-house for the REMIND-MFA [@remind-mfa] and the external TRANSIENCE EU MFA [@eu-mfa].
Further external applications are currently in early development stage.

# Acknowledgements

Thank you to Stefan Pauliuk and other contributors to ODYM [@odym], which forms the conceptual basis for *flodym*.

We gratefully acknowledge funding from the TRANSIENCE project, grant number 101137606, funded by the European Commission within the Horizon Europe Research and Innovation Programme, from the Kopernikus-Projekt Ariadne through the German Federal Ministry of Education and Research (grant no. 03SFK5A0-2), and from the PRISMA project funded by the European Commission within the Horizon Europe Research and Innovation Programme under grant agreement No. 101081604 (PRISMA).

# References
