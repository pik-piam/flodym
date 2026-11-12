"""Creation of the dimensions of an MFA system from their definitions.

Shared by :py:meth:`flodym.DataReader.read_dimensions` and
:py:meth:`flodym.MFASystem.from_definition`. The items of a dimension are taken from its
definition, from a function which reads them, or, for dimensions created by splitting or combining
splits, from the other dimensions.
"""

from collections.abc import Callable, Mapping
from typing import Any

from .dimensions import Dimension, DimensionSet
from .mfa_definition import DimensionDefinition, _dimension_dependency_order

ReadDimension = Callable[[DimensionDefinition], Dimension]
"""Function which creates a dimension from its definition, e.g.
:py:meth:`flodym.DataReader.read_dimension`."""


def build_dimensions(
    definitions: list[DimensionDefinition], read_dimension: ReadDimension
) -> DimensionSet:
    """Create the dimensions given by their definitions.

    Dimensions with :py:attr:`flodym.DimensionDefinition.items` are created from these.
    Dimensions with :py:attr:`flodym.DimensionDefinition.splitting` are created from their parent
    with :py:meth:`flodym.Dimension.split`, and dimensions with
    :py:attr:`flodym.DimensionDefinition.combine_splits` with
    :py:meth:`flodym.Dimension.combine_splits`. All others are created by ``read_dimension``.
    Sub-dimensions (see :py:attr:`flodym.DimensionDefinition.parent`) are linked to their parent.

    Args:
        definitions: Definitions of the dimensions.
        read_dimension: Function creating a dimension which is neither given by items in its
            definition, nor created from other dimensions.

    Returns:
        The dimensions, in the order of the definitions.
    """
    dimensions: dict[str, Dimension] = {}
    for definition in _dimension_dependency_order(definitions):
        if definition.splitting is not None:
            dimension = split_dimension(definition, dimensions)
        elif definition.combine_splits is not None:
            dimension = combine_split_dimensions(definition, dimensions)
        else:
            if definition.items is not None:
                dimension = Dimension(
                    name=definition.name,
                    letter=definition.letter,
                    items=list(definition.items),
                    dtype=definition.dtype,
                    parent_items=definition.parent_items,
                )
            else:
                dimension = read_dimension(definition)
            if definition.parent is not None:
                dimension = link_parent(dimension, definition, dimensions[definition.parent])
        dimensions[definition.letter] = dimension
    return DimensionSet(dim_list=[dimensions[d.letter] for d in definitions])


def dimension_reader_from_items(
    definitions: list[DimensionDefinition],
    dimension_items: Mapping[str, Any] | ReadDimension | None,
) -> ReadDimension:
    """Turn the dimension items given to :py:meth:`flodym.MFASystem.from_definition` into a reader.

    A function is returned unchanged. A mapping is checked completely before any dimension is
    created, so that all errors are reported at once.

    Args:
        definitions: Definitions of the dimensions.
        dimension_items: Items of the dimensions which are read, keyed by dimension letter or
            name. For a dimension with a parent, a mapping from each item to its parent item;
            otherwise a list. Alternatively, a function creating a dimension from its definition,
            or None if all items are given in the definitions.

    Returns:
        Function creating a dimension from its definition.

    Raises:
        ValueError: If a key is neither a defined letter nor name, if a dimension is given twice,
            if items are given for a dimension which has items in its definition or is created
            from other dimensions, if the type of a value does not fit the definition, or if no
            items are given for a dimension which needs them.
    """
    if callable(dimension_items):
        return dimension_items
    items_by_letter = _items_by_letter(definitions, dimension_items or {})

    def read_dimension(definition: DimensionDefinition) -> Dimension:
        return _dimension_from_items(definition, items_by_letter[definition.letter])

    return read_dimension


def link_parent(
    dimension: Dimension, definition: DimensionDefinition, parent: Dimension
) -> Dimension:
    """Return a copy of a read dimension, linked to its parent.

    The parent items are converted to the datatype of the parent, if it has one.

    Raises:
        ValueError: If the read dimension has no parent items.
    """
    if dimension.parent_items is None:
        raise ValueError(
            f"Dimension '{definition.name}' has parent '{definition.parent}' in its definition, "
            "but no parent items were read. Give the parent item of each item, e.g. as a second "
            "column in the dimension file."
        )
    parent_items = dimension.parent_items
    if parent.dtype is not None:
        parent_items = [parent.dtype(item) for item in parent_items]
    return Dimension(
        name=dimension.name,
        letter=dimension.letter,
        items=dimension.items,
        dtype=dimension.dtype,
        parent=parent,
        parent_items=parent_items,
    )


def split_dimension(definition: DimensionDefinition, dimensions: dict[str, Dimension]) -> Dimension:
    """Create a dimension by splitting items of its parent, as given in the definition.

    Args:
        definition: Definition with parent and splitting.
        dimensions: Already created dimensions by letter, including those the definition
            depends on.
    """
    if definition.parent is None or definition.splitting is None:
        raise ValueError(f"Dimension definition '{definition.name}' has no parent or splitting.")
    parent = dimensions[definition.parent]
    splitting = {
        (parent.dtype(item) if parent.dtype is not None else item): dimensions[letter]
        for item, letter in definition.splitting.items()
    }
    split = parent.split(name=definition.name, letter=definition.letter, splitting=splitting)
    return Dimension.model_validate({**dict(split), "dtype": definition.dtype})


def combine_split_dimensions(
    definition: DimensionDefinition, dimensions: dict[str, Dimension]
) -> Dimension:
    """Create a dimension with two levels of splitting, as given in the definition.

    Args:
        definition: Definition with combine_splits.
        dimensions: Already created dimensions by letter, including those the definition
            depends on.
    """
    if definition.combine_splits is None:
        raise ValueError(f"Dimension definition '{definition.name}' has no combine_splits.")
    upper, *lower = (dimensions[letter] for letter in definition.combine_splits)
    combined = Dimension.combine_splits(
        name=definition.name, letter=definition.letter, upper=upper, lower=lower
    )
    return Dimension.model_validate({**dict(combined), "dtype": definition.dtype})


def _needs_items(definition: DimensionDefinition) -> bool:
    """Whether the items of a dimension are neither in its definition nor created."""
    return (
        definition.items is None
        and definition.splitting is None
        and definition.combine_splits is None
    )


def _items_by_letter(
    definitions: list[DimensionDefinition], dimension_items: Mapping[str, Any]
) -> dict[str, Any]:
    """Check the given dimension items against the definitions and key them by letter.

    Keys of length one are letters, longer keys are names.

    Raises:
        ValueError: Listing all problems, see :py:func:`dimension_reader_from_items`.
    """
    by_letter = {definition.letter: definition for definition in definitions}
    by_name = {definition.name: definition for definition in definitions}
    problems: list[str] = []
    items_by_letter: dict[str, Any] = {}
    for key, value in dimension_items.items():
        definition = (by_letter if len(key) == 1 else by_name).get(key)
        if definition is None:
            problems.append(f"'{key}' is neither the letter nor the name of a defined dimension.")
            continue
        if definition.letter in items_by_letter:
            problems.append(f"Dimension '{definition.name}' is given twice, by letter and name.")
            continue
        items_by_letter[definition.letter] = value
        if definition.items is not None:
            problems.append(f"Dimension '{definition.name}' has items in its definition already.")
        elif not _needs_items(definition):
            problems.append(
                f"Dimension '{definition.name}' is created by splitting or combine_splits, so no "
                "items can be given for it."
            )
        elif definition.parent is not None and not isinstance(value, Mapping):
            problems.append(
                f"Dimension '{definition.name}' has a parent, so its items must be given as a "
                "mapping from each item to its parent item."
            )
        elif definition.parent is None and (isinstance(value, (Mapping, str))):
            problems.append(
                f"Dimension '{definition.name}' has no parent, so its items must be given as a list."
            )
    missing = [d.name for d in definitions if _needs_items(d) and d.letter not in items_by_letter]
    if missing:
        problems.append(
            f"No items are given for the dimensions {missing}, neither in their definitions nor "
            "in dimension_items."
        )
    if problems:
        raise ValueError("Invalid dimension items:\n- " + "\n- ".join(problems))
    return items_by_letter


def _dimension_from_items(definition: DimensionDefinition, items: Any) -> Dimension:
    """Create a dimension from a list of items, or a mapping from items to parent items."""
    if isinstance(items, Mapping):
        return Dimension(
            name=definition.name,
            letter=definition.letter,
            items=list(items.keys()),
            dtype=definition.dtype,
            parent_items=list(items.values()),
        )
    return Dimension(
        name=definition.name, letter=definition.letter, items=list(items), dtype=definition.dtype
    )
