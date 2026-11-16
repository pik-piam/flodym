"""Sparse linear equation systems over arrays with named dimensions.

The unknowns and the equations of a :py:class:`LinearSystem` come in blocks, each with one entry
per item of a :py:class:`flodym.DimensionSet`. Terms are added array-wise
(:py:meth:`LinearSystem.add_term`); the module knows nothing about processes or flows. It is used
by the simultaneous computation of processes (``flodym._process_equations``).

After :py:meth:`LinearSystem.assemble`, :py:func:`classify` finds which unknowns are determined
by the structure of the equations (coarse Dulmage–Mendelsohn decomposition), and :py:func:`solve`
computes them. Numerically singular parts are found by the fine decomposition into strongly
connected components; they and all unknowns depending on them are not determined.
"""

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import (
    breadth_first_order,
    connected_components,
    maximum_bipartite_matching,
)
from scipy.sparse.linalg import SuperLU, splu

from .dimensions import DimensionSet

MAX_DENSE_COMPONENT_SIZE = 64
"""Components of singular systems up to this size are checked for singularity as dense
matrices, together with others of the same size."""
MAX_DENSE_ENTRIES = 10_000_000
"""Number of entries of the dense matrices checked at once."""


class LabelledArray(Protocol):
    """An array with named dimensions, such as a :py:class:`flodym.FlodymArray`."""

    @property
    def dims(self) -> DimensionSet:
        """Dimensions of the array."""
        ...

    @property
    def values(self) -> np.ndarray:
        """Values of the array, with one axis per dimension, in the order of :py:attr:`dims`."""
        ...


@dataclass
class ArrayBlock:
    """Consecutive unknowns or equations of a :py:class:`LinearSystem`, one per item of dims."""

    key: Hashable
    """Identifier given by the caller."""
    dims: DimensionSet
    """Dimensions of the block. The items are ordered as the values of an array over them."""
    offset: int
    """Index of the first unknown or equation of the block in the system."""

    @property
    def size(self) -> int:
        """Number of unknowns or equations of the block."""
        return self.dims.total_size

    @property
    def indices(self) -> slice:
        """Indices of the unknowns or equations of the block in the system."""
        return slice(self.offset, self.offset + self.size)

    def item(self, index: int) -> str:
        """Description of the item with the given index in the system, e.g. ``"Time 2001"``."""
        positions = np.unravel_index(index - self.offset, self.dims.shape)
        return ", ".join(
            f"{dim.name} {dim.items[position]}" for dim, position in zip(self.dims, positions)
        )


@dataclass
class AssembledSystem:
    """The linear equation system ``matrix @ unknowns = rhs`` of a :py:class:`LinearSystem`."""

    matrix: sparse.csr_matrix
    """Coefficients, one row per equation and one column per unknown. Contains no explicit
    zeros."""
    rhs: np.ndarray
    """Right-hand side, one entry per equation."""
    unknowns: dict[Hashable, ArrayBlock]
    """Blocks of unknowns, by key, in the order of their offsets."""
    equations: list[ArrayBlock]
    """Blocks of equations, sorted by offset."""

    def unknown_blocks(self, indices: np.ndarray) -> list[ArrayBlock]:
        """Block of each of the given unknowns."""
        blocks = list(self.unknowns.values())
        return [blocks[position] for position in _block_positions(blocks, indices)]

    def equation_blocks(self, indices: np.ndarray) -> list[ArrayBlock]:
        """Block of each of the given equations."""
        return [self.equations[position] for position in _block_positions(self.equations, indices)]


@dataclass
class Classification:
    """Which unknowns are determined by the structure of the equations, see :py:func:`classify`."""

    determined: np.ndarray
    """For each unknown, whether it is determined."""
    equation_of_unknown: np.ndarray
    """For each unknown, the equation matched to it, or -1."""
    redundant_equations: np.ndarray
    """Equations which are not matched to an unknown. They contain only determined unknowns."""


@dataclass
class Solution:
    """Values of the unknowns of an :py:class:`AssembledSystem`, see :py:func:`solve`."""

    system: AssembledSystem
    values: np.ndarray
    """Value of each unknown, NaN if it is not determined."""
    determined: np.ndarray
    """For each unknown, whether it is determined."""
    singular_part: np.ndarray
    """For each unknown, the number of the singular part it belongs to, or -1. A singular part is
    a set of unknowns whose matched equations have no unique solution, although their structure
    suggests one. Unknowns depending on a singular part are not determined either, but do not
    belong to it."""
    not_finite: np.ndarray
    """For each unknown, whether it is not determined since its matched equation, or one of the
    equations it depends on, has coefficients or a right-hand side which are not finite."""
    not_finite_equations: np.ndarray
    """Equations with coefficients or a right-hand side which are not finite (NaN or infinite)."""
    redundant_equations: np.ndarray
    """Equations which were not needed for the solution and contain only determined unknowns.
    They are fulfilled if the given values are consistent."""

    def is_determined(self, key: Hashable) -> bool:
        """Whether the unknowns with the given key exist and are all determined."""
        block = self.system.unknowns.get(key)
        return block is not None and bool(np.all(self.determined[block.indices]))

    def array_values(self, key: Hashable) -> np.ndarray:
        """Values of a block of unknowns, shaped as an array over its dimensions."""
        block = self.system.unknowns[key]
        return self.values[block.indices].reshape(block.dims.shape)

    def max_residuals(self) -> list[tuple[ArrayBlock, float, int]]:
        """Largest absolute residual of the redundant equations, per block of equations.

        Returns:
            For each block with redundant equations: the block, the largest absolute residual
            (left-hand side minus right-hand side), and the equation where it occurs.
        """
        rows = self.redundant_equations
        if not len(rows):
            return []
        values = np.where(self.determined, self.values, 0.0)
        errors = np.abs(self.system.matrix[rows] @ values - self.system.rhs[rows])
        positions = _block_positions(self.system.equations, rows)
        # sorted by block, and within a block by decreasing error: the first of each block
        order = np.lexsort((-errors, positions))
        is_first = np.ones(len(order), dtype=bool)
        is_first[1:] = positions[order][1:] != positions[order][:-1]
        return [
            (self.system.equations[positions[index]], float(errors[index]), int(rows[index]))
            for index in order[is_first]
        ]

    def summary(self) -> "SolutionSummary":
        """Which unknowns are determined, without the matrix and the values."""
        singular_unknowns = np.flatnonzero(self.singular_part >= 0)
        return SolutionSummary(
            unknowns=self.system.unknowns,
            equations=self.system.equations,
            determined=self.determined,
            singular_unknowns=singular_unknowns,
            singular_parts=self.singular_part[singular_unknowns],
            not_finite_equations=self.not_finite_equations,
        )


@dataclass
class SolutionSummary:
    """Which unknowns of a :py:class:`Solution` are determined, and why not.

    Keeps the blocks and small arrays only, to explain afterwards which unknowns are not
    determined.
    """

    unknowns: dict[Hashable, ArrayBlock]
    """Blocks of unknowns, by key, in the order of their offsets."""
    equations: list[ArrayBlock]
    """Blocks of equations, sorted by offset."""
    determined: np.ndarray
    """For each unknown, whether it is determined."""
    singular_unknowns: np.ndarray
    """Unknowns which belong to a singular part, see :py:attr:`Solution.singular_part`."""
    singular_parts: np.ndarray
    """The singular part of each of :py:attr:`singular_unknowns`."""
    not_finite_equations: np.ndarray
    """Equations with values which are not finite, see :py:attr:`Solution.not_finite_equations`."""

    def is_determined(self, key: Hashable) -> bool:
        """Whether the unknowns with the given key exist and are all determined."""
        block = self.unknowns.get(key)
        return block is not None and bool(np.all(self.determined[block.indices]))

    def unknown_blocks(self, indices: np.ndarray) -> list[ArrayBlock]:
        """Block of each of the given unknowns."""
        blocks = list(self.unknowns.values())
        return [blocks[position] for position in _block_positions(blocks, indices)]

    def equation_blocks(self, indices: np.ndarray) -> list[ArrayBlock]:
        """Block of each of the given equations."""
        positions = _block_positions(self.equations, indices)
        return [self.equations[position] for position in positions]


class LinearSystem:
    """A linear equation system, built from blocks of unknowns and of equations.

    Each term of an equation block has the form ``sign * ⟨c_1 * ... * c_n * v⟩_Q``: the product
    of coefficient arrays ``c_i`` and a block of unknowns or a constant array ``v``, broadcast
    over the union of their dimensions, and summed over the dimensions not in the dimensions
    ``Q`` of the equation block. Terms with a constant go to the right-hand side.

    Example: with unknowns ``x`` over (t, r), a share ``s`` over (r) and an equation block over
    (t), ``add_term(equation, x, [s])`` adds ``sum_r s[r] * x[t, r]`` to the equation of each t.
    """

    def __init__(self) -> None:
        self.unknowns: dict[Hashable, ArrayBlock] = {}
        """Blocks of unknowns, by key."""
        self.equations: list[ArrayBlock] = []
        """Blocks of equations, in the order they were added."""
        self.n_unknowns = 0
        """Number of unknowns."""
        self.n_equations = 0
        """Number of equations."""
        self._rows: list[np.ndarray] = []
        self._cols: list[np.ndarray] = []
        self._values: list[np.ndarray] = []
        self._rhs_rows: list[np.ndarray] = []
        self._rhs_values: list[np.ndarray] = []

    def add_unknown(self, key: Hashable, dims: DimensionSet) -> ArrayBlock:
        """Add a block of unknowns, or return the one added before with the same key."""
        if key not in self.unknowns:
            self.unknowns[key] = ArrayBlock(key=key, dims=dims, offset=self.n_unknowns)
            self.n_unknowns += dims.total_size
        return self.unknowns[key]

    def add_equation(self, key: Hashable, dims: DimensionSet) -> ArrayBlock:
        """Add a block of equations, which are zero until terms are added."""
        block = ArrayBlock(key=key, dims=dims, offset=self.n_equations)
        self.equations.append(block)
        self.n_equations += block.size
        return block

    def add_term(
        self,
        equation: ArrayBlock,
        factor: ArrayBlock | LabelledArray,
        coefficients: Sequence[LabelledArray] = (),
        sign: float = 1.0,
    ) -> None:
        """Add a term to the left-hand side of a block of equations.

        For each item of the union of the dimensions of the equation, the factor and the
        coefficients, one entry is added: to the row of the item restricted to the dimensions of
        the equation, and to the column of the item restricted to the dimensions of the factor.
        Entries with the same row and column are summed when the system is assembled.

        Args:
            equation: Block of equations to add the term to.
            factor: Block of unknowns, or an array of constants, which go to the right-hand side.
            coefficients: Arrays the factor is multiplied with.
            sign: Factor of the term.
        """
        sizes = _sizes(
            [equation.dims, factor.dims] + [coefficient.dims for coefficient in coefficients]
        )
        joint = tuple(sizes)
        shape = tuple(sizes.values())
        rows = np.broadcast_to(_flat_index(equation, joint, sizes), shape).ravel()
        values = sign * np.ones((1,) * len(joint))
        for coefficient in coefficients:
            values = values * _broadcastable(coefficient, joint, sizes)
        if isinstance(factor, ArrayBlock):
            self._rows.append(rows)
            self._cols.append(np.broadcast_to(_flat_index(factor, joint, sizes), shape).ravel())
            self._values.append(np.broadcast_to(values, shape).ravel())
        else:
            constants = values * _broadcastable(factor, joint, sizes)
            self._rhs_rows.append(rows)
            self._rhs_values.append(-np.broadcast_to(constants, shape).ravel())

    def assemble(self) -> AssembledSystem:
        """Assemble the sparse matrix and the right-hand side.

        Entries with the same row and column are summed. Entries which are zero, also after
        summing, are dropped, so that the structure of the matrix reflects its values.
        """
        no_indices = np.zeros(0, dtype=np.int64)
        values = np.concatenate(self._values) if self._values else np.zeros(0)
        rows = np.concatenate(self._rows) if self._rows else no_indices
        cols = np.concatenate(self._cols) if self._cols else no_indices
        # the construction sums entries with the same row and column
        matrix = sparse.csr_matrix(
            (values, (rows, cols)), shape=(self.n_equations, self.n_unknowns)
        )
        matrix.eliminate_zeros()
        rhs = np.zeros(self.n_equations)
        for rhs_rows, rhs_values in zip(self._rhs_rows, self._rhs_values):
            rhs += np.bincount(rhs_rows, weights=rhs_values, minlength=self.n_equations)
        return AssembledSystem(
            matrix=matrix, rhs=rhs, unknowns=dict(self.unknowns), equations=list(self.equations)
        )


def classify(system: AssembledSystem) -> Classification:
    """Find the unknowns which are determined by the structure of the equations.

    A maximum matching between equations and unknowns assigns to each unknown at most one
    equation that computes it. Unknowns reachable from an unmatched unknown along alternating
    paths (an equation containing the unknown, then the unknown matched to this equation, and so
    on) are not determined. The others are determined, and their matched equations form a square
    system, which contains no other unknowns. Unmatched equations are redundant.
    """
    matrix = system.matrix
    n_unknowns = matrix.shape[1]
    unknown_of_equation = np.asarray(maximum_bipartite_matching(matrix, perm_type="column"))
    matched = np.flatnonzero(unknown_of_equation >= 0)
    equation_of_unknown = np.full(n_unknowns, -1, dtype=np.int64)
    equation_of_unknown[unknown_of_equation[matched]] = matched

    # Edges lead from each unknown to the unknowns matched to the equations it appears in. The
    # undetermined unknowns are those reachable from the unmatched ones.
    coo = matrix.tocoo()
    in_matched_equation = unknown_of_equation[coo.row] >= 0
    undetermined = _reachable(
        n_unknowns,
        coo.col[in_matched_equation],
        unknown_of_equation[coo.row[in_matched_equation]],
        np.flatnonzero(equation_of_unknown < 0),
    )
    return Classification(
        determined=~undetermined,
        equation_of_unknown=equation_of_unknown,
        redundant_equations=np.flatnonzero(unknown_of_equation < 0),
    )


def solve(system: AssembledSystem, pivot_tolerance: float) -> Solution:
    """Compute the unknowns which are determined by the equations.

    The unknowns which are determined by the structure (see :py:func:`classify`) are computed from
    their matched equations, with one sparse LU factorisation. Unknowns whose matched equation,
    or an equation they depend on, has values which are not finite are not determined.

    If the factorisation fails, the matched equations are decomposed into strongly connected
    components (the fine Dulmage–Mendelsohn decomposition, i.e. a block triangular form), each a
    set of unknowns which depend on each other. Components whose equations are singular, and all
    unknowns depending on them, are not determined; the others are solved.

    Args:
        system: The equation system.
        pivot_tolerance: Pivots of the LU factorisation, or singular values of a component, with
            at most this absolute value count as zero. The entries of the matrix should be of
            order one.

    Returns:
        The values of all unknowns, and which are determined.
    """
    classification = classify(system)
    n_unknowns = system.matrix.shape[1]
    values = np.full(n_unknowns, np.nan)
    singular_part = np.full(n_unknowns, -1, dtype=np.int64)
    not_finite = np.zeros(n_unknowns, dtype=bool)
    finite_equations = _finite_equations(system)

    unknowns = np.flatnonzero(classification.determined)
    equations = classification.equation_of_unknown[unknowns]
    square = system.matrix[equations][:, unknowns].tocsc()
    rhs = system.rhs[equations]
    blocked = _depending_on_not_finite(square, finite_equations[equations])
    if np.any(blocked):
        not_finite[unknowns[blocked]] = True
        unknowns, rhs = unknowns[~blocked], rhs[~blocked]
        square = square[~blocked][:, ~blocked].tocsc()
    square_values = _solve_square(square, rhs, pivot_tolerance)
    if square_values is None:
        square_values, square_parts = _solve_triangular_blocks(square, rhs, pivot_tolerance)
        singular_part[unknowns] = square_parts
    values[unknowns] = square_values
    determined = ~np.isnan(values)

    redundant = classification.redundant_equations
    redundant = redundant[finite_equations[redundant]]
    undetermined_count = abs(system.matrix[redundant]) @ (~determined).astype(np.float64)
    return Solution(
        system=system,
        values=values,
        determined=determined,
        singular_part=singular_part,
        not_finite=not_finite,
        not_finite_equations=np.flatnonzero(~finite_equations),
        redundant_equations=redundant[undetermined_count == 0],
    )


def _finite_equations(system: AssembledSystem) -> np.ndarray:
    """For each equation, whether its coefficients and its right-hand side are finite."""
    finite = np.isfinite(system.rhs)
    matrix = system.matrix
    not_finite_entries = ~np.isfinite(matrix.data)
    if np.any(not_finite_entries):
        rows = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
        finite[rows[not_finite_entries]] = False
    return finite


def _depending_on_not_finite(square: sparse.spmatrix, finite: np.ndarray) -> np.ndarray:
    """For each unknown of a square system, whether it depends on an equation which is not finite.

    Args:
        square: The matched equations of the unknowns, on the diagonal.
        finite: For each matched equation, whether its values are finite.
    """
    if np.all(finite):
        return np.zeros(len(finite), dtype=bool)
    depends_from, depends_to = _dependencies(square)
    return _reachable(len(finite), depends_from, depends_to, np.flatnonzero(~finite))


def _dependencies(square: sparse.spmatrix) -> tuple[np.ndarray, np.ndarray]:
    """Edges of the dependencies between the unknowns of a square system.

    The diagonal holds the matched equations. If the unknown ``j`` appears in the equation of
    the unknown ``i``, then ``i`` depends on ``j``: there is an edge from ``j`` to ``i``.
    """
    coo = square.tocoo()
    off_diagonal = coo.row != coo.col
    return coo.col[off_diagonal], coo.row[off_diagonal]


def _solve_triangular_blocks(
    square: sparse.csc_matrix, rhs: np.ndarray, pivot_tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a singular square system whose diagonal holds the matched equations.

    Returns:
        The values, NaN for unknowns which are not determined, and for each unknown the number of
        the singular component it belongs to, or -1.
    """
    size = square.shape[0]
    depends_from, depends_to = _dependencies(square)
    dependencies = sparse.csr_matrix(
        (np.ones(len(depends_from)), (depends_from, depends_to)), shape=(size, size)
    )
    _, components = connected_components(dependencies, directed=True, connection="strong")
    coo = square.tocoo()
    in_component = components[coo.row] == components[coo.col]
    singular = _singular_components(
        components,
        coo.row[in_component],
        coo.col[in_component],
        coo.data[in_component],
        pivot_tolerance,
    )
    is_singular = np.isin(components, singular)
    undetermined = _reachable(size, depends_from, depends_to, np.flatnonzero(is_singular))
    values = np.full(size, np.nan)
    solvable = np.flatnonzero(~undetermined)
    # the components of the solvable part are checked already, so no pivot tolerance is needed
    solvable_values = _solve_square(square[solvable][:, solvable].tocsc(), rhs[solvable], None)
    if solvable_values is None:
        # only if the floating point computation breaks down, e.g. by an overflow
        return values, np.zeros(size, dtype=np.int64)
    values[solvable] = solvable_values
    return values, np.where(is_singular, components, -1)


def _singular_components(
    components: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    pivot_tolerance: float,
) -> np.ndarray:
    """Components whose diagonal block is singular.

    Components with at most :py:data:`MAX_DENSE_COMPONENT_SIZE` unknowns are checked as dense
    matrices, together with the others of the same size, by their smallest singular value.
    Larger components are checked one by one, by the pivots of their LU factorisation.

    Args:
        components: Component of each unknown.
        rows: Row of each entry of the diagonal blocks.
        cols: Column of each entry of the diagonal blocks.
        values: Value of each entry of the diagonal blocks.
        pivot_tolerance: Blocks with a smallest singular value or pivot of at most this value
            are singular.
    """
    n_components = int(components.max()) + 1
    sizes = np.bincount(components, minlength=n_components)
    # position of each unknown within its component
    order = np.argsort(components, kind="stable")
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    position = np.empty(len(components), dtype=np.int64)
    position[order] = np.arange(len(components)) - starts[components[order]]
    entry_components = components[rows]
    singular: list[np.ndarray] = []
    for size in np.unique(sizes):
        of_size = np.flatnonzero(sizes == size)
        entries = np.flatnonzero(sizes[entry_components] == size)
        # entries sorted by component, so that those of each chunk of components are a slice
        entries = entries[np.argsort(entry_components[entries], kind="stable")]
        component_starts = np.searchsorted(entry_components[entries], of_size)
        component_starts = np.append(component_starts, len(entries))
        if size > MAX_DENSE_COMPONENT_SIZE:
            for index, component in enumerate(of_size):
                block_entries = entries[component_starts[index] : component_starts[index + 1]]
                block = sparse.csc_matrix(
                    (
                        values[block_entries],
                        (position[rows[block_entries]], position[cols[block_entries]]),
                    ),
                    shape=(size, size),
                )
                if _factorise(block, pivot_tolerance) is None:
                    singular.append(np.array([component]))
            continue
        chunk_size = max(1, MAX_DENSE_ENTRIES // (size * size))
        for first in range(0, len(of_size), chunk_size):
            chunk = of_size[first : first + chunk_size]
            chunk_entries = entries[component_starts[first] : component_starts[first + len(chunk)]]
            index_in_chunk = np.searchsorted(chunk, entry_components[chunk_entries])
            dense = np.zeros((len(chunk), size, size))
            entry_rows, entry_cols = position[rows[chunk_entries]], position[cols[chunk_entries]]
            dense[index_in_chunk, entry_rows, entry_cols] = values[chunk_entries]
            smallest = np.linalg.svd(dense, compute_uv=False)[:, -1]
            singular.append(chunk[~(smallest > pivot_tolerance)])
    return np.concatenate(singular) if singular else np.zeros(0, dtype=np.int64)


def _solve_square(
    square: sparse.csc_matrix, rhs: np.ndarray, pivot_tolerance: float | None
) -> np.ndarray | None:
    """Solution of a square system, None if it is singular or the solution is not finite.

    Without pivot tolerance, the system is only singular if the factorisation fails.
    """
    if square.shape[0] == 0:
        return np.zeros(0)
    factorisation = _factorise(square, pivot_tolerance)
    if factorisation is None:
        return None
    values = factorisation.solve(rhs)
    return values if np.all(np.isfinite(values)) else None


def _factorise(square: sparse.csc_matrix, pivot_tolerance: float | None) -> SuperLU | None:
    """LU factorisation of a square matrix, None if it is singular within the pivot tolerance."""
    try:
        factorisation = splu(square)
    except RuntimeError:
        # SuperLU raises this if the matrix is exactly singular
        return None
    pivots = np.abs(factorisation.U.diagonal())
    if pivot_tolerance is not None and np.any(pivots <= pivot_tolerance):
        return None
    return factorisation


def _reachable(
    n_nodes: int, edge_from: np.ndarray, edge_to: np.ndarray, sources: np.ndarray
) -> np.ndarray:
    """For each node of a directed graph, whether it can be reached from one of the sources."""
    start = n_nodes
    graph = sparse.csr_matrix(
        (
            np.ones(len(edge_from) + len(sources)),
            (
                np.concatenate([edge_from, np.full(len(sources), start)]),
                np.concatenate([edge_to, sources]),
            ),
        ),
        shape=(n_nodes + 1, n_nodes + 1),
    )
    order = np.asarray(breadth_first_order(graph, start, directed=True, return_predecessors=False))
    reachable = np.zeros(n_nodes + 1, dtype=bool)
    reachable[order] = True
    return reachable[:n_nodes]


def _block_positions(blocks: Sequence[ArrayBlock], indices: np.ndarray) -> np.ndarray:
    """Position of the block of each of the given indices, for blocks sorted by offset."""
    offsets = np.array([block.offset for block in blocks])
    return np.searchsorted(offsets, indices, side="right") - 1


def _sizes(dims_list: Sequence[DimensionSet]) -> dict[str, int]:
    """Size of each dimension of the given sets, by letter, in order of first appearance.

    Raises:
        ValueError: If a letter has different sizes in different sets.
    """
    sizes: dict[str, int] = {}
    for dims in dims_list:
        for dim in dims:
            if sizes.setdefault(dim.letter, dim.len) != dim.len:
                raise ValueError(
                    f"Dimension '{dim.letter}' has {dim.len} items in one array and "
                    f"{sizes[dim.letter]} in another."
                )
    return sizes


def _flat_index(block: ArrayBlock, joint: tuple[str, ...], sizes: dict[str, int]) -> np.ndarray:
    """Index in the system of each item of the block, broadcastable over the joint dimensions."""
    positions = []
    for letter in block.dims.letters:
        shape = [1] * len(joint)
        shape[joint.index(letter)] = sizes[letter]
        positions.append(np.arange(sizes[letter], dtype=np.int64).reshape(shape))
    if not positions:
        return np.full((1,) * len(joint), block.offset, dtype=np.int64)
    return block.offset + np.ravel_multi_index(tuple(positions), block.dims.shape)


def _broadcastable(
    array: LabelledArray, joint: tuple[str, ...], sizes: dict[str, int]
) -> np.ndarray:
    """Values of an array, with its axes ordered and extended to broadcast over joint dims."""
    letters = array.dims.letters
    axes = [letters.index(letter) for letter in joint if letter in letters]
    shape = [sizes[letter] if letter in letters else 1 for letter in joint]
    return np.transpose(np.asarray(array.values, dtype=np.float64), axes).reshape(shape)
