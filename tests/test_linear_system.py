"""Tests for the assembly, classification and solution of sparse linear systems."""

import numpy as np
import pytest

import flodym._linear_system as linear_system
from flodym import Dimension, DimensionSet, FlodymArray
from flodym._linear_system import LinearSystem, classify, solve

DIMS = DimensionSet(
    dim_list=[
        Dimension(name="Time", letter="t", items=[2000, 2001, 2002]),
        Dimension(name="Region", letter="r", items=["north", "south"]),
        Dimension(name="Good", letter="g", items=["cars", "houses", "tools", "toys"]),
    ]
)
SCALAR = DimensionSet.empty()


def random_array(letters: tuple[str, ...], rng: np.random.Generator) -> FlodymArray:
    dims = DIMS.get_subset(letters)
    return FlodymArray(dims=dims, values=rng.uniform(0.5, 2.0, dims.shape))


def constant(value: float) -> FlodymArray:
    return FlodymArray(dims=SCALAR, values=np.array(value))


@pytest.mark.parametrize(
    "equation_letters, unknown_letters, coefficient_letters",
    [
        (("t",), ("t",), ()),
        (("t", "r"), ("t",), (("r",),)),
        (("t",), ("t", "r", "g"), (("g",),)),
        (("g", "t"), ("r", "t"), (("r", "g"), ("t",))),
        ((), ("t", "r"), ()),
        (("r", "g"), ("t",), (("g", "r"),)),
    ],
)
def test_term_matches_einsum(equation_letters, unknown_letters, coefficient_letters):
    """A term is the product of its arrays, summed to the dimensions of the equation."""
    rng = np.random.default_rng(0)
    unknown_values = random_array(unknown_letters, rng)
    coefficients = [random_array(letters, rng) for letters in coefficient_letters]
    known = random_array(unknown_letters, rng)

    system = LinearSystem()
    system.add_unknown("other", DIMS.get_subset(("g",)))
    unknown = system.add_unknown("x", unknown_values.dims)
    equation = system.add_equation("equation", DIMS.get_subset(equation_letters))
    system.add_term(equation, unknown, coefficients, sign=2.0)
    system.add_term(equation, known, coefficients, sign=-1.0)
    assembled = system.assemble()

    def expected(values: np.ndarray) -> np.ndarray:
        operands = [values] + [c.values for c in coefficients]
        subscripts = [unknown_letters] + [c.dims.letters for c in coefficients]
        inputs = ",".join("".join(letters) for letters in subscripts)
        result = np.einsum(f"{inputs}->{''.join(equation_letters)}", *operands)
        return np.broadcast_to(result, DIMS.get_subset(equation_letters).shape).ravel()

    unknowns = np.zeros(system.n_unknowns)
    unknowns[unknown.indices] = unknown_values.values.ravel()
    np.testing.assert_allclose(assembled.matrix @ unknowns, 2.0 * expected(unknown_values.values))
    np.testing.assert_allclose(assembled.rhs, expected(known.values))


def loop_system(share: float | np.ndarray, dims: DimensionSet = SCALAR) -> LinearSystem:
    """x = 3 + share * y and y = x, a loop with the input 3; z has no equation."""
    system = LinearSystem()
    x = system.add_unknown("x", dims)
    y = system.add_unknown("y", dims)
    system.add_unknown("z", SCALAR)
    first = system.add_equation("first", dims)
    system.add_term(first, x)
    system.add_term(first, y, [FlodymArray(dims=dims, values=np.asarray(share))], sign=-1.0)
    system.add_term(first, constant(3.0), sign=-1.0)
    second = system.add_equation("second", dims)
    system.add_term(second, x)
    system.add_term(second, y, sign=-1.0)
    return system


def test_classify_and_solve():
    assembled = loop_system(0.4).assemble()
    classification = classify(assembled)
    np.testing.assert_array_equal(classification.determined, [True, True, False])
    assert len(classification.redundant_equations) == 0
    solution = solve(assembled, pivot_tolerance=1e-12)
    np.testing.assert_allclose(solution.array_values("x"), 5.0)
    assert not solution.is_determined("z")
    assert np.all(solution.singular_part == -1)


def test_singular_items_are_found_and_others_solved():
    """The loop has no exit in the second year only."""
    time = DIMS.get_subset(("t",))
    solution = solve(loop_system(np.array([0.4, 1.0, 0.4]), time).assemble(), 1e-12)
    np.testing.assert_allclose(solution.array_values("x"), [5.0, np.nan, 5.0])
    singular = np.flatnonzero(solution.singular_part >= 0)
    blocks = solution.system.unknown_blocks(singular)
    assert [(b.key, b.item(i)) for b, i in zip(blocks, singular)] == [
        ("x", "Time 2001"),
        ("y", "Time 2001"),
    ]


def test_unknowns_depending_on_a_singular_part_are_not_determined():
    """w = x + 1 depends on the singular loop of x and y; v = 2 does not."""
    system = loop_system(1.0)
    w = system.add_unknown("w", SCALAR)
    v = system.add_unknown("v", SCALAR)
    equation = system.add_equation("w", SCALAR)
    system.add_term(equation, w)
    system.add_term(equation, system.unknowns["x"], sign=-1.0)
    system.add_term(equation, constant(1.0), sign=-1.0)
    equation = system.add_equation("v", SCALAR)
    system.add_term(equation, v)
    system.add_term(equation, constant(2.0), sign=-1.0)
    solution = solve(system.assemble(), pivot_tolerance=1e-12)
    assert not solution.is_determined("w") and not solution.is_determined("x")
    np.testing.assert_allclose(solution.array_values("v"), 2.0)
    # w depends on the singular part, but does not belong to it
    assert solution.singular_part[system.unknowns["w"].offset] == -1
    assert solution.singular_part[system.unknowns["x"].offset] >= 0


def test_entries_summing_to_zero_are_dropped():
    """Exact zeros, also from summed entries, do not count for the structure."""
    system = LinearSystem()
    x = system.add_unknown("x", SCALAR)
    equation = system.add_equation("equation", SCALAR)
    system.add_term(equation, x, sign=0.5)
    system.add_term(equation, x, sign=-0.5)
    assembled = system.assemble()
    assert assembled.matrix.nnz == 0
    assert not classify(assembled).determined[0]


def test_redundant_equations_and_residuals():
    system = LinearSystem()
    x = system.add_unknown("x", SCALAR)
    for value in (2.0, 2.0, 3.0):
        equation = system.add_equation(f"x = {value}", SCALAR)
        system.add_term(equation, x)
        system.add_term(equation, constant(value), sign=-1.0)
    solution = solve(system.assemble(), pivot_tolerance=1e-12)
    assert len(solution.redundant_equations) == 2
    residuals = {block.key: error for block, error, _ in solution.max_residuals()}
    value = solution.values[0]
    expected = {f"x = {v}": abs(value - v) for v in (2.0, 2.0, 3.0)}
    assert residuals == {key: expected[key] for key in residuals}


def test_underdetermined_part_is_not_solved():
    """x + y = 1 leaves both undetermined; z = 2 is determined independently."""
    system = LinearSystem()
    x = system.add_unknown("x", SCALAR)
    y = system.add_unknown("y", SCALAR)
    z = system.add_unknown("z", SCALAR)
    first = system.add_equation("x + y = 1", SCALAR)
    system.add_term(first, x)
    system.add_term(first, y)
    system.add_term(first, constant(1.0), sign=-1.0)
    second = system.add_equation("z = 2", SCALAR)
    system.add_term(second, z)
    system.add_term(second, constant(2.0), sign=-1.0)
    solution = solve(system.assemble(), pivot_tolerance=1e-12)
    np.testing.assert_array_equal(solution.determined, [False, False, True])
    np.testing.assert_allclose(solution.array_values("z"), 2.0)


def test_inconsistent_sizes_raise():
    system = LinearSystem()
    other_time = DimensionSet(dim_list=[Dimension(name="Time", letter="t", items=[1, 2])])
    x = system.add_unknown("x", DIMS.get_subset(("t",)))
    equation = system.add_equation("equation", other_time)
    with pytest.raises(ValueError, match="Dimension 't' has"):
        system.add_term(equation, x)


def rings_system(factors: list[np.ndarray]) -> LinearSystem:
    """Independent rings ``x_i - a_i * x_(i+1) = 1``, singular if the factors multiply to one."""
    system = LinearSystem()
    for ring, ring_factors in enumerate(factors):
        unknowns = [system.add_unknown((ring, i), SCALAR) for i in range(len(ring_factors))]
        for i, factor in enumerate(ring_factors):
            equation = system.add_equation((ring, i), SCALAR)
            system.add_term(equation, unknowns[i])
            system.add_term(equation, unknowns[(i + 1) % len(unknowns)], [constant(factor)], -1.0)
            system.add_term(equation, constant(1.0), sign=-1.0)
    return system


@pytest.mark.parametrize(
    "max_dense_size, max_dense_entries",
    [(64, 10_000_000), (2, 10_000_000), (64, 20)],
    ids=["dense", "LU for larger components", "several chunks"],
)
def test_singular_components_are_found(max_dense_size, max_dense_entries, monkeypatch):
    monkeypatch.setattr(linear_system, "MAX_DENSE_COMPONENT_SIZE", max_dense_size)
    monkeypatch.setattr(linear_system, "MAX_DENSE_ENTRIES", max_dense_entries)
    rng = np.random.default_rng(3)
    factors = [rng.uniform(0.3, 0.9, size) for size in (1, 2, 3, 3, 4, 3, 2, 3)]
    singular_rings = [2, 4, 5]
    for ring in singular_rings:
        factors[ring][-1] = 1.0 / np.prod(factors[ring][:-1])
    solution = solve(rings_system(factors).assemble(), pivot_tolerance=1e-12)
    for ring, ring_factors in enumerate(factors):
        keys = [(ring, i) for i in range(len(ring_factors))]
        determined = [solution.is_determined(key) for key in keys]
        assert determined == [ring not in singular_rings] * len(keys), ring
    parts = {solution.singular_part[solution.system.unknowns[(ring, 0)].offset] for ring in [2, 4]}
    assert len(parts) == 2 and -1 not in parts


def test_values_which_are_not_finite():
    """x = nan, y = x + 1 depends on it, z = 2 does not."""
    system = LinearSystem()
    x = system.add_unknown("x", SCALAR)
    y = system.add_unknown("y", SCALAR)
    z = system.add_unknown("z", SCALAR)
    for unknown, value in ((x, np.nan), (z, 2.0)):
        equation = system.add_equation(unknown.key, SCALAR)
        system.add_term(equation, unknown)
        system.add_term(equation, constant(value), sign=-1.0)
    equation = system.add_equation("y", SCALAR)
    system.add_term(equation, y)
    system.add_term(equation, x, sign=-1.0)
    system.add_term(equation, constant(1.0), sign=-1.0)
    solution = solve(system.assemble(), pivot_tolerance=1e-12)
    np.testing.assert_array_equal(solution.not_finite, [True, True, False])
    np.testing.assert_array_equal(solution.determined, [False, False, True])
    np.testing.assert_array_equal(solution.not_finite_equations, [0])
