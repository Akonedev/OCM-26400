"""Tests solveur symbolique SymPy (OCM-26400) — audit M17."""
import pytest
from ocm26400 import equation_solver as es


SYMPY = es._HAS_SYMPY
skip_no_sympy = pytest.mark.skipif(not SYMPY, reason="SymPy non installé")


@skip_no_sympy
def test_solve_linear():
    assert es.solve_linear(3, 2, 11) == 3.0
    assert es.solve_linear(2, 0, 10) == 5.0
    assert es.solve_linear(0, 5, 3) is None         # a=0


@skip_no_sympy
def test_solve_quadratic():
    sols = es.solve_equation("x**2 - 5*x + 6")      # racines 2 et 3
    assert set(int(s) for s in sols) == {2, 3}


@skip_no_sympy
def test_derivative():
    assert es.derivative("x**3 + 2*x") == "3*x**2 + 2"
    assert es.derivative("x**2") == "2*x"


@skip_no_sympy
def test_integrate():
    assert es.integrate("x**2") == "x**3/3"


@skip_no_sympy
def test_simplify():
    assert es.simplify("(x**2-1)/(x-1)") == "x + 1"


@skip_no_sympy
def test_factor():
    assert es.factor("x**2 - 9") == "(x - 3)*(x + 3)"


@skip_no_sympy
def test_solve_system():
    sol = es.solve_system("2*x+3*y-7", "x-y-1")
    assert int(sol["x"]) == 2 and int(sol["y"]) == 1


@skip_no_sympy
def test_equation_solver_rules():
    rules = es.equation_solver_rules()
    assert len(rules) == 6
    domains = {r.domain for r in rules}
    assert "algebra" in domains and "calculus" in domains


@skip_no_sympy
def test_verify_solve():
    """verify rejette le faux (paradigme OCM)."""
    sols = es.solve_equation("x**2 - 4")
    assert set(int(s) for s in sols) == {-2, 2}
    assert es.solve_equation("x**2 - 4") != [1, 2]   # faux rejeté
