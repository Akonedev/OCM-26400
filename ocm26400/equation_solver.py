"""Solveur d'équations symbolique (SymPy) — audit M17.

Réfute audit M17 : « Équation solver symbolique (SymPy) » manquant. On intègre SymPy
pour résoudre équations, systèmes, dérivées, intégrales, simplifications — capacités
mathématiques RÉELLES (pas mod p) nécessaires pour AIME/HMMT/IMO.

Toutes les fonctions sont VÉRIFIABLES (apply→résultat, verify compare) — fidèles au
paradigme OCM. SymPy est le backend symbolique (déterministe, exact) ; OCM l'orchestre
via ses règles vérifiables.
"""
from __future__ import annotations
from typing import List, Optional, Tuple

try:
    import sympy as sp
    _HAS_SYMPY = True
except ImportError:
    _HAS_SYMPY = False

_X = sp.symbols("x") if _HAS_SYMPY else None


def _parse(expr_str: str):
    """Parse une expression string en expression SymPy (variable x)."""
    if not _HAS_SYMPY:
        return None
    return sp.sympify(expr_str, locals={"x": _X})


def solve_linear(a: float, b: float, c: float) -> Optional[float]:
    """Résout a*x + b = c → x. None si a=0."""
    if a == 0:
        return None
    return (c - b) / a


def solve_equation(eq_str: str, var: str = "x") -> Optional[List]:
    """Résout une équation string (ex 'x**2 - 5*x + 6') → liste de solutions.
    eq_str = le membre de gauche ( = 0 implicite)."""
    if not _HAS_SYMPY:
        return None
    v = sp.symbols(var)
    expr = sp.sympify(eq_str, locals={var: v})
    sols = sp.solve(expr, v)
    return [sp.nsimplify(s) for s in sols] if sols else []


def solve_system(eq1: str, eq2: str, v1: str = "x", v2: str = "y") -> Optional[dict]:
    """Résout un système 2×2 linéaire. eq ex '2*x + 3*y - 7' (=0)."""
    if not _HAS_SYMPY:
        return None
    s1, s2 = sp.symbols(f"{v1} {v2}")
    e1 = sp.sympify(eq1, locals={v1: s1, v2: s2})
    e2 = sp.sympify(eq2, locals={v1: s1, v2: s2})
    sol = sp.solve([e1, e2], (s1, s2))
    return {str(k): sp.nsimplify(val) for k, val in sol.items()} if sol else {}


def derivative(expr_str: str, var: str = "x") -> Optional[str]:
    """Dérivée symbolique."""
    if not _HAS_SYMPY:
        return None
    v = sp.symbols(var)
    return str(sp.diff(sp.sympify(expr_str, locals={var: v}), v))


def integrate(expr_str: str, var: str = "x") -> Optional[str]:
    """Intégrale indéfinie symbolique."""
    if not _HAS_SYMPY:
        return None
    v = sp.symbols(var)
    return str(sp.integrate(sp.sympify(expr_str, locals={var: v}), v))


def simplify(expr_str: str) -> Optional[str]:
    """Simplification algébrique."""
    if not _HAS_SYMPY:
        return None
    return str(sp.simplify(sp.sympify(expr_str, locals={"x": _X})))


def factor(expr_str: str) -> Optional[str]:
    """Factorisation."""
    if not _HAS_SYMPY:
        return None
    return str(sp.factor(sp.sympify(expr_str, locals={"x": _X})))


# ---------------- verify (paradigme OCM) ----------------

def verify_solve(args, output) -> bool:
    eq_str = args[0]
    sols = solve_equation(eq_str)
    return sols == output


def equation_solver_rules() -> list:
    """Règles solveur (intégration RuleLibrary)."""
    from .rules import Rule
    rules = []
    rules.append(Rule(name="solve_linear", domain="algebra",
                      desc="résout a*x+b=c", arity=3,
                      fn=lambda a, b, c: solve_linear(a, b, c)))
    rules.append(Rule(name="derivative", domain="calculus",
                      desc="dérivée symbolique (SymPy)", arity=1, fn=derivative))
    rules.append(Rule(name="integrate", domain="calculus",
                      desc="intégrale symbolique (SymPy)", arity=1, fn=integrate))
    rules.append(Rule(name="simplify", domain="algebra",
                      desc="simplification algébrique", arity=1, fn=simplify))
    rules.append(Rule(name="factor", domain="algebra",
                      desc="factorisation", arity=1, fn=factor))
    rules.append(Rule(name="solve_eq", domain="algebra",
                      desc="résout équation = 0", arity=1, fn=solve_equation))
    return rules


if __name__ == "__main__":
    if not _HAS_SYMPY:
        print("[equation_solver] SymPy non installé — pip install sympy")
    else:
        print("solve_linear 3x+2=11 →", solve_linear(3, 2, 11))
        print("solve_eq x²-5x+6 →", solve_equation("x**2 - 5*x + 6"))
        print("derivative x³+2x →", derivative("x**3 + 2*x"))
        print("integrate x² →", integrate("x**2"))
        print("simplify (x²-1)/(x-1) →", simplify("(x**2-1)/(x-1)"))
        print("factor x²-9 →", factor("x**2 - 9"))
        print("système 2x+3y=7, x-y=1 →", solve_system("2*x+3*y-7", "x-y-1"))
        print(f"\n{len(equation_solver_rules())} règles solveur (algebra/calculus via SymPy)")
