"""Matrice jacobienne + changement de variables — réfute audit JAC.

EX-B289-301. Calcul vectoriel : Jacobienne (dérivées partielles d'une fonction
vectorielle), déterminant jacobien (changement de variables dans une intégrale),
divergence, gradient, rotationnel. Math EXACTE via SymPy.

* jacobian(funcs, vars) → matrice des dérivées partielles ∂fᵢ/∂xⱼ.
* jacobian_determinant → |J| (pour intégrales, coordonnées polaires/sphériques).
* gradient, divergence, curl, laplacien (analyse vectorielle).
Vérifiable : jacobienne exacte (formelle SymPy).
"""
from __future__ import annotations
from typing import List
import sympy as sp


def jacobian(func_exprs: List[str], variables: List[str]) -> list:
    """Matrice jacobienne : J[i][j] = ∂fᵢ/∂xⱼ. funcs et vars en strings."""
    vars_syms = sp.symbols(variables)
    if isinstance(vars_syms, sp.Symbol):
        vars_syms = [vars_syms]
    funcs = [sp.sympify(f, locals={v: s for v, s in zip(variables, vars_syms)})
             for f in func_exprs]
    J = sp.Matrix(funcs).jacobian(vars_syms)
    return J.tolist()


def jacobian_determinant(func_exprs: List[str], variables: List[str]):
    """Déterminant de la jacobienne (changement de variables)."""
    J = sp.Matrix(jacobian(func_exprs, variables))
    return sp.simplify(J.det())


def gradient(expr: str, variables: List[str]) -> list:
    """Gradient (∇f) : vecteur des dérivées partielles."""
    vars_syms = sp.symbols(variables)
    if isinstance(vars_syms, sp.Symbol):
        vars_syms = [vars_syms]
    f = sp.sympify(expr, locals={v: s for v, s in zip(variables, vars_syms)})
    return [str(sp.diff(f, v)) for v in vars_syms]


def divergence(vec_exprs: List[str], variables: List[str]) -> str:
    """Divergence (∇·F) : somme des dérivées partielles."""
    vars_syms = sp.symbols(variables)
    if isinstance(vars_syms, sp.Symbol):
        vars_syms = [vars_syms]
    vec = [sp.sympify(f, locals={v: s for v, s in zip(variables, vars_syms)})
           for f in vec_exprs]
    return str(sum(sp.diff(f, v) for f, v in zip(vec, vars_syms)))


def curl(vec_exprs: List[str], variables: List[str] = None) -> list:
    """Rotationnel (∇×F) en 3D. variables = [x,y,z]."""
    variables = variables or ["x", "y", "z"]
    x, y, z = sp.symbols(" ".join(variables))
    P, Q, R = [sp.sympify(f, locals={"x": x, "y": y, "z": z}) for f in vec_exprs]
    curl_x = sp.diff(R, y) - sp.diff(Q, z)
    curl_y = sp.diff(P, z) - sp.diff(R, x)
    curl_z = sp.diff(Q, x) - sp.diff(P, y)
    return [str(curl_x), str(curl_y), str(curl_z)]


def laplacian(expr: str, variables: List[str]) -> str:
    """Laplacien (∇²f) : somme des dérivées secondes."""
    vars_syms = sp.symbols(variables)
    if isinstance(vars_syms, sp.Symbol):
        vars_syms = [vars_syms]
    f = sp.sympify(expr, locals={v: s for v, s in zip(variables, vars_syms)})
    return str(sum(sp.diff(f, v, 2) for v in vars_syms))


def polar_jacobian() -> str:
    """Jacobienne du changement coordonnées polaires (r,θ)→(x,y) = r."""
    return str(jacobian_determinant(["r*cos(theta)", "r*sin(theta)"], ["r", "theta"]))


if __name__ == "__main__":
    print("[jacobian] f=(x²y, x+y²), vars (x,y) :")
    for row in jacobian(["x**2*y", "x+y**2"], ["x", "y"]):
        print(f"  {row}")
    print("[jacobian] dét polaire (r,θ)→(x,y) :", polar_jacobian(), "(doit être r)")
    print("[jacobian] gradient x²+y² :", gradient("x**2+y**2", ["x", "y"]))
    print("[jacobian] divergence (x², y², z²) :", divergence(["x**2", "y**2", "z**2"], ["x", "y", "z"]))
    print("[jacobian] laplacien x²+y² :", laplacian("x**2+y**2", ["x", "y"]))
