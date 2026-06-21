"""Tests jacobienne + analyse vectorielle (OCM-26400) — audit JAC."""
import pytest
from ocm26400.jacobian import (jacobian, jacobian_determinant, gradient,
                                divergence, curl, laplacian, polar_jacobian)


def test_jacobian_matrix():
    J = jacobian(["x**2", "x*y"], ["x", "y"])
    # [[2x, 0],[y, x]]
    assert "2*x" in str(J[0][0])
    assert "x" in str(J[1][1])


def test_polar_jacobian_is_r():
    """Déterminant polaire (r,θ)→(x,y) = r (changement de variables classique)."""
    det = polar_jacobian()
    assert det.strip() == "r"


def test_gradient():
    g = gradient("x**2+y**2", ["x", "y"])
    assert "2*x" in g[0] and "2*y" in g[1]


def test_divergence():
    d = divergence(["x", "y", "z"], ["x", "y", "z"])   # div(x,y,z) = 3
    assert d.strip() == "3"


def test_curl_conservative_zero():
    """Rotationnel d'un champ conservateur (gradient) = 0."""
    c = curl(["x", "0", "0"], ["x", "y", "z"])
    assert all(part.strip() == "0" for part in c)


def test_laplacian():
    lap = laplacian("x**2+y**2", ["x", "y"])    # ∇²(x²+y²) = 4
    assert lap.strip() == "4"
