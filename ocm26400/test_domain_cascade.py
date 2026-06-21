"""Tests scratchpad cascade multi-domaine (OCM-26400)."""
from ocm26400.domain_cascade import (
    physics_cascade, chemistry_cascade, genetics_cascade, math_cascade,
    crypto_cascade, run_all_domain_cascades,
)


def test_physics_cascade():
    r = physics_cascade()
    assert r.primitive_ok and r.cascade_ok


def test_chemistry_cascade():
    r = chemistry_cascade()
    assert r.primitive_ok and r.cascade_ok


def test_genetics_cascade():
    r = genetics_cascade()
    assert r.primitive_ok and r.cascade_ok


def test_math_cascade():
    r = math_cascade()
    assert r.primitive_ok and r.cascade_ok


def test_crypto_cascade():
    r = crypto_cascade()
    assert r.primitive_ok and r.cascade_ok


def test_all_domains():
    rep = run_all_domain_cascades()
    assert rep["verdict"] == "ALL_DOMAINS_CASCADE_100"
    assert rep["n_primitives_grokked"] == rep["n_domains"]
    assert rep["n_cascades_ok"] == rep["n_domains"]
