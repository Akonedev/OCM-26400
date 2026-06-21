"""Tests LNS + RNS (OCM-26400)."""
import pytest
from ocm26400.alt_number_systems import LNS, RNS, lns_multiply, lns_divide, rns_add, rns_multiply


def test_lns_multiply():
    assert abs(lns_multiply(3, 7) - 21) < 1e-6


def test_lns_divide():
    assert abs(lns_divide(12, 4) - 3) < 1e-6


def test_lns_fractional():
    assert abs(lns_multiply(2.5, 4) - 10) < 1e-6


def test_lns_class_ops():
    a, b = LNS(6), LNS(7)
    assert abs((a * b).to_float() - 42) < 1e-6
    assert abs((a / b).to_float() - 6/7) < 1e-6


def test_rns_add():
    bases = [7, 11, 13]
    assert rns_add(15, 23, bases) == 38


def test_rns_multiply():
    bases = [7, 11, 13]
    result = rns_multiply(15, 23, bases)
    M = 7 * 11 * 13
    assert result == (15 * 23) % M


def test_rns_no_carry():
    """RNS : chaque résidu calculé indépendamment (pas de carry entre modules)."""
    bases = [5, 7, 11]
    a = RNS(8, bases)
    b = RNS(9, bases)
    c = a + b
    assert c.residues == [(8+9)%5, (8+9)%7, (8+9)%11]
    assert c.to_int() == 17
