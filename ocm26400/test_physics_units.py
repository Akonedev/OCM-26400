"""Tests physique RÉELLE avec unités SI + dimensional analysis (OCM-26400) — audit C3."""
import pytest
from ocm26400.physics_units import (
    Quantity, newton_second, kinetic_energy, velocity, weight, ohms_law,
    energy_mass, verify_law, DimensionError, dimensionally_consistent, UNITS, D,
)


def test_newton_second_real_units():
    F = newton_second(Quantity(2.0, "kg"), Quantity(3.0, "m/s2"))
    assert F.value == 6.0 and F.unit == "N"
    assert F.dims == UNITS["N"]


def test_kinetic_energy():
    Ek = kinetic_energy(Quantity(2.0, "kg"), Quantity(3.0, "m/s"))
    assert Ek.value == 9.0 and Ek.unit == "J"


def test_ohms_law():
    I = ohms_law(Quantity(12.0, "V"), Quantity(4.0, "ohm"))
    assert I.value == 3.0 and I.unit == "A"


def test_velocity():
    v = velocity(Quantity(100.0, "m"), Quantity(20.0, "s"))
    assert v.value == 5.0 and v.unit == "m/s"


def test_dimensional_addition_rejects_incompatible():
    """Ajouter N + m = INVALIDE (compétence physique réelle)."""
    with pytest.raises(DimensionError):
        Quantity(1, "N") + Quantity(1, "m")


def test_dimensional_addition_accepts_compatible():
    r = Quantity(2, "N") + Quantity(3, "N")
    assert r.value == 5 and r.unit == "N"


def test_verify_law_correct():
    F = newton_second(Quantity(2.0, "kg"), Quantity(3.0, "m/s2"))
    assert verify_law("newton_second", (Quantity(2, "kg"), Quantity(3, "m/s2")), F) is True


def test_verify_law_rejects_wrong_value():
    """verify rejette une valeur fausse (paradigme OCM)."""
    wrong = Quantity(99.0, "N")            # pas 6
    assert verify_law("newton_second", (Quantity(2, "kg"), Quantity(3, "m/s2")), wrong) is False


def test_verify_law_rejects_wrong_dimensions():
    """verify rejette un résultat avec mauvaise dimension."""
    bad_dims = Quantity(6.0, "J")          # bonne valeur mais Joules au lieu de Newton
    assert verify_law("newton_second", (Quantity(2, "kg"), Quantity(3, "m/s2")), bad_dims) is False


def test_law_rejects_wrong_input_dimensions():
    """newton_second rejette une masse en mètres (pas kg)."""
    with pytest.raises(DimensionError):
        newton_second(Quantity(2.0, "m"), Quantity(3.0, "m/s2"))


def test_energy_mass_einstein():
    E = energy_mass(Quantity(1.0, "kg"))
    assert E.value == pytest.approx(299_792_458.0 ** 2, rel=1e-6)
    assert E.unit == "J"


def test_dimensionally_consistent_check():
    assert dimensionally_consistent(UNITS["N"], "N") is True
    assert dimensionally_consistent(UNITS["J"], "N") is False   # J ≠ N


def test_quantity_multiplication():
    """m × a → dimension N (force)."""
    m = Quantity(2.0, "kg")
    a = Quantity(3.0, "m/s2")
    F = m * a
    assert F.value == 6.0
    assert F.dims == UNITS["N"]
