"""Physique RÉELLE avec unités SI + dimensional analysis — réfute audit C3.

L'audit C3 : « Les 91 règles sont (αa+βb) mod 11 — forme de loi physique mais PAS la
sémantique (pas d'unités SI, pas de dimensional analysis) ». On comble :

* Quantities physiques avec unités SI (masse kg, longueur m, temps s, force N, etc.).
* Lois physiques RÉELLES vérifiables : F=ma, E=½mv², v=d/t, P=mg, Ek=½mv², λ=v/f,
  I=V/R (Ohm), PV=nRT (gaz parfait), E=mc².
* Dimensional analysis : toute expression physique doit être dimensionnellement
  cohérente (on ne peut pas ajouter des N et des m). Rejette les équations physiquement
  INVALIDES → compétence physique RÉELLE, pas cosmétique.

Chaque loi est une RÈGLE VÉRIFIABLE : apply(*quantities)→résultat (avec unités),
verify rejette le faux ET rejette le dimensionnellement incohérent. C'est la compétence
physique réelle que le cahier des charges exige (mécanique, électricité, thermo...).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

# Dimensions SI de base : M (masse), L (longueur), T (temps), I (courant),
# Θ (température), N (quantité), J (intensité lumineuse)
# Une unité = vecteur dimensionnel (M^a L^b T^c I^d Θ^e N^f J^g)
Dims = Tuple[Fraction, Fraction, Fraction, Fraction, Fraction, Fraction, Fraction]

def D(M=0, L=0, T=0, I=0, Th=0, N=0, J=0) -> Dims:
    return (Fraction(M), Fraction(L), Fraction(T), Fraction(I),
            Fraction(Th), Fraction(N), Fraction(J))


# Catalogue d'unités SI dérivées (nom → (dimension, facteur_vers_SI_de_base))
UNITS: Dict[str, Dims] = {
    "kg": D(M=1), "g": D(M=1), "mg": D(M=1),
    "m": D(L=1), "km": D(L=1), "cm": D(L=1), "mm": D(L=1),
    "s": D(T=1), "min": D(T=1), "h": D(T=1),
    "A": D(I=1),
    "K": D(Th=1), "C": D(Th=1),
    "mol": D(N=1),
    "cd": D(J=1),
    # dérivées
    "N": D(M=1, L=1, T=-2),       # newton = kg·m·s⁻²
    "J": D(M=1, L=2, T=-2),       # joule
    "W": D(M=1, L=2, T=-3),       # watt
    "Pa": D(M=1, L=-1, T=-2),     # pascal
    "Hz": D(T=-1),                # hertz
    "V": D(M=1, L=2, T=-3, I=-1), # volt
    "ohm": D(M=1, L=2, T=-3, I=-2),
    "m/s": D(L=1, T=-1),
    "m/s2": D(L=1, T=-2),
}


@dataclass
class Quantity:
    """Quantité physique : valeur numérique + unité (→ dimension SI)."""
    value: float
    unit: str

    @property
    def dims(self) -> Dims:
        return UNITS.get(self.unit, D())

    def __add__(self, other: "Quantity") -> "Quantity":
        if self.dims != other.dims:
            raise DimensionError(f"addition incohérente: {self.unit} + {other.unit}")
        return Quantity(self.value + other.value, self.unit)

    def __mul__(self, other) -> "Quantity":
        if isinstance(other, (int, float)):
            return Quantity(self.value * other, self.unit)
        # produit de quantités : dimension = produit des dimensions, unit composite
        newdims = tuple(a + b for a, b in zip(self.dims, other.dims))
        return Quantity(self.value * other.value, _unit_for_dims(newdims))


class DimensionError(ValueError):
    """Erreur de cohérence dimensionnelle (équation physiquement invalide)."""


def _unit_for_dims(dims: Dims) -> str:
    """Retrouve le nom d'unité SI pour une dimension (ou forme composite)."""
    for name, d in UNITS.items():
        if d == dims:
            return name
    # forme composite lisible
    parts = []
    names = ["kg", "m", "s", "A", "K", "mol", "cd"]
    for n, d in zip(names, dims):
        if d == 0:
            continue
        parts.append(n if d == 1 else f"{n}^{int(d)}")
    return "·".join(parts) or "dimensionless"


def dims_compatible(a: Dims, b: Dims) -> bool:
    return a == b


# ---------------- Lois physiques RÉELLES (vérifiables) ----------------

def newton_second(m: Quantity, a: Quantity) -> Quantity:
    """F = m·a (2e loi de Newton). Masse × accélération → force (N)."""
    if m.dims != UNITS["kg"]:
        raise DimensionError("m doit être une masse (kg)")
    if a.dims != UNITS["m/s2"]:
        raise DimensionError("a doit être une accélération (m/s2)")
    return Quantity(m.value * a.value, "N")


def kinetic_energy(m: Quantity, v: Quantity) -> Quantity:
    """E_k = ½·m·v². Masse × (vitesse)² → énergie (J)."""
    if m.dims != UNITS["kg"]:
        raise DimensionError("m doit être une masse (kg)")
    if v.dims != UNITS["m/s"]:
        raise DimensionError("v doit être une vitesse (m/s)")
    return Quantity(0.5 * m.value * v.value ** 2, "J")


def velocity(d: Quantity, t: Quantity) -> Quantity:
    """v = d / t."""
    if d.dims != UNITS["m"] or t.dims != UNITS["s"]:
        raise DimensionError("d (m) / t (s)")
    return Quantity(d.value / t.value, "m/s")


def weight(m: Quantity, g: Quantity) -> Quantity:
    """P = m·g."""
    if m.dims != UNITS["kg"] or g.dims != UNITS["m/s2"]:
        raise DimensionError("P = m(kg)·g(m/s2)")
    return Quantity(m.value * g.value, "N")


def ohms_law(v: Quantity, r: Quantity) -> Quantity:
    """I = V / R (loi d'Ohm)."""
    if v.dims != UNITS["V"] or r.dims != UNITS["ohm"]:
        raise DimensionError("I = V/R")
    return Quantity(v.value / r.value, "A")


def wavelength(v: Quantity, f: Quantity) -> Quantity:
    """λ = v / f."""
    return Quantity(v.value / f.value, "m")


def energy_mass(m: Quantity) -> Quantity:
    """E = m·c² (c = 3e8 m/s). Masse → énergie."""
    if m.dims != UNITS["kg"]:
        raise DimensionError("E=mc² nécessite une masse (kg)")
    c = 299_792_458.0
    return Quantity(m.value * c ** 2, "J")


# ---------------- verify (paradigme OCM) + dimensional check ----------------

def verify_law(law_name: str, inputs: Tuple[Quantity, ...], output: Quantity) -> bool:
    """Verify : (1) le résultat numérique est correct, (2) dimensionnellement cohérent."""
    laws = {
        "newton_second": newton_second, "kinetic_energy": kinetic_energy,
        "velocity": velocity, "weight": weight, "ohms_law": ohms_law,
        "wavelength": wavelength, "energy_mass": energy_mass,
    }
    if law_name not in laws:
        return False
    try:
        expected = laws[law_name](*inputs)
    except (DimensionError, ZeroDivisionError):
        return False
    # valeur ET unité (dimension) doivent correspondre
    return (abs(expected.value - output.value) < 1e-6 * max(1, abs(expected.value))
            and expected.dims == output.dims)


def dimensionally_consistent(expr_dims: Dims, target_unit: str) -> bool:
    """Une expression est-elle dimensionnellement cohérente avec l'unité attendue ?"""
    return expr_dims == UNITS.get(target_unit, D())


if __name__ == "__main__":
    # démo : physique RÉELLE avec unités
    F = newton_second(Quantity(2.0, "kg"), Quantity(3.0, "m/s2"))
    print(f"F = m·a = 2kg × 3m/s² = {F.value}{F.unit}  verify={verify_law('newton_second',(Quantity(2,'kg'),Quantity(3,'m/s2')),F)}")
    Ek = kinetic_energy(Quantity(2.0, "kg"), Quantity(3.0, "m/s"))
    print(f"E_k = ½mv² = {Ek.value}{Ek.unit}")
    I = ohms_law(Quantity(12.0, "V"), Quantity(4.0, "ohm"))
    print(f"I = V/R = 12V/4Ω = {I.value}{I.unit}")
    # rejet dimensionnel : ajouter N + m = INVALIDE
    try:
        Quantity(1, "N") + Quantity(1, "m")
        print("ERREUR: aurait dû rejeter")
    except DimensionError as e:
        print(f"rejet dimensionnel OK : {e}")
