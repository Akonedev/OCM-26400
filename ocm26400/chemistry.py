"""Chimie RÉELLE — équilibrage d'équations + masses molaires — domaine compétent.

L'audit : les règles 'chemistry' sont cosmétiques (mod 11). On ajoute de la chimie
VÉRIFIABLE réelle :
* Équilibrage d'équations chimiques (solveur par coefficients stœchiométriques).
  ex : H2 + O2 → H2O équilibré = 2H2 + 1O2 → 2H2O.
* Masses molaires (table d'éléments) → masse d'un composé.
* Conservation : atomes égaux des 2 côtés (vérification).

Chaque opération est VÉRIFIABLE (l'équation équilibrée respecte la conservation).
C'est la compétence chimie RÉELLE (pas mod 11).
"""
from __future__ import annotations
from fractions import Fraction
from typing import Dict, List, Tuple

# Masses molaires (g/mol) — éléments courants
ATOMIC_MASS: Dict[str, float] = {
    "H": 1.008, "He": 4.003, "Li": 6.94, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085,
    "P": 30.974, "S": 32.06, "Cl": 35.45, "K": 39.098, "Ca": 40.078, "Fe": 55.845,
    "Cu": 63.546, "Zn": 65.38, "Ag": 107.868, "Au": 196.967,
}

import re


def parse_formula(formula: str) -> Dict[str, int]:
    """Parse une formule chimique (H2O, C6H12O6, Ca(OH)2) → {élément: compte}.
    Gère les parenthèses et indices."""
    # tokenise : élément (Maj+min) + nombre, ou () group
    counts: Dict[str, int] = {}

    def add(d1, d2, mult=1):
        for k, v in d2.items():
            d1[k] = d1.get(k, 0) + v * mult

    def parse(s, mult):
        i = 0
        local = {}
        while i < len(s):
            if s[i] == "(":
                # trouve la parenthèse fermante
                depth = 1
                j = i + 1
                while j < len(s) and depth:
                    if s[j] == "(":
                        depth += 1
                    elif s[j] == ")":
                        depth -= 1
                    j += 1
                inner = s[i + 1:j - 1]
                # nombre après )
                m = re.match(r"\d+", s[j:])
                num = int(m.group()) if m else 1
                add(local, parse(inner, num), mult)
                i = j + (len(m.group()) if m else 0)
            elif s[i].isupper():
                m = re.match(r"[A-Z][a-z]?", s[i:])
                elem = m.group()
                i += len(elem)
                m2 = re.match(r"\d+", s[i:])
                num = int(m2.group()) if m2 else 1
                local[elem] = local.get(elem, 0) + num * mult
                i += len(m2.group()) if m2 else 0
            else:
                i += 1
        return local

    return parse(formula, 1)


def molar_mass(formula: str) -> float:
    """Masse molaire d'un composé (g/mol)."""
    counts = parse_formula(formula)
    return round(sum(ATOMIC_MASS.get(e, 0) * n for e, n in counts.items()), 3)


def atom_count_sides(reactants: List[str], products: List[str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Compte les atomes de chaque côté (coefficients = 1)."""
    left: Dict[str, int] = {}
    right: Dict[str, int] = {}
    for f in reactants:
        for e, n in parse_formula(f).items():
            left[e] = left.get(e, 0) + n
    for f in products:
        for e, n in parse_formula(f).items():
            right[e] = right.get(e, 0) + n
    return left, right


def is_balanced(reactants: List[str], products: List[str], coeffs_l=None, coeffs_r=None
                ) -> bool:
    """Vérifie la conservation des atomes (équation équilibrée ?)."""
    cl = coeffs_l or [1] * len(reactants)
    cr = coeffs_r or [1] * len(products)
    left, right = {}, {}
    for f, c in zip(reactants, cl):
        for e, n in parse_formula(f).items():
            left[e] = left.get(e, 0) + n * c
    for f, c in zip(products, cr):
        for e, n in parse_formula(f).items():
            right[e] = right.get(e, 0) + n * c
    return left == right


def balance_simple(reactants: List[str], products: List[str]) -> Tuple[List[int], List[int]]:
    """Équilibre par recherche de coefficients entiers petits (1..6).
    Retourne (coeffs_gauche, coeffs_droite). Solveur naïf mais correct pour équations
    usuelles (combustion, synthèse)."""
    n_l, n_r = len(reactants), len(products)
    # éléments impliqués
    elems = set()
    for f in reactants + products:
        elems |= set(parse_formula(f))
    elems = sorted(elems)
    from itertools import product as iprod
    for coeffs in iprod(range(1, 7), repeat=n_l + n_r):
        cl, cr = coeffs[:n_l], coeffs[n_l:]
        if is_balanced(reactants, products, cl, cr):
            # normalise : PGCD = 1
            from math import gcd
            from functools import reduce
            g = reduce(gcd, coeffs)
            if g > 1:
                continue   # on veut la forme la plus simple (déjà found plus bas)
            return list(cl), list(cr)
    # fallback : forme non-équilibrée
    return [1] * n_l, [1] * n_r


if __name__ == "__main__":
    print("[chemistry] masse molaire H2O =", molar_mass("H2O"), "g/mol")
    print("[chemistry] masse molaire C6H12O6 =", molar_mass("C6H12O6"), "g/mol")
    print("[chemistry] masse molaire Ca(OH)2 =", molar_mass("Ca(OH)2"), "g/mol")
    # équilibrage
    reac, prod = ["H2", "O2"], ["H2O"]
    cl, cr = balance_simple(reac, prod)
    eq = " + ".join(f"{c if c>1 else ''}{f}" for c, f in zip(cl, reac)) + " → " + \
         " + ".join(f"{c if c>1 else ''}{f}" for c, f in zip(cr, prod))
    print(f"[chemistry] équilibrage {reac}→{prod} : {eq}  balanced={is_balanced(reac,prod,cl,cr)}")
    reac2, prod2 = ["CH4", "O2"], ["CO2", "H2O"]
    cl2, cr2 = balance_simple(reac2, prod2)
    eq2 = " + ".join(f"{c if c>1 else ''}{f}" for c, f in zip(cl2, reac2)) + " → " + \
          " + ".join(f"{c if c>1 else ''}{f}" for c, f in zip(cr2, prod2))
    print(f"[chemistry] combustion CH4 : {eq2}  balanced={is_balanced(reac2,prod2,cl2,cr2)}")
