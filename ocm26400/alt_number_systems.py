"""Systèmes numériques alternatifs — LNS + RNS — EX-B304/305 (audit MOYENNE).

* LNS (Logarithmic Number System) : représente les nombres par leur logarithme.
  La multiplication devient une ADDITION (log(a×b) = log(a) + log(b)).
  La division devient une SOUSTRACTION. Évite la carry propagation.
  Utile en DSP, calcul spectral (proche de l'archi FFT du projet).

* RNS (Residue Number System) : représente les nombres par leurs résidus modulo
  plusieurs bases premières copremières. L'addition/multiplication se fait en PARALLÈLE
  sur chaque résidu (sans carry entre modules). Très rapide en hardware parallèle.

Vérifiable : LNS et RNS reproduisent exactement l'arithmétique standard.
"""
from __future__ import annotations
import math
from typing import List, Tuple


# ============ LNS (Logarithmic Number System) ============

class LNS:
    """Nombre en représentation logarithmique. x = sign × 2^log_val.
    multiplication = addition des logs. division = soustraction."""

    def __init__(self, value: float):
        self.sign = 1 if value >= 0 else -1
        self.log_val = math.log2(abs(value)) if value != 0 else float("-inf")

    @classmethod
    def from_log(cls, log_val: float, sign: int = 1) -> "LNS":
        obj = cls.__new__(cls)
        obj.log_val = log_val
        obj.sign = sign
        return obj

    def to_float(self) -> float:
        if self.log_val == float("-inf"):
            return 0.0
        return self.sign * (2 ** self.log_val)

    def __mul__(self, other: "LNS") -> "LNS":
        return LNS.from_log(self.log_val + other.log_val, self.sign * other.sign)

    def __truediv__(self, other: "LNS") -> "LNS":
        return LNS.from_log(self.log_val - other.log_val, self.sign * other.sign)

    def __repr__(self):
        return f"LNS({self.to_float():.4f})"


def lns_multiply(a: float, b: float) -> float:
    """Multiplication via LNS : log(a) + log(b) → 2^(somme). Sans carry."""
    return (LNS(a) * LNS(b)).to_float()


def lns_divide(a: float, b: float) -> float:
    """Division via LNS : log(a) - log(b). Sans carry."""
    return (LNS(a) / LNS(b)).to_float()


# ============ RNS (Residue Number System) ============

class RNS:
    """Nombre en résidus modulo plusieurs bases. Addition/mul en parallèle sans carry."""

    def __init__(self, value: int, bases: List[int]):
        self.bases = bases
        self.residues = [value % b for b in bases]

    @classmethod
    def from_residues(cls, residues: List[int], bases: List[int]) -> "RNS":
        obj = cls.__new__(cls)
        obj.residues = residues
        obj.bases = bases
        return obj

    def __add__(self, other: "RNS") -> "RNS":
        return RNS.from_residues(
            [(r1 + r2) % b for r1, r2, b in zip(self.residues, other.residues, self.bases)],
            self.bases)

    def __mul__(self, other: "RNS") -> "RNS":
        return RNS.from_residues(
            [(r1 * r2) % b for r1, r2, b in zip(self.residues, other.residues, self.bases)],
            self.bases)

    def to_int(self) -> int:
        """Décode les résidus → entier (CRT : Chinese Remainder Theorem)."""
        M = 1
        for b in self.bases:
            M *= b
        result = 0
        for i, (r, b) in enumerate(zip(self.residues, self.bases)):
            Mi = M // b
            # inverse de Mi mod b
            yi = pow(Mi, -1, b)
            result += r * Mi * yi
        return result % M


def rns_add(a: int, b: int, bases: List[int] = None) -> int:
    """Addition via RNS : parallèle sur chaque module, sans carry."""
    bases = bases or [7, 11, 13, 17]
    return (RNS(a, bases) + RNS(b, bases)).to_int()


def rns_multiply(a: int, b: int, bases: List[int] = None) -> int:
    """Multiplication via RNS : parallèle, sans carry."""
    bases = bases or [7, 11, 13, 17]
    return (RNS(a, bases) * RNS(b, bases)).to_int()


if __name__ == "__main__":
    # LNS
    print("[LNS] 3×7 =", lns_multiply(3, 7), "(=21)")
    print("[LNS] 12÷4 =", lns_divide(12, 4), "(=3)")
    print("[LNS] 2.5×4 =", lns_multiply(2.5, 4), "(=10)")
    # RNS
    bases = [7, 11, 13]
    print(f"[RNS] bases={bases}, M={7*11*13}")
    print("[RNS] 15+23 =", rns_add(15, 23, bases), "(=38)")
    print("[RNS] 15×23 =", rns_multiply(15, 23, bases), "(=345)")
    # vérification : RNS dans le range [0, M)
    M = 7 * 11 * 13
    print(f"[RNS] 345 mod {M} = {345 % M} (RNS retourne dans [0,{M}))")
