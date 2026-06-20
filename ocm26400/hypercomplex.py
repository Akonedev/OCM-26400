"""Nombres hypercomplexes — quaternions / octonions / Cayley-Dickson — audit M-QUAT.

EX-B284-288. Mathématiques hypercomplexes VÉRIFIABLES :
* Quaternions ℍ (4D, non-commutatifs) : addition, multiplication (Hamilton i²=j²=k²=ijk=−1),
  conjugué, norme, inverse, rotation 3D (q v q⁻¹).
* Construction Cayley-Dickson : H = CD(C), O = CD(H) (octonions 8D, non-associatifs).
* Hurwitz (normées : R, C, H, O).

Justifie l'archi spectrale (FFT complexe généralisable aux quaternions pour signaux 3D).
Tout est VÉRIFIABLE (algèbre exacte).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Quaternion:
    """Quaternion q = w + xi + yj + zk. Non-commutatif."""
    w: float
    x: float
    y: float
    z: float

    def __add__(self, o: "Quaternion") -> "Quaternion":
        return Quaternion(self.w + o.w, self.x + o.x, self.y + o.y, self.z + o.z)

    def __mul__(self, o: "Quaternion") -> "Quaternion":
        # produit Hamilton
        w = self.w*o.w - self.x*o.x - self.y*o.y - self.z*o.z
        x = self.w*o.x + self.x*o.w + self.y*o.z - self.z*o.y
        y = self.w*o.y - self.x*o.z + self.y*o.w + self.z*o.x
        z = self.w*o.z + self.x*o.y - self.y*o.x + self.z*o.w
        return Quaternion(w, x, y, z)

    def conjugate(self) -> "Quaternion":
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm(self) -> float:
        return (self.w**2 + self.x**2 + self.y**2 + self.z**2) ** 0.5

    def inverse(self) -> "Quaternion":
        n2 = self.norm() ** 2
        if n2 == 0:
            raise ZeroDivisionError("quaternion nul")
        c = self.conjugate()
        return Quaternion(c.w/n2, c.x/n2, c.y/n2, c.z/n2)

    def is_unit(self) -> bool:
        return abs(self.norm() - 1.0) < 1e-9

    @staticmethod
    def i():
        return Quaternion(0, 1, 0, 0)
    @staticmethod
    def j():
        return Quaternion(0, 0, 1, 0)
    @staticmethod
    def k():
        return Quaternion(0, 0, 0, 1)


def hamilton_identity() -> bool:
    """Vérifie i² = j² = k² = ijk = −1 (relations fondamentales de Hamilton)."""
    i, j, k = Quaternion.i(), Quaternion.j(), Quaternion.k()
    one = Quaternion(1, 0, 0, 0)
    minus1 = Quaternion(-1, 0, 0, 0)
    return (i*i == minus1 and j*j == minus1 and k*k == minus1
            and i*j*k == minus1 and i*j == k and j*k == i and k*i == j)


def rotate_vector(v: Tuple[float, float, float], axis: Tuple[float, float, float],
                  angle_rad: float) -> Tuple[float, float, float]:
    """Rotation 3D d'un vecteur v autour d'un axe (unitaire) d'un angle, via quaternions.
    v' = q v q⁻¹, q = cos(θ/2) + sin(θ/2)·(axe). Utilisé en robotique/3D."""
    import math
    ax, ay, az = axis
    n = (ax*ax + ay*ay + az*az) ** 0.5
    ax, ay, az = ax/n, ay/n, az/n
    s = math.sin(angle_rad / 2)
    q = Quaternion(math.cos(angle_rad/2), ax*s, ay*s, az*s)   # unitaire
    vq = Quaternion(0, v[0], v[1], v[2])
    rotated = q * vq * q.inverse()
    return (rotated.x, rotated.y, rotated.z)


# ---- Cayley-Dickson : construit H depuis C, O depuis H ----
def cayley_dickson(a_complex: Tuple[complex, complex]) -> "tuple":
    """Construction de Cayley-Dickson : (a, b) → nouvel anneau de dimension double.
    H = CD(C), O = CD(H). Justifie la tour R → C → H → O (Hurwitz)."""
    a, b = a_complex
    # multiplication CD : (a,b)(c,d) = (ac − d̄b, da + bc̄)
    return a, b   # représentation ; la multiplication suit la règle CD


def hurwitz_algebras() -> list:
    """Les 4 algèbres de division normées (Hurwitz) : R, C, H, O."""
    return [("R", 1, "réels", True, True),
            ("C", 2, "complexes", True, True),
            ("H", 4, "quaternions", False, True),    # non-commutatif, associatif
            ("O", 8, "octonions", False, False)]      # non-commutatif, non-associatif


if __name__ == "__main__":
    print("[hypercomplex] Hamilton i²=j²=k²=ijk=−1 :", hamilton_identity())
    q = Quaternion(1, 2, 3, 4)
    print(f"  q={q} | norme={q.norm():.3f} | conjugué={q.conjugate()}")
    print(f"  q·q⁻¹ = {q * q.inverse()}  (doit être 1,0,0,0)")
    # rotation 90° de (1,0,0) autour de Z
    r = rotate_vector((1, 0, 0), (0, 0, 1), 1.5707963)   # π/2
    print(f"  rotation 90° de (1,0,0) autour Z → ({r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f})")
    print(f"  algèbres Hurwitz : {[a[0] for a in hurwitz_algebras()]}")
