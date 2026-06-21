"""Théorie de l'information — Shannon — vague 3.

* Entropie de Shannon H(X) = −Σ p·log₂(p) (mesure d'incertitude, bits).
* Entropie conjointe H(X,Y), conditionnelle H(X|Y).
* Information mutuelle I(X;Y) = H(X) − H(X|Y).
* Divergence KL (Kullback-Leibler) entre 2 distributions.
Vérifiable : formules exactes, H ∈ [0, log₂(n)].
"""
from __future__ import annotations
import math
from typing import Dict, List


def entropy(probs: List[float]) -> float:
    """H(X) = −Σ p·log₂(p). probs = distribution de probabilité (somme = 1)."""
    return -sum(p * math.log2(p) for p in probs if p > 0)


def entropy_counts(counts: List[int]) -> float:
    """Entropie depuis des comptages (normalise en distribution)."""
    total = sum(counts)
    if total == 0:
        return 0.0
    return entropy([c / total for c in counts])


def max_entropy(n: int) -> float:
    """Entropie maximale pour n événements = log₂(n) (distribution uniforme)."""
    return math.log2(n) if n > 0 else 0.0


def joint_entropy(joint: List[List[float]]) -> float:
    """H(X,Y) depuis une matrice de probabilités jointes."""
    flat = [p for row in joint for p in row]
    return -sum(p * math.log2(p) for p in flat if p > 0)


def kl_divergence(p: List[float], q: List[float]) -> float:
    """D_KL(P‖Q) = Σ p·log₂(p/q). 0 si P=Q."""
    return sum(pi * math.log2(pi / qi) for pi, qi in zip(p, q) if pi > 0 and qi > 0)


def mutual_information(px: List[float], py: List[float], joint: List[List[float]]) -> float:
    """I(X;Y) = ΣΣ p(x,y)·log₂(p(x,y)/(p(x)p(y)))."""
    i = 0.0
    for r, pxy_row in enumerate(joint):
        for c, pxy in enumerate(pxy_row):
            if pxy > 0 and px[r] > 0 and py[c] > 0:
                i += pxy * math.log2(pxy / (px[r] * py[c]))
    return i


if __name__ == "__main__":
    print("[info] H(pièce équilibrée) =", entropy([0.5, 0.5]), "(=1 bit)")
    print("[info] H(pièce biaisée 0.9/0.1) =", round(entropy([0.9, 0.1]), 4))
    print("[info] H max pour 8 états =", max_entropy(8), "(=3 bits)")
    print("[info] KL(uniforme‖biaisée) =", round(kl_divergence([0.5, 0.5], [0.9, 0.1]), 4))
    print("[info] I(X;Y) X=Y parfaite =", round(mutual_information([0.5, 0.5], [0.5, 0.5],
          [[0.5, 0], [0, 0.5]]), 4), "(=1 bit, dépendance parfaite)")
