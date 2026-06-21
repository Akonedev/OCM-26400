"""Statistiques descriptives + inférentielles + Bayes — vague 3.

* Descriptives : moyenne, médiane, mode, variance, écart-type, quantiles, étendue.
* Inférentielles : corrélation (Pearson), covariance, régression linéaire.
* Bayésien : théorème de Bayes (P(H|D) = P(D|H)P(H)/P(D)), mise à jour de croyance.
Vérifiable : formules exactes. Pas de corpus (données synthétiques/torch pur).
"""
from __future__ import annotations
import math
from typing import Dict, List, Tuple


def mean(x: List[float]) -> float:
    return sum(x) / len(x) if x else 0.0


def median(x: List[float]) -> float:
    s = sorted(x)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def mode(x: List[float]) -> List[float]:
    if not x:
        return []
    from collections import Counter
    c = Counter(x)
    m = max(c.values())
    return sorted(k for k, v in c.items() if v == m)


def variance(x: List[float], ddof: int = 1) -> float:
    if len(x) <= ddof:
        return 0.0
    m = mean(x)
    return sum((v - m) ** 2 for v in x) / (len(x) - ddof)


def stdev(x: List[float], ddof: int = 1) -> float:
    return math.sqrt(variance(x, ddof))


def quantile(x: List[float], q: float) -> float:
    if not x:
        return 0.0
    s = sorted(x)
    idx = q * (len(s) - 1)
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return s[int(idx)]
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


def covariance(x: List[float], y: List[float]) -> float:
    n = min(len(x), len(y))
    if n == 0:
        return 0.0
    mx, my = mean(x[:n]), mean(y[:n])
    return sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (n - 1) if n > 1 else 0.0


def correlation(x: List[float], y: List[float]) -> float:
    """Corrélation de Pearson ∈ [-1, 1]."""
    sx, sy = stdev(x), stdev(y)
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return covariance(x, y) / (sx * sy)


def linear_regression(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Régression linéaire y = a·x + b. Retourne (pente, ordonnée)."""
    n = min(len(x), len(y))
    mx, my = mean(x[:n]), mean(y[:n])
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den = sum((x[i] - mx) ** 2 for i in range(n))
    a = num / den if den != 0 else 0.0
    b = my - a * mx
    return a, b


def bayes(prior: float, likelihood: float, marginal: float) -> float:
    """Théorème de Bayes : P(H|D) = P(D|H)·P(H) / P(D). Mise à jour de croyance."""
    if marginal <= 0:
        return 0.0
    return likelihood * prior / marginal


def bayes_update(prior: float, sensitivity: float, specificity: float,
                 prevalence: float = None) -> dict:
    """Test diagnostique : P(malade|positif) = VPP (valeur prédictive positive).
    sensitivity = P(+|malade), specificity = P(-|sain). prior = P(malade)."""
    p = prevalence if prevalence is not None else prior
    p_pos = sensitivity * p + (1 - specificity) * (1 - p)   # P(+)
    vpp = bayes(p, sensitivity, p_pos)                       # P(malade|+)
    return {"prior": p, "vpp_P_malade_pos": round(vpp, 4),
            "marginal_P_pos": round(p_pos, 4)}


if __name__ == "__main__":
    data = [2, 4, 4, 4, 5, 5, 7, 9]
    print(f"[stats] data={data}")
    print(f"  mean={mean(data)} median={median(data)} mode={mode(data)} "
          f"std={stdev(data):.3f} Q1={quantile(data,0.25)} Q3={quantile(data,0.75)}")
    x = [1, 2, 3, 4, 5]; y = [2, 4, 5, 4, 5]
    print(f"  corrélation(x,y)={correlation(x, y):.3f} régression={linear_regression(x, y)}")
    # Bayes : test 99% sensible, 95% spécifique, prévalence 1%
    print(f"  Bayes diagnostic : {bayes_update(0.01, 0.99, 0.95)}")
