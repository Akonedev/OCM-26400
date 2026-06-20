"""Calibration épistémique — Brier / ECE / fiabilité — réfute audit H18.

EX-B186/202. Honnêteté épistémique mesurable : un modèle bien calibré dit '90% confiant'
et a raison dans 90% des cas. On mesure :
* Brier score : erreur quadratique moyenne des probabilités (0=parfait, 1=pire).
* Expected Calibration Error (ECE) : écart entre confiance et accuracy par bin.
* Reliability : la confiance reflète-t-elle la justesse réelle ?
C'est la 'conscience épistémique' mesurable (vs le threshold grossier de la KB).
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np


def brier_score(probs: List[float], outcomes: List[int]) -> float:
    """Brier score = mean((p - y)²). p ∈ [0,1] confiance, y ∈ {0,1} réalité. 0=parfait."""
    p = np.array(probs)
    y = np.array(outcomes, dtype=float)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(probs: List[float], outcomes: List[int],
                               n_bins: int = 10) -> float:
    """ECE : écart moyen |confiance − accuracy| par bin de confiance."""
    p = np.array(probs)
    y = np.array(outcomes, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(p)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if mask.sum() == 0:
            continue
        acc = y[mask].mean()
        conf = p[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def reliability_data(probs: List[float], outcomes: List[int], n_bins: int = 10
                     ) -> List[Tuple[float, float, int]]:
    """Courbe de fiabilité : (confiance_moyenne, accuracy, n) par bin."""
    p = np.array(probs)
    y = np.array(outcomes, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if mask.sum() == 0:
            continue
        out.append((float(p[mask].mean()), float(y[mask].mean()), int(mask.sum())))
    return out


def is_well_calibrated(probs: List[float], outcomes: List[int], threshold: float = 0.1) -> bool:
    """Un modèle est bien calibré si ECE < threshold (typiquement 0.1)."""
    return expected_calibration_error(probs, outcomes) < threshold


def confidence_summary(probs: List[float], outcomes: List[int]) -> dict:
    """Résumé calibration : Brier, ECE, fiabilité, verdict."""
    return {
        "brier": round(brier_score(probs, outcomes), 4),
        "ece": round(expected_calibration_error(probs, outcomes), 4),
        "mean_confidence": round(float(np.mean(probs)), 4),
        "accuracy": round(float(np.mean(outcomes)), 4),
        "well_calibrated": is_well_calibrated(probs, outcomes),
        "verdict": ("WELL_CALIBRATED" if is_well_calibrated(probs, outcomes)
                    else "MISCALIBRATED"),
    }


if __name__ == "__main__":
    # modèle bien calibré : confiance = réalité
    rng = np.random.RandomState(0)
    p_good = rng.uniform(0, 1, 200)
    y_good = (rng.uniform(0, 1, 200) < p_good).astype(int)
    print("[calibration] modèle BIEN calibré :", confidence_summary(p_good.tolist(), y_good.tolist()))
    # modèle sur-confiant : toujours 0.9 mais accuracy 0.5
    p_bad = [0.9] * 200
    y_bad = [1, 0] * 100
    print("[calibration] modèle SUR-confiant :", confidence_summary(p_bad, y_bad))
