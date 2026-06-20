"""Tests vague 2 — hypercomplex / calibration / continual_learning."""
import math
import pytest
from ocm26400.hypercomplex import Quaternion, hamilton_identity, rotate_vector, hurwitz_algebras
from ocm26400.calibration import brier_score, expected_calibration_error, confidence_summary
from ocm26400.continual_learning import demo_ewc, EWCCallback


# ---- hypercomplex ----
def test_hamilton_identity():
    assert hamilton_identity() is True


def test_quaternion_arithmetic():
    q = Quaternion(1, 2, 3, 4)
    assert (q + Quaternion(0, 0, 0, 0)) == q
    # i*j = k
    assert Quaternion.i() * Quaternion.j() == Quaternion.k()


def test_quaternion_norm_inverse():
    q = Quaternion(1, 2, 3, 4)
    prod = q * q.inverse()
    assert abs(prod.w - 1.0) < 1e-6
    assert abs(prod.x) < 1e-6 and abs(prod.y) < 1e-6 and abs(prod.z) < 1e-6


def test_rotate_vector_90_z():
    r = rotate_vector((1, 0, 0), (0, 0, 1), math.pi / 2)
    assert abs(r[0]) < 1e-6 and abs(r[1] - 1.0) < 1e-6 and abs(r[2]) < 1e-6


def test_hurwitz_four_algebras():
    algs = hurwitz_algebras()
    assert [a[0] for a in algs] == ["R", "C", "H", "O"]
    assert algs[2][3] is False   # H non-commutatif


# ---- calibration ----
def test_brier_perfect():
    assert brier_score([0.0, 1.0], [0, 1]) == 0.0


def test_brier_worst():
    assert brier_score([1.0, 1.0], [0, 0]) == 1.0


def test_ece_well_calibrated():
    """Confiance = réalité → ECE faible."""
    import numpy as np
    rng = np.random.RandomState(1)
    p = rng.uniform(0, 1, 300)
    y = (rng.uniform(0, 1, 300) < p).astype(int)
    assert expected_calibration_error(p.tolist(), y.tolist()) < 0.2


def test_ece_overconfident_high():
    """Toujours 0.9 mais acc 0.5 → ECE élevé."""
    assert expected_calibration_error([0.9] * 100, [1, 0] * 50) > 0.3


def test_summary_verdict():
    s_good = confidence_summary([0.5, 0.5], [1, 0])
    assert s_good["verdict"] == "WELL_CALIBRATED"


# ---- continual learning ----
def test_ewc_reduces_forgetting():
    res = demo_ewc(n_steps=200)
    # EWC réduit (ou égal) l'oubli par rapport à sans EWC
    assert res["mse_A_after_B_WITH_ewc"] <= res["mse_A_after_B_WITHOUT_ewc"] + 0.05


def test_ewc_callback_runs():
    import torch, torch.nn as nn
    m = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
    ewc = EWCCallback(m, lam=10.0)
    x = torch.randn(8, 2)
    y = torch.randn(8, 1)
    ewc.compute_fisher(x, y, n_samples=8)
    pen = ewc.penalty()
    assert pen.item() >= 0
