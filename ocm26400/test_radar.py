"""Tests radar/SAR (OCM-26400) — modalité radar."""
import numpy as np
from ocm26400.radar import (
    range_profile, range_doppler_matrix, matched_filter, cfar_detection,
    simulate_radar_returns, evaluate_radar,
)


def test_range_profile_finds_peak():
    # une cible crée un retour (sinusoïde) — la FFT révèle sa fréquence
    t = np.arange(64)
    echo = 10.0 * np.exp(2j * np.pi * 5 * t / 64)  # fréquence bin 5
    rp = range_profile(echo, n_range=64)
    assert rp.argmax() == 5  # pic au bin fréquentiel 5


def test_range_doppler_shape():
    echoes = np.random.randn(16, 64) + 1j * np.random.randn(16, 64)
    rd = range_doppler_matrix(echoes)
    assert rd.shape == (16, 64)


def test_matched_filter_amplifies():
    pulse = np.array([1, 0, 0, 0], dtype=float)
    echo = np.array([0, 0, 1, 0, 0], dtype=float)  # cible retardée
    mf = matched_filter(echo, pulse)
    assert mf.max() > 0


def test_cfar_detects_above_noise():
    data = np.ones(20) * 0.1  # bruit
    data[10] = 5.0  # cible forte
    det = cfar_detection(data, guard=1, train=3, pfa=1e-3)
    assert 10 in det


def test_simulate_returns_gt():
    echoes, gt = simulate_radar_returns(3, seed=0)
    assert echoes.shape[0] > 0 and len(gt) == 3


def test_evaluate_radar_runs():
    res = evaluate_radar(n_targets=2, snr_db=20, seed=0)
    assert res["detection_rate"] >= 0.3  # au moins 1/3 cibles à SNR 20
