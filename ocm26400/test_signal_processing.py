"""Tests traitement du signal (OCM-26400) — audit SIG."""
import math
import pytest
from ocm26400.signal_processing import (
    PIDController, simulate_pid_to_setpoint, spectrum, dominant_frequency,
    lowpass_moving_average, highpass, remove_dc, rms, zero_crossing_rate,
)


def test_pid_converges_to_setpoint():
    r = simulate_pid_to_setpoint(1.0, n_steps=200)
    assert r["converged"] is True
    assert abs(r["final_state"] - 1.0) < 0.1


def test_pid_reaches_different_setpoints():
    for sp in [0.5, 2.0, -1.0]:
        r = simulate_pid_to_setpoint(sp, n_steps=300)
        assert abs(r["final_state"] - sp) < 0.15


def test_spectrum_returns_amps_freqs():
    sig = [math.sin(2 * math.pi * 5 * i / 100) for i in range(200)]
    amp, freqs = spectrum(sig)
    assert len(amp) == len(freqs) and len(amp) > 0


def test_dominant_frequency_detected():
    """Un signal 10Hz → fréquence dominante ≈ 10Hz."""
    t = [i / 100 for i in range(400)]
    sig = [math.sin(2 * math.pi * 10 * ti) for ti in t]
    fdom = dominant_frequency(sig, sample_rate=100)
    assert abs(fdom - 10.0) < 2.0


def test_lowpass_smooths():
    noisy = [1, 5, 1, 5, 1, 5]
    smoothed = lowpass_moving_average(noisy, window=3)
    assert max(smoothed) - min(smoothed) < max(noisy) - min(noisy)


def test_highpass_removes_low():
    sig = [10 + math.sin(i) for i in range(20)]   # DC + oscillation
    hp = highpass(sig)
    assert abs(sum(hp) / len(hp)) < abs(sum(sig) / len(sig))   # DC retiré


def test_remove_dc_zero_mean():
    sig = [5, 6, 4, 5, 6]
    assert abs(sum(remove_dc(sig))) < 1e-6


def test_rms_positive():
    assert rms([3, 4]) == pytest.approx(math.sqrt(12.5))


def test_zero_crossing():
    assert zero_crossing_rate([-1, 1, -1, 1]) == 3
    assert zero_crossing_rate([1, 2, 3]) == 0
