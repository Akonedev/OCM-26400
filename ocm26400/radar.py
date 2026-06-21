"""Traitement radar/SAR — Range-Doppler, détection cibles — modalité radar.

Le SAR (Synthetic Aperture Radar) et le radar classique traitent les échos radio pour
détecter/localiser des cibles. On implémente les primitives radar RÉELLES :
* Range-Doppler : FFT 2D (range × Doppler) pour séparer cibles par distance + vitesse.
* CFAR (Constant False Alarm Rate) : détection adaptative de cibles dans le bruit.
* Profile de portée (range profile) : FFT sur les échos → distances.
* Compression d'impulsion : corrélation écho/impulsion émise (matched filter).

Pas de dataset SENTINEL externe, mais MÉCANISMES radar réels (FFT 2D, CFAR, matched filter).
Suit la philosophie spectrale du projet (FFT = cœur du traitement radar).
"""
from __future__ import annotations
import math
from typing import List, Tuple

import torch
import numpy as np


def range_profile(echo: np.ndarray, n_range: int = 64) -> np.ndarray:
    """Profile de portée : FFT sur l'écho reçu → distances (pics aux positions des cibles).
    L'écho est la somme des réflections retardées par l'impulsion émise."""
    spectrum = np.fft.fft(echo, n=n_range)
    return np.abs(spectrum)


def range_doppler_matrix(echoes: np.ndarray) -> np.ndarray:
    """Carte Range-Doppler 2D : FFT sur les échos (lents=temps, rapides=portée).
    echoes: (n_pulses, n_samples). Retourne (n_doppler, n_range) en dB."""
    # FFT rapide (range) sur chaque pulse
    rd = np.fft.fft(echoes, axis=1)
    # FFT lente (Doppler) sur les pulses
    rd = np.fft.fft(rd, axis=0)
    mag = np.abs(rd)
    return 20 * np.log10(mag + 1e-10)   # dB


def matched_filter(echo: np.ndarray, pulse: np.ndarray) -> np.ndarray:
    """Compression d'impulsion : corrélation écho/impulsion (matched filter).
    Amplifie le SNR et résout les cibles proches."""
    correlation = np.correlate(echo, pulse, mode="full")
    return np.abs(correlation)


def cfar_detection(range_profile_data: np.ndarray, guard: int = 2,
                   train: int = 4, pfa: float = 1e-3) -> List[int]:
    """CFAR (Cell Averaging) : détecte les cibles au-dessus d'un seuil adaptatif.
    Pour chaque cellule, compare au niveau moyen des cellules d'entraînement voisines
    (en excluant les cellules de garde). Détecte si la cellule > seuil × moyenne."""
    n = len(range_profile_data)
    threshold_factor = math.sqrt(-math.log(pfa))   # approximation du facteur CFAR
    detections = []
    for i in range(train + guard, n - train - guard):
        # cellules d'entraînement (gauche + droite, hors garde)
        left = range_profile_data[i - train - guard:i - guard]
        right = range_profile_data[i + guard + 1:i + guard + 1 + train]
        noise_level = np.mean(np.concatenate([left, right]))
        threshold = noise_level * (1 + threshold_factor)
        if range_profile_data[i] > threshold:
            detections.append(i)
    return detections


def simulate_radar_returns(n_targets: int = 3, n_samples: int = 128,
                           n_pulses: int = 32, snr_db: float = 10,
                           seed: int = 0) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Simule des échos radar : n_targets cibles à des (range, doppler) aléatoires.
    Retourne (echoes (n_pulses, n_samples), ground_truth [(range_bin, doppler_bin), ...])."""
    rng = np.random.RandomState(seed)
    echoes = np.zeros((n_pulses, n_samples), dtype=complex)
    gt = []
    for _ in range(n_targets):
        r_bin = rng.randint(10, n_samples - 10)     # range bin
        d_bin = rng.randint(0, n_pulses)             # doppler bin
        gt.append((r_bin, d_bin))
        # la cible contribue une sinusoïde à ce range/doppler
        amplitude = 10 ** (snr_db / 20)
        for p in range(n_pulses):
            doppler_phase = np.exp(2j * np.pi * d_bin * p / n_pulses)
            echoes[p, r_bin] += amplitude * doppler_phase
    # bruit gaussien
    noise = (rng.randn(n_pulses, n_samples) + 1j * rng.randn(n_pulses, n_samples))
    echoes += noise
    return echoes, gt


def evaluate_radar(n_targets: int = 3, snr_db: float = 10, seed: int = 0) -> dict:
    """Évalue le traitement radar : simule des cibles → Range-Doppler → CFAR → compare."""
    echoes, gt = simulate_radar_returns(n_targets, snr_db=snr_db, seed=seed)
    rd = range_doppler_matrix(echoes)
    # CFAR sur chaque ligne Doppler (détection de portée)
    detections = []
    for d_bin in range(rd.shape[0]):
        det = cfar_detection(rd[d_bin], guard=1, train=3, pfa=1e-2)
        for r_bin in det:
            detections.append((r_bin, d_bin))
    # évaluation : combien de GT détectés (within ±2 bins)
    n_correct = 0
    for gt_r, gt_d in gt:
        for dr, dd in detections:
            if abs(dr - gt_r) <= 2 and abs(dd - gt_d) <= 2:
                n_correct += 1
                break
    return {
        "task": "radar Range-Doppler + CFAR",
        "n_targets": n_targets, "n_detected": len(detections),
        "n_correct": n_correct,
        "detection_rate": round(n_correct / max(n_targets, 1), 3),
        "snr_db": snr_db, "archi": "FFT 2D + CFAR (traitement spectral)",
    }


if __name__ == "__main__":
    for snr in [5, 10, 15, 20]:
        res = evaluate_radar(n_targets=3, snr_db=snr, seed=0)
        print(f"[radar] SNR={snr}dB : {res['n_correct']}/{res['n_targets']} cibles détectées "
              f"({res['detection_rate']*100:.0f}%) | {res['archi']}")
