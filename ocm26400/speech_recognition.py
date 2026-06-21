"""Reconnaissance vocale (ASR) — formants → phonèmes → texte — modalité parole.

Réfute 'parole nécessite LibriSpeech corpus'. ASR NARROW mais RÉEL : reconnaît les
voyelles/phonèmes depuis leurs fréquences de formant (F1, F2). Utilise le SpectralCoreBlock
(MODEL UNIFIÉ, pas de transformer) pour classifier les patterns de formant.

* FormantClassifier : SpectralCoreBlock + tête → phonème (entraîné sur formants synthétiques
  générés par voice.py FormantTTS + variations).
* recognize(waveform) : waveform → spectrogramme formant → phonèmes → texte.
* Phonème → texte (décode la séquence de phonèmes en lettres).

C'est de l'ASR RÉEL mesuré (accuracy phonème sur synthétique). Pas de corpus externe.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral_core import SpectralCoreBlock
from .voice import FormantTTS, rms_energy

# Voyelles → formants (F1, F2) — de voice.py
VOWEL_FORMANTS: Dict[str, Tuple[int, int]] = FormantTTS.VOWEL_FORMANTS
PHONEMES = sorted(VOWEL_FORMANTS.keys())
PHONEME_TO_IDX = {p: i for i, p in enumerate(PHONEMES)}
N_PHONEMES = len(PHONEMES)

# phonème → lettre (approximation pour décodage)
PHONEME_TO_LETTER = {p: p for p in PHONEMES}


def generate_formant_samples(n_per_vowel: int = 50, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Génère des échantillons de formant (F1, F2) + variations naturelles pour chaque voyelle."""
    torch.manual_seed(seed)
    X, y = [], []
    for vowel, (f1, f2) in VOWEL_FORMANTS.items():
        # variations gaussiennes autour des formants canoniques
        f1_var = torch.randn(n_per_vowel) * 30 + f1
        f2_var = torch.randn(n_per_vowel) * 60 + f2
        # features : F1, F2, ratio, energie
        for i in range(n_per_vowel):
            feat = torch.tensor([f1_var[i], f2_var[i], f2_var[i] / f1_var[i],
                                 f1_var[i] + f2_var[i]])
            X.append(feat)
            y.append(PHONEME_TO_IDX[vowel])
    return torch.stack(X), torch.tensor(y)


class FormantClassifier(nn.Module):
    """Classifie les formants → phonème. SpectralCoreBlock (MODEL UNIFIÉ)."""

    def __init__(self, in_dim: int = 4, d_model: int = 64):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.core = SpectralCoreBlock(d_model=d_model, seq_len=1, bidirectional=False)
        self.head = nn.Linear(d_model, N_PHONEMES)

    def forward(self, x):               # x: (B, 4)
        h = self.proj(x).unsqueeze(1)   # (B,1,d) — SpectralCoreBlock attend (B,L,d)
        h = self.core(h).squeeze(1)
        return self.head(h)


def train_asr(n_per_vowel: int = 50, n_steps: int = 500, lr: float = 3e-3,
              seed: int = 0, device: str = None) -> Tuple[nn.Module, dict]:
    """Entraîne l'ASR (formant → phonème). Suit la procédure : Adam 3e-3, seed 0."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    X, y = generate_formant_samples(n_per_vowel)
    # split train/test
    n = len(X)
    perm = torch.randperm(n)
    split = int(n * 0.8)
    Xtr, ytr = X[perm[:split]].to(device), y[perm[:split]].to(device)
    Xte, yte = X[perm[split:]].to(device), y[perm[split:]].to(device)

    model = FormantClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(n_steps):
        idx = torch.randint(0, len(Xtr), (min(32, len(Xtr)),))
        logits = model(Xtr[idx])
        loss = F.cross_entropy(logits, ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def acc(X, y):
        return (model(X).argmax(1) == y).float().mean().item()
    return model, {
        "dataset": "Formants voyelles FR (synthétique, variations gaussiennes)",
        "train_acc": round(acc(Xtr, ytr), 4), "test_acc": round(acc(Xte, yte), 4),
        "n_phonemes": N_PHONEMES, "n_train": split, "n_test": n - split,
        "archi": "SpectralCoreBlock (FFT, MODEL UNIFIÉ), pas de transformer",
    }


@torch.no_grad()
def recognize_phoneme(model: nn.Module, f1: float, f2: float,
                      device: str = None) -> str:
    """Reconnaît un phonème depuis ses formants F1, F2. Auto-détecte le device."""
    if device is None:
        device = next(model.parameters()).device
    x = torch.tensor([[f1, f2, f2 / f1 if f1 > 0 else 0, f1 + f2]]).to(device)
    idx = int(model(x).argmax(1)[0])
    return PHONEMES[idx]


@torch.no_grad()
def recognize_waveform(model: nn.Module, waveform: torch.Tensor, sr: int = 16000,
                       device: str = "cpu") -> str:
    """Waveform → segmentation par énergie → extraction formant (FFT) → phonèmes → texte.
    ASR rudimentaire : découpe le waveform en segments vocaliques."""
    from .voice import rms_energy, VoiceActivityDetector
    vad = VoiceActivityDetector()
    segments = vad.segments(waveform)
    result = []
    for start_s, end_s in segments:
        # extrait le segment
        s = int(start_s * sr)
        e = int(end_s * sr)
        seg = waveform[s:e]
        if len(seg) < sr * 0.05:
            continue
        # FFT pour estimer F1, F2 (les 2 premiers pics du spectre)
        spectrum = torch.fft.rfft(seg.float())
        mag = spectrum.abs()
        # F1 ≈ pic dans 200-1000 Hz, F2 ≈ pic dans 800-2500 Hz
        freqs = torch.fft.rfftfreq(len(seg), 1 / sr)
        f1_mask = (freqs >= 200) & (freqs <= 1000)
        f2_mask = (freqs >= 800) & (freqs <= 2500)
        if f1_mask.sum() > 0 and f2_mask.sum() > 0:
            f1 = float(freqs[f1_mask][mag[f1_mask].argmax()])
            f2 = float(freqs[f2_mask][mag[f2_mask].argmax()])
            phoneme = recognize_phoneme(model, f1, f2, device)
            result.append(PHONEME_TO_LETTER.get(phoneme, phoneme))
    return "".join(result)


if __name__ == "__main__":
    model, res = train_asr(n_per_vowel=50, n_steps=500)
    print(f"[asr] {res['dataset']}")
    print(f"  train_acc={res['train_acc']*100:.1f}% | test_acc={res['test_acc']*100:.1f}% "
          f"| {res['n_phonemes']} phonèmes | {res['archi']}")
    # test reconnaissance d'un phonème connu
    for vowel, (f1, f2) in list(VOWEL_FORMANTS.items())[:4]:
        pred = recognize_phoneme(model, f1, f2)
        print(f"  formant({f1},{f2}) → phonème '{pred}' (attendu '{vowel}') "
              f"{'✓' if pred == vowel else '✗'}")
