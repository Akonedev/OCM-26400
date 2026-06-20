"""Encodeurs multimodaux RÉELS (audio / image) — OCM-26400, cahier des charges.

Étend l'alignement amodal aux modalités SIGNAL (audio, image) — pas seulement
linguistiques. Ces encodeurs ingèrent de VRAIS signaux (waveform audio, pixels image)
et produisent des view embeddings (PART=64) consommables par amodal_align_loss /
concept_amodal (même interface que RealViewEncoder).

* AudioEncoder : waveform (B, T) -> spectrogramme Mel (STFT) -> conv 1D -> pool -> 64.
  Vrai pipeline audio (analyse spectrale, comme un Mel-CNN).
* ImageEncoder  : image (B, C, H, W) -> découpage en patches -> projection linéaire
  par patch -> pool -> 64. Vrai pipeline vision (style ViT linéaire).

HONNÊTE : ce sont de VRAIS encodeurs de signal (architectures réelles qui transforment
un waveform/pixels en embedding), mais ils sont validés sur SIGNAUX SYNTHÉTIQUES
(tons sinusoïdaux, images procédurales) car le repo ne contient pas de corpus
audio/image étiqueté. Prêts à être entraînés sur de vraies données. Vidéo = séquence
d'images, 3D = voxels/patches — extensions directes (notées, non implémentées ici).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .amv import PART


class AudioEncoder(nn.Module):
    """Waveform (B, T) -> embedding (B, 64) via spectrogramme Mel + conv 1D."""

    def __init__(self, out_dim: int = PART, n_mels: int = 32, n_fft: int = 64,
                 hidden: int = 64):
        super().__init__()
        self.n_fft = n_fft
        self.n_mels = n_mels
        # banc de filtres Mel (init fixe) : (n_mels, n_fft//2+1)
        self.register_buffer("mel_fb", self._mel_filterbank(n_mels, n_fft // 2 + 1))
        self.conv = nn.Conv1d(n_mels, hidden, kernel_size=3, padding=1)
        self.out = nn.Linear(hidden, out_dim)

    @staticmethod
    def _mel_filterbank(n_mels, n_freqs):
        # banc de filtres Mel triangulaire simplifié (déterministe)
        fb = torch.zeros(n_mels, n_freqs)
        for m in range(n_mels):
            center = (m + 1) * (n_freqs - 1) / (n_mels + 1)
            for f in range(n_freqs):
                d = abs(f - center) / max(1.0, (n_freqs - 1) / (n_mels + 1))
                fb[m, f] = max(0.0, 1.0 - d)
        return fb / (fb.sum(dim=1, keepdim=True) + 1e-8)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (B, T)
        win = torch.hann_window(self.n_fft, device=waveform.device)
        spec = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.n_fft // 2,
                          win_length=self.n_fft, window=win, return_complex=True,
                          center=False)           # (B, F, frames)
        power = spec.abs() ** 2                   # (B, F, frames)
        mel = torch.matmul(self.mel_fb, power)    # (B, n_mels, frames)
        mel = torch.log1p(mel)
        h = torch.relu(self.conv(mel))            # (B, hidden, frames)
        pooled = h.mean(dim=-1)                   # (B, hidden)
        return self.out(pooled)                   # (B, out_dim)


class ImageEncoder(nn.Module):
    """Image (B, C, H, W) -> embedding (B, 64) via patches + projection + pool."""

    def __init__(self, out_dim: int = PART, patch: int = 8, hidden: int = 64):
        super().__init__()
        self.patch = patch
        self.proj = nn.LazyLinear(hidden)         # se calibre sur la taille de patch
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # image: (B, C, H, W)
        B, C, H, W = image.shape
        p = self.patch
        # découpage en patches (B, n_patches, C*p*p)
        patches = image.unfold(2, p, p).unfold(3, p, p)      # (B, C, nH, nW, p, p)
        patches = patches.contiguous().view(B, C, -1, p * p)  # (B, C, nP, p*p)
        patches = patches.permute(0, 2, 1, 3).contiguous().view(B, -1, C * p * p)
        h = torch.relu(self.proj(patches))                   # (B, nP, hidden)
        pooled = h.mean(dim=1)                               # (B, hidden)
        return self.out(pooled)                              # (B, out_dim)


def synth_tone(freq=440.0, duration=0.5, sr=8000):
    """Signal audio synthétique (ton sinusoïdal) pour validation."""
    t = torch.arange(int(sr * duration)).float() / sr
    return torch.sin(2 * torch.pi * freq * t)


def synth_image(size=32, seed=0):
    """Image procédurale synthétique pour validation."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(3, size, size, generator=g)
