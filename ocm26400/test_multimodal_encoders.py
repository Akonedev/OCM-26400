"""Tests TDD — encodeurs multimodaux RÉELS audio/image (OCM-26400).

Valide que les encodeurs ingèrent de vrais signaux (waveform, image) -> embedding 64,
differentiables, et distinguent des signaux différents. Signaux synthétiques (le repo
n'a pas de corpus audio/image étiqueté).
"""
import torch

from ocm26400.multimodal_encoders import (
    AudioEncoder, ImageEncoder, synth_tone, synth_image,
)
from ocm26400.infonce import multimodal_l_consist


def test_audio_encoder_shape_and_grad():
    enc = AudioEncoder(out_dim=64, n_fft=64)
    wav = torch.stack([synth_tone(440.0), synth_tone(880.0)])   # (2, T)
    emb = enc(wav)
    assert emb.shape == (2, 64)
    emb.sum().backward()
    assert enc.conv.weight.grad is not None


def test_image_encoder_shape_and_grad():
    enc = ImageEncoder(out_dim=64, patch=8)
    img = torch.stack([synth_image(32, seed=0), synth_image(32, seed=1)])  # (2,3,32,32)
    emb = enc(img)
    assert emb.shape == (2, 64)
    emb.sum().backward()
    assert enc.proj.weight.grad is not None


def test_audio_distinguishes_frequencies():
    """Deux tons de fréquences différentes -> embeddings différents."""
    enc = AudioEncoder(out_dim=64, n_fft=64)
    wav = torch.stack([synth_tone(220.0), synth_tone(2000.0)])
    emb = enc(wav)
    assert not torch.allclose(emb[0], emb[1])


def test_image_distinguishes_content():
    """Deux images différentes -> embeddings différents."""
    enc = ImageEncoder(out_dim=64, patch=8)
    img = torch.stack([synth_image(32, seed=0), synth_image(32, seed=1)])
    emb = enc(img)
    assert not torch.allclose(emb[0], emb[1])


def test_amodal_accepts_signal_views():
    """Les embeddings audio/image s'alimentent dans l'alignement amodal (InfoNCE)."""
    N = 6
    aud = AudioEncoder(out_dim=64, n_fft=64)
    vid = ImageEncoder(out_dim=64, patch=8)
    a = aud(torch.stack([synth_tone(100 + 50 * i) for i in range(N)]))   # (N,64)
    v = vid(torch.stack([synth_image(16, seed=i) for i in range(N)]))    # (N,64)
    loss = multimodal_l_consist([a, v])                                    # aligne audio<->image
    assert loss.requires_grad
    loss.backward()
