"""Tests TDD — encodeurs multimodaux RÉELS audio/image (OCM-26400).

Valide que les encodeurs ingèrent de vrais signaux (waveform, image) -> embedding 64,
differentiables, et distinguent des signaux différents. Signaux synthétiques (le repo
n'a pas de corpus audio/image étiqueté).
"""
import torch

from ocm26400.multimodal_encoders import (
    AudioEncoder, ImageEncoder, VideoEncoder, ThreeDEncoder,
    synth_tone, synth_image, synth_video, synth_voxel,
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


def test_video_encoder_shape_and_grad():
    """Vidéo (séquence de frames) -> embedding 64, differentiable."""
    enc = VideoEncoder(out_dim=64, patch=4)
    vid = torch.stack([synth_video(frames=4, size=16, seed=i) for i in range(2)])  # (2,4,3,16,16)
    emb = enc(vid)
    assert emb.shape == (2, 64)
    emb.sum().backward()
    assert enc.frame_enc.proj.weight.grad is not None


def test_3d_encoder_shape_and_grad():
    """Volume 3D (voxels) -> embedding 64, differentiable."""
    enc = ThreeDEncoder(out_dim=64)
    vol = torch.stack([synth_voxel(grid=16, seed=i) for i in range(2)])   # (2,1,16,16,16)
    emb = enc(vol)
    assert emb.shape == (2, 64)
    emb.sum().backward()
    assert enc.conv.weight.grad is not None


def test_video_and_3d_distinguish_content():
    """Vidéos / volumes différents -> embeddings différents."""
    venc = VideoEncoder(out_dim=64, patch=4)
    denc = ThreeDEncoder(out_dim=64)
    vids = torch.stack([synth_video(4, 16, 0), synth_video(4, 16, 1)])
    vols = torch.stack([synth_voxel(16, 0), synth_voxel(16, 1)])
    assert not torch.allclose(venc(vids)[0], venc(vids)[1])
    assert not torch.allclose(denc(vols)[0], denc(vols)[1])
