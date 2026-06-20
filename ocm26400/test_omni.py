"""Tests TDD — OmniModel UNIFIÉ (OCM-26400, 'modèle omni unifié, pas de wrapper').

Valide : un SEUL modèle, noyau AMV partagé entre modalités, têtes de classification ET
de génération, loss joint différentiable (paradigme d'entraînement complet).
"""
import torch

from ocm26400.omni import OmniModel, joint_loss
from ocm26400.reasoner import ReasonerBlock
from ocm26400.spectral_core import SpectralCoreBlock


def _model():
    return OmniModel(n_audio_classes=5, n_image_classes=10, audio_feat=32, img_side=8)


def _audio_batch(B=4):
    return torch.randn(B, 1200)              # waveform (B, T)


def _image_batch(B=4):
    return torch.randn(B, 1, 8, 8)           # (B, C, H, W)


def test_omni_encode_to_shared_amv():
    """Audio ET image -> même espace AMV (B, 256), via le MÊME noyau partagé."""
    m = _model()
    amv_a = m.encode("audio", _audio_batch())
    amv_i = m.encode("image", _image_batch())
    assert amv_a.shape == (4, 256) and amv_i.shape == (4, 256)
    # un seul noyau partagé (spectral par défaut = archi utilisateur, ou MLP) = unifié, pas wrapper
    cores = [mod for mod in m.modules() if isinstance(mod, (ReasonerBlock, SpectralCoreBlock))]
    assert len(cores) == 1


def test_omni_classify_shapes():
    m = _model()
    assert m.classify("audio", _audio_batch()).shape == (4, 5)
    assert m.classify("image", _image_batch()).shape == (4, 10)


def test_omni_generate_shapes():
    """Génération conditionnée (label -> signal) : shapes correctes, GÉNÉRATION APPRISE."""
    m = _model()
    label = torch.tensor([0, 1, 2, 3])
    assert m.generate("audio", label).shape == (4, 32)        # features audio générés
    assert m.generate("image", label).shape == (4, 64)        # pixels (8x8) générés


def test_joint_loss_differentiable_across_modalities():
    """Loss multi-tâche (classify + generate) : gradient sur noyau + toutes les têtes."""
    m = _model()
    ya = torch.tensor([0, 1, 2, 3]); yi = torch.tensor([0, 1, 2, 3])
    batch = {
        "audio": {"x": _audio_batch(), "y": ya, "feat": torch.randn(4, 32)},
        "image": {"x": _image_batch(), "y": yi, "feat": torch.randn(4, 64)},
    }
    loss, parts = joint_loss(m, batch)
    assert loss.requires_grad
    loss.backward()
    # noyau partagé + têtes classif + têtes génération ont toutes reçu du gradient
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.core.parameters())
    assert m.audio_cls.weight.grad is not None and m.image_cls.weight.grad is not None
    assert m.audio_dec.weight.grad is not None and m.image_dec.weight.grad is not None
    assert "audio_gen" in parts and "image_cls" in parts


def test_one_model_multiple_competencies():
    """UN modèle, DEUX modalités : classify + generate pour chacune (unification réelle)."""
    m = _model()
    out = {
        "audio_cls": m.classify("audio", _audio_batch(2)).shape,
        "image_cls": m.classify("image", _image_batch(2)).shape,
        "audio_gen": m.generate("audio", torch.tensor([0, 1])).shape,
        "image_gen": m.generate("image", torch.tensor([0, 1])).shape,
    }
    assert out["audio_cls"] == (2, 5) and out["image_cls"] == (2, 10)
    assert out["audio_gen"] == (2, 32) and out["image_gen"] == (2, 64)
