"""Tests capture multimodale unifiée (OCM-26400)."""
import torch
from ocm26400.unified_capture import UnifiedCapture, ConceptCapture


def test_capture_multiple_modalities_one_pass():
    """Capture en une passe : texte + audio + image d'un concept."""
    uc = UnifiedCapture(dim=64)
    cap = uc.capture_concept("chat",
                             audio=torch.sin(torch.linspace(0, 6.28, 800)),
                             image=torch.randn(28, 28),
                             text_vec=torch.randn(64))
    mods = cap.modalities()
    assert "audio" in mods and "image" in mods and "text" in mods


def test_concept_similarity_runs():
    uc = UnifiedCapture(dim=64)
    uc.capture_concept("a", text_vec=torch.ones(64))
    uc.capture_concept("b", text_vec=torch.ones(64))
    s = uc.concepts["a"].similarity(uc.concepts["b"], "text")
    assert abs(s - 1.0) < 1e-4     # vecteurs identiques → cos = 1


def test_associate_returns_ranked():
    uc = UnifiedCapture(dim=64)
    uc.capture_concept("chat", text_vec=torch.ones(64))
    uc.capture_concept("chien", text_vec=torch.zeros(64))
    res = uc.associate("text", torch.ones(64), "text", k=2)
    assert len(res) == 2
    # "chat" (ones) plus similaire à la requête ones que "chien" (zeros)
    names = [r[0] for r in res]
    assert names[0] == "chat"


def test_alignment_quality_reports():
    uc = UnifiedCapture(dim=64)
    uc.capture_concept("a", text_vec=torch.randn(64))
    uc.capture_concept("b", text_vec=torch.randn(64))
    q = uc.alignment_quality()
    assert q["n_concepts"] == 2


def test_capture_optional_modalities():
    """Capture fonctionne même avec une seule modalité."""
    uc = UnifiedCapture(dim=64)
    cap = uc.capture_concept("x", text_vec=torch.randn(64))
    assert cap.modalities() == ["text"]
