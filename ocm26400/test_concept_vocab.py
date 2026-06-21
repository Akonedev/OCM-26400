"""Tests ConceptVocab (OCM-26400) — IDs numériques pour grokking."""
import torch
from ocm26400.concept_vocab import ConceptVocab, ConceptModel


def test_concept_vocab_numbers():
    cv = ConceptVocab()
    assert cv.get_num_id(0) == 4
    assert cv.get_num_id(16) == 20
    assert cv.get_num_id(16) != cv.get_num_id(3)  # pas de collision


def test_text_to_ids():
    cv = ConceptVocab()
    ids = cv.text_to_ids("Janet has 16 eggs")
    assert cv.START in ids
    assert cv.END in ids
    assert cv.get_num_id(16) in ids  # 16 → NUM_16


def test_ids_roundtrip():
    cv = ConceptVocab()
    ids = cv.text_to_ids("5 + 3 = 8")
    decoded = cv.ids_to_text(ids)
    assert "5" in decoded and "3" in decoded and "8" in decoded


def test_concept_model_forward():
    m = ConceptModel(vocab_size=1000, d_model=256, seq_len=8)
    x = torch.randint(0, 1000, (2, 5))
    out = m(x)
    assert out.shape == (2, 5, 256)


def test_concept_model_loss():
    m = ConceptModel(vocab_size=1000, d_model=256, seq_len=8)
    x = torch.randint(0, 1000, (2, 5))
    t = torch.randint(0, 1000, (2, 5))
    loss = m.loss_1cos(x, t)
    assert loss.item() >= 0  # 1-cos ∈ [0, 2]


def test_no_collision():
    """LE test : 2 nombres différents ont des IDs différents (pas de collision)."""
    cv = ConceptVocab()
    for a, b in [(5, 69), (5, 133), (100, 200), (0, 9999)]:
        assert cv.get_num_id(a) != cv.get_num_id(b), f"collision: {a} and {b}"
