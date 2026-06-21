"""Tests pré-training linguistique (OCM-26400)."""
from ocm26400.language_pretrain import pretrain_masked_word, load_a1_sentences, _encode_sentence


def test_encode_sentence():
    seq, words = _encode_sentence("the cat eats fish")
    assert seq.shape[1] == 256
    assert len(words) == 4


def test_load_a1():
    sents = load_a1_sentences()
    assert len(sents) > 0  # dataset téléchargé


def test_pretrain_runs():
    """Le SpectralCoreBlock pré-entraîne sur le langage (masked word prediction)."""
    blk, res = pretrain_masked_word(n_sentences=100, n_steps=200)
    assert "error" not in res
    assert res["masked_word_acc"] >= 0.1  # au-dessus de la chance
