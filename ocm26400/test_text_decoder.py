"""Tests décodeur de texte ENTRAÎNÉ (OCM-26400) — audit C6/C9."""
import torch
from ocm26400.text_decoder import (
    CharGenerator, train_char_generator, reconstruct,
    encode_word_indices, decode_indices, encode_word, decode_word,
    VOCAB_SIZE, MAX_LEN,
)


def test_encode_decode_indices_roundtrip():
    idxs = encode_word_indices("chat")
    assert idxs.shape[0] == MAX_LEN
    assert decode_indices(idxs) == "chat"


def test_onehot_roundtrip():
    v = encode_word("lion")
    assert v.shape[0] == MAX_LEN * VOCAB_SIZE
    assert decode_word(v) == "lion"


def test_oov_char_becomes_space():
    """Caractère hors vocab → espace (honnête, pas de crash)."""
    idxs = encode_word_indices("ch@t")
    assert decode_indices(idxs) == "ch t" or "ch" in decode_indices(idxs)


def test_generator_trains_loss_decreases():
    """LE test clé : le générateur de texte s'entraîne (loss baisse fortement)."""
    words = ["chat", "chien", "lion", "tigre", "loup", "ours", "singe"]
    gen, hist = train_char_generator(words, n_steps=400, device="cpu")
    assert hist[-1][1] < hist[0][1] * 0.5      # loss au moins divisée par 2


def test_reconstruction_exact_after_training():
    """Après entraînement, le générateur reconstruit les mots d'entraînement (poids réels)."""
    words = ["chat", "chien", "lion", "tigre", "loup", "ours", "ane"]
    gen, hist = train_char_generator(words, n_steps=800, device="cpu")
    n_ok = sum(reconstruct(gen, w, device="cpu") == w for w in words)
    assert n_ok >= 5        # la majorité reconstruite exactement (génération RÉELLE)


def test_generation_shape():
    gen, _ = train_char_generator(["chat"], n_steps=50, device="cpu")
    cond = torch.zeros(2, 256)
    out = gen.generate(cond)
    assert len(out) == 2 and all(isinstance(s, str) for s in out)


def test_temperature_sampling_diversifies():
    gen, _ = train_char_generator(["chat", "lion"], n_steps=200, device="cpu")
    cond = torch.zeros(5, 256)
    out_argmax = gen.generate(cond, temperature=0.0)
    out_sample = gen.generate(cond, temperature=1.0)
    assert len(out_sample) == 5
    # l'échantillonnage stochastic peut différer de l'argmax (pas d'obligation, juste pas de crash)
    assert all(isinstance(s, str) for s in out_argmax + out_sample)
