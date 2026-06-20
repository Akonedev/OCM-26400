"""Tests primitives linguistiques + lemmatiseur branché sur traduction."""
from ocm26400.linguistics import (
    capture_all, phonemes, syllables, morphemes_of, classify_affix,
    tmesis_analysis, etymology_family, semantic_traits,
)
from ocm26400.language_primitives import lemmatize_fr, lemmatize_en, inflect_adjective, to_adverb
from ocm26400.nlp_tools import translate_word


def test_lemmatize_fr_regular():
    assert lemmatize_fr("mange") == "manger"
    assert lemmatize_fr("mangé") == "manger"
    assert lemmatize_fr("chantait") == "chanter"


def test_lemmatize_fr_irregular():
    assert lemmatize_fr("sont") == "être"
    assert lemmatize_fr("allez") == "aller"


def test_lemmatize_fr_infinitive_kept():
    """Un infinitif (-er/-ir/-re) est gardé intact (pas de faux lemmatize)."""
    assert lemmatize_fr("défaire") == "défaire"
    assert lemmatize_fr("manger") == "manger"


def test_lemmatize_en():
    assert lemmatize_en("running") == "run"
    assert lemmatize_en("cats") == "cat"
    assert lemmatize_en("went") == "go"
    assert lemmatize_en("happiest") == "happy"


def test_translate_uses_lemmatizer():
    """LE test feedback : 'mange' (conjugué) traduit via le lemme 'manger'."""
    assert translate_word("mange") == "eat"
    assert translate_word("mangé") == "eat"


def test_inflect_adjective():
    assert inflect_adjective("grand", feminine=True) == "grande"
    assert inflect_adjective("beau", feminine=True) == "belle"
    assert inflect_adjective("vif", feminine=True) == "vive"
    assert inflect_adjective("grand", feminine=True, plural=True) == "grandes"


def test_to_adverb():
    assert to_adverb("rapide") == "rapidement"
    assert to_adverb("vif") == "vivement"
    assert to_adverb("constant") == "constamment"


def test_phonemes():
    ph = phonemes("chat")
    assert "/ʃ/" in ph and "/a/" in ph


def test_syllables():
    assert len(syllables("bonjour")) >= 2


def test_morphemes_decomposition():
    m = morphemes_of("rapidement")
    assert m.radical == "rapid"
    assert "ement" in m.suffixes or "ment" in m.suffixes


def test_morphemes_prefix():
    m = morphemes_of("défaire")
    assert "dé" in m.prefixes and m.radical == "faire"


def test_affix_classification():
    assert "sémantique" in classify_affix("dé")
    assert "grammatical" in classify_affix("s")


def test_tmesis_detection():
    t = tmesis_analysis("aujourd'hui")
    assert t["has_tmesis"] is True


def test_capture_all_comprehensive():
    """capture_all extrait TOUTES les primitives en une passe."""
    c = capture_all("chat")
    d = c.to_dict()
    assert d["phonemes"]                   # phonèmes présents
    assert d["semantic_traits"]            # traits sémantiques
    assert "category" in d and "lexeme" in d and "syllables" in d


def test_capture_all_rapidement():
    c = capture_all("rapidement")
    d = c.to_dict()
    assert d["radical"] == "rapid"
    assert d["derivation_type"] == "dérivationnel"


def test_etymology_family():
    fam = etymology_family("lumineux")
    assert "lumière" in fam or "lumineux" in fam
