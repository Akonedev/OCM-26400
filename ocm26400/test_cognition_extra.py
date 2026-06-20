"""Tests commonsense (M2) + stream (H16)."""
from ocm26400.commonsense import CommonSense, default_commonsense
from ocm26400.stream import stream_chars, stream_words, stream_cot, TokenStream


def test_commonsense_causal_inference():
    cs = default_commonsense()
    assert "verre_cassé" in cs.what_happens(["verre_tombe"])
    assert "sol_mouillé" in cs.what_happens(["pluie"])
    assert "glace_fond" in cs.what_happens(["glace_chaleur"])


def test_commonsense_properties():
    cs = default_commonsense()
    assert cs.has_property("feu", "chaud")
    assert cs.has_property("verre", "fragile")
    assert not cs.has_property("pierre", "fragile")


def test_commonsense_answer_question():
    cs = default_commonsense()
    a = cs.answer("que se passe-t-il si le verre tombe ?")
    assert "verre" in a and "cass" in a


def test_commonsense_abstains_on_unknown():
    cs = default_commonsense()
    assert "abstention" in cs.answer("combien de pattes a un sphinx ?") or "sait pas" in cs.answer("xyz qwerty abc")


def test_stream_chars():
    out = list(stream_chars("abc"))
    assert out == ["a", "b", "c"]


def test_stream_words():
    out = list(stream_words("hello world foo"))
    assert out == ["hello", "world", "foo"]


def test_stream_cot_numbered():
    out = list(stream_cot(["a", "b"]))
    assert out == ["Step 1: a", "Step 2: b"]


def test_token_stream_consume():
    ts = TokenStream()
    seen = []
    full = ts.consume(stream_chars("XY"), on_token=seen.append)
    assert full == "XY" and seen == ["X", "Y"]
    assert ts.collected() == "XY"


def test_stream_forward_chaining_fixpoint():
    """Le chaînage causal atteint un point fixe (ne boucle pas)."""
    cs = CommonSense(facts={"verre_tombe"})
    derived = cs.infer()
    assert "verre_cassé" in derived
    assert len(derived) < 100    # pas d'explosion infinie
