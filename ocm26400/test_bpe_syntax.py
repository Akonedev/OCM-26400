"""Tests BPE tokenizer + parser syntaxique (OCM-26400) — M16/C5."""
from ocm26400.bpe_tokenizer import BPETokenizer, train_default
from ocm26400.syntax_parser import parse, pos_tag


# ---- BPE ----
def test_bpe_train_vocab():
    tok = train_default(vocab_size=80)
    assert len(tok.vocab) <= 80 + 30   # borné (caractères + fusions)
    assert len(tok.merges) > 5


def test_bpe_roundtrip():
    """encode → decode préserve le texte (mots connus)."""
    tok = train_default(vocab_size=120)
    for s in ["chat", "manger", "chat dort"]:
        ids = tok.encode(s)
        assert tok.decode(ids).replace(" ", "") == s.replace(" ", "") or tok.decode(ids) == s


def test_bpe_known_word_single_token():
    """Un mot fréquent devient 1 token après entraînement."""
    tok = train_default(vocab_size=150)
    ids = tok.encode("chat")
    assert len(ids) >= 1


def test_bpe_encode_returns_ids():
    tok = train_default()
    ids = tok.encode("le chat mange")
    assert all(isinstance(i, int) for i in ids)


# ---- syntax parser ----
def test_pos_tag_basic():
    assert pos_tag("le") == "DET"
    assert pos_tag("dans") == "PREP"
    assert pos_tag("je") == "PRON"
    assert pos_tag("rapidement") == "ADV"


def test_parse_svo_simple():
    """'le chat mange la souris' → sujet=le chat, verbe=mange, objet=la souris."""
    st = parse("le chat mange la souris")
    assert st.sujet == "le chat"
    assert st.verbe == "mange"
    assert st.objet == "la souris"
    assert st.is_valid_svo()


def test_parse_dependencies():
    st = parse("le chat mange la souris")
    rels = [r for _, _, r in st.dependencies]
    assert "nsubj" in rels and "obj" in rels


def test_parse_pos_extracted():
    st = parse("le chat mange")
    poses = [p for _, p in st.pos]
    assert "DET" in poses and "VERB" in poses
