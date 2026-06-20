"""Tests document learner (cycle PDF/URL → KB → retrieval + citations) — OCM-26400."""
from ocm26400.document_learner import DocumentLearner, text_embedding


def test_text_embedding_normalized():
    v = text_embedding("la photosynthèse produit du glucose")
    assert v.shape[0] == 64
    assert abs(float(v.norm()) - 1.0) < 1e-3       # normalisé


def test_learn_text_chunks():
    dl = DocumentLearner()
    n = dl.learn_text("a b c d e f g h i j k l m n o p q r s t u v w x y z " * 5, "doc")
    assert dl.size() == n and n >= 1


def test_retrieve_returns_source_citation():
    dl = DocumentLearner()
    dl.learn_text("La deuxième loi de Newton donne F=ma, force masse accélération.", "phys")
    res = dl.retrieve("deuxième loi de Newton force", top_k=1)
    assert len(res) == 1
    chunk, source, conf = res[0]
    assert source == "phys"
    assert 0.0 <= conf <= 1.0


def test_answer_or_abstention():
    dl = DocumentLearner()
    dl.learn_text("La photosynthèse produit du glucose et de l'oxygène chez les plantes.", "bio")
    # requête pertinente → répond
    chunk, src, conf = dl.answer("photosynthèse glucose plantes")
    assert chunk is not None and src == "bio"
    # requête OOD (aucun doc) → abstention
    dl2 = DocumentLearner()
    chunk2, src2, conf2 = dl2.answer("capitale du Brésil")
    assert chunk2 is None              # abstention épistémique


def test_learn_url_real_or_safe_fail():
    """learn_url : soit apprend (réseau), soit échoue proprement (pas d'exception)."""
    dl = DocumentLearner()
    n = dl.learn_url("https://example.com")
    assert isinstance(n, int) and n >= 0     # jamais d'exception


def test_size_grows_with_learning():
    dl = DocumentLearner()
    assert dl.size() == 0
    dl.learn_text("premier document avec assez de mots pour chunker ici.", "a")
    s1 = dl.size()
    dl.learn_text("second document distinct avec d autres mots pour chunker.", "b")
    assert dl.size() >= s1
