"""Tests TDD — tool-use / apprentissage externe (OCM-26400, spec 'je sais pas->rechercher->apprendre').

Valide le cycle retrieve -> outil -> apprendre / abstention, et la rétention
(re-poser la question => retrouvé car appris).
"""
from ocm26400.learned_vocab import LearnedVocab
from ocm26400.knowledge_base import KnowledgeBase
from ocm26400.tools import StaticTool, LearningAgent


def _setup():
    vocab = LearnedVocab(n=20, init="ortho", seed=0).freeze()
    kb = KnowledgeBase(vocab, threshold=0.5)
    # question (str) -> vecteur concept (canonical(i))
    concept_vectors = {f"q{i}": vocab.canonical(i) for i in range(20)}
    return vocab, kb, concept_vectors


def test_static_tool_returns_known_answers():
    tool = StaticTool({"capitale france": "Paris", "capitale espagne": "Madrid"})
    assert tool.query("capitale france") == "Paris"
    assert tool.query("inconnu") is None


def test_ask_retrieves_when_known():
    vocab, kb, cv = _setup()
    kb.store(3, "Paris")
    ag = LearningAgent(kb, StaticTool({}), concept_vectors=cv)
    val, mode = ag.ask("q3")
    assert mode == "retrieved"
    assert val == "Paris"


def test_ask_learns_from_tool_when_unknown():
    """KB vide, l'outil a la réponse -> apprise depuis l'externe."""
    vocab, kb, cv = _setup()
    tool = StaticTool({"q7": "Madrid"})
    ag = LearningAgent(kb, tool, concept_vectors=cv)
    val, mode = ag.ask("q7")
    assert mode == "learned_from_tool"
    assert val == "Madrid"
    assert ag.knows_now("q7")          # désormais connu


def test_reask_after_learning_is_retrieved():
    """Spec : 'en reposant la même question, répond directement car apprise'."""
    vocab, kb, cv = _setup()
    ag = LearningAgent(kb, StaticTool({"q5": "Londres"}), concept_vectors=cv)
    v1, m1 = ag.ask("q5")
    assert m1 == "learned_from_tool"
    v2, m2 = ag.ask("q5")              # re-pose
    assert m2 == "retrieved"           # cette fois : mémoire
    assert v2 == "Londres"


def test_ask_abstains_when_no_source():
    """KB vide + outil sans réponse -> abstention honnête."""
    vocab, kb, cv = _setup()
    ag = LearningAgent(kb, StaticTool({}), concept_vectors=cv)
    val, mode = ag.ask("q_inexistante")
    assert mode == "abstained"
    assert val is None
