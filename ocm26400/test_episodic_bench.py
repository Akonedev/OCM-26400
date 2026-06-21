"""Tests mémoire épisodique + bench public (OCM-26400)."""
import torch
from ocm26400.episodic_memory import EpisodicMemory, Episode
from ocm26400.bench_public import run_all_public_benchmarks


def test_episodic_store_recall():
    mem = EpisodicMemory(dim=64)
    mem.store("chat jardin", {"lieu": "jardin"}, outcome="vu")
    mem.store("chien parc", {"lieu": "parc"}, outcome="vu")
    assert mem.size() == 2
    found = mem.recall_by_content("chat", top_k=1)
    assert len(found) == 1 and "chat" in found[0].content


def test_episodic_consolidate():
    mem = EpisodicMemory(dim=64)
    for _ in range(3):
        mem.store("chat vu", outcome="observé")
    rule = mem.consolidate_to_rule()
    assert rule is not None and "observé" in rule


def test_episodic_replay():
    mem = EpisodicMemory(dim=64)
    for i in range(5):
        mem.store(f"épisode {i}")
    replayed = mem.replay(3)
    assert len(replayed) == 3


def test_episodic_embedding_recall():
    mem = EpisodicMemory(dim=64)
    mem.store("a", embedding=torch.ones(64))
    mem.store("b", embedding=torch.zeros(64))
    found = mem.recall(torch.ones(64), top_k=1)
    assert len(found) == 1 and found[0][0].content == "a"


def test_bench_public_runs():
    rep = run_all_public_benchmarks(quick=True)
    assert rep["model"] == "OCM-26400"
    assert rep["no_transformer"] is True
    assert "benchmarks" in rep
