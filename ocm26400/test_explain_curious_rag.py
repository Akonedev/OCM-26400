"""Tests explainer / curiosity / multihop_rag — M9/M21/M11."""
from ocm26400.explainer import explain_deduction, explain_abstention, explain_trace, Explanation
from ocm26400.curiosity import CuriosityDrive, NoveltyMemory, select_curious
from ocm26400.multihop_rag import rerank, extract_entities, multi_hop_retrieve
from ocm26400.document_learner import DocumentLearner
import torch


def test_explain_deduction():
    e = explain_deduction("q?", "a", ["p1", "p2"], "modus ponens", ["s1"])
    assert e.reasoning_type == "déduction"
    assert e.answer == "a" and "modus ponens" in e.rules_applied
    assert "Prémisses" in e.render()


def test_explain_abstention():
    e = explain_abstention("q?", "inconnu")
    assert e.reasoning_type == "abstention" and e.answer is None
    assert e.confidence == 0.0


def test_explain_trace():
    e = explain_trace([("3*4=12", "multiplication"), ("12+5=17", "addition")], "q", 17)
    assert len(e.steps) == 2 and len(e.premises) == 2


def test_novelty_familiar_vs_new():
    mem = NoveltyMemory()
    s = torch.ones(8)
    mem.observe(s)
    assert mem.novelty(s) < 0.1            # familier
    assert mem.novelty(torch.randn(8)) > 0.5   # nouveau


def test_curiosity_selects_novel():
    drive = CuriosityDrive(dim=8)
    for _ in range(3):
        drive.observe(torch.ones(8) * 0.5)
    idx, _ = select_curious(drive, [torch.ones(8) * 0.5, torch.randn(8)])
    assert idx == 1     # choisit le nouveau


def test_curiosity_intrinsic_reward():
    drive = CuriosityDrive(dim=8)
    drive.observe(torch.ones(8))
    r_fam = drive.intrinsic_reward(torch.ones(8))
    r_new = drive.intrinsic_reward(torch.randn(8))
    assert r_new >= r_fam


def test_extract_entities():
    ents = extract_entities("Apple est une entreprise. Steve Jobs l'a fondée.")
    assert len(ents) > 0


def test_multihop_rag_chains():
    dl = DocumentLearner(threshold=0.3, margin=0.02)
    dl.learn_text("L'iPhone créé par Apple.", "a")
    dl.learn_text("Apple fondée par Jobs.", "b")
    res = multi_hop_retrieve(dl, "iPhone", hops=2)
    assert res["n_hops"] >= 1
    assert len(res["sources"]) >= 1


def test_rerank_orders():
    dl = DocumentLearner()
    cands = ["le chat noir dort", "la voiture rouge", "chat et souris"]
    ranked = rerank(dl, "chat", cands, top_k=2)
    assert len(ranked) == 2
    assert "chat" in ranked[0][0]
