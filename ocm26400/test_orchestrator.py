"""Tests TDD — orchestrateur multi-agents + MoE + DA + Juge (OCM-26400).

Valide le cadre : dispatch parallèle d'experts, critique DA, arbitrage juge (quorum),
routage MoE, et la mise à l'échelle à des centaines d'agents.
"""
from ocm26400.orchestrator import (
    ExpertAgent, DevAdvocate, Judge, MoERouter, Orchestrator,
)


def _agree_experts(n, answer="4", conf=0.9):
    return [ExpertAgent(f"e{i}", "math", lambda q, a=answer, c=conf: (a, c)) for i in range(n)]


def test_moe_routes_to_relevant_domain():
    router = MoERouter({"math": ["calcul", "plus", "nombre"],
                        "geo": ["capitale", "pays", "ville"]})
    assert router.route("quelle est la capitale du pays X") == ["geo"]
    assert router.route("calcule 2 plus 3") == ["math"]


def test_orchestrator_dispatches_experts_and_judge():
    """3 experts d'accord -> verdict = réponse commune, quorum atteint."""
    orch = Orchestrator(_agree_experts(3, "Paris"),
                        advocates=[DevAdvocate("da", lambda a, q: ("ok", 0.05))],
                        judge=Judge(quorum=0.5))
    res = orch.run("capitale ?")
    assert res["verdict"] == "Paris"
    assert res["confidence"] >= 0.5
    assert res["n_experts"] == 3


def test_judge_no_quorum_when_advocates_raise_doubt():
    """DA fortement en doute + experts faibles -> pas de quorum (verdict None)."""
    experts = [ExpertAgent("e", "math", lambda q: ("x", 0.4))]
    orch = Orchestrator(experts,
                        advocates=[DevAdvocate("da", lambda a, q: ("réfuté", 0.95))],
                        judge=Judge(quorum=0.6))
    res = orch.run("q ?")
    assert res["verdict"] is None


def test_dev_advocate_reduces_final_confidence():
    """Même experts, un DA qui doute fort baisse la confiance finale vs sans doute."""
    base = _agree_experts(3, "42", conf=0.9)

    soft = Orchestrator(base, advocates=[DevAdvocate("da", lambda a, q: ("ok", 0.0))]).run("q")
    hard = Orchestrator(base, advocates=[DevAdvocate("da", lambda a, q: ("non", 0.8))]).run("q")
    assert hard["confidence"] < soft["confidence"]


def test_orchestrator_scales_to_many_agents():
    """Lancable à des centaines d'agents (parallèle, profondeur > taille)."""
    n = 200
    orch = Orchestrator(_agree_experts(n, "ok", conf=0.95),
                        advocates=[DevAdvocate("da", lambda a, q: ("", 0.0))],
                        judge=Judge(quorum=0.5), max_workers=16)
    res = orch.run("q ?")
    assert res["n_experts"] == n
    assert res["verdict"] == "ok"


def test_moe_selects_only_routed_experts():
    """MoE : seuls les experts du domaine routé sont consultés."""
    experts = [ExpertAgent("math", "math", lambda q: ("r_math", 0.9)),
               ExpertAgent("geo", "geo", lambda q: ("r_geo", 0.9))]
    router = MoERouter({"math": ["calcul"], "geo": ["capitale"]})
    orch = Orchestrator(experts, advocates=[DevAdvocate("da", lambda a, q: ("", 0.0))],
                        router=router, judge=Judge(quorum=0.4))
    res = orch.run("calcule 1+1")
    assert res["n_experts"] == 1
    assert res["verdict"] == "r_math"
