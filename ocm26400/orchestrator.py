"""Orchestrateur multi-agents + MoE + DA + Juge (OCM-26400, cahier des charges).

Le cahier des charges demande : « multi-agents / mixture of experts », « lancer
plusieurs agents en même temps avec un orchestrateur pour guider, vérifier, confirmer,
valider, toujours avec des DA et des Juges », « des centaines d'agents » (profondeur >
taille). On implémente le CADRE d'orchestration — c'est la méthodologie même du projet
(EXPERT_PANEL_VERDICT.md : collège d'experts → Devil's Advocates → juge).

* ExpertAgent  : expert d'un domaine, callable (query -> réponse + confiance).
* DevAdvocate   : critique adverse (tente de réfuter/réduire la confiance).
* Judge         : arbitre entre experts + critiques -> verdict synthétisé.
* MoERouter     : route la requête vers le(s) bon(s) domaine(s) (Mixture of Experts).
* Orchestrator  : dispatch PARALLÈLE (ThreadPoolExecutor) à N experts, M DA, 1 juge.
  Lancable à des centaines d'agents.

HONNÊTE : les agents sont des CALLABLES Python (le cadre d'orchestration est réel :
dispatch parallèle, DA adverse, juge, MoE, quorum). Dans le prototype, les 'experts'
sont des fonctions/règles ; en production ils seraient backs par des LLM/modules
spécialisés (interface identique). On teste le CADRE, pas des LLM autonomes.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Tuple

ExpertFn = Callable[[str], Tuple[str, float]]      # (réponse, confiance 0..1)


@dataclass
class ExpertAgent:
    name: str
    domain: str
    fn: ExpertFn

    def answer(self, query: str) -> Tuple[str, float]:
        return self.fn(query)


@dataclass
class DevAdvocate:
    """Devil's Advocate : tente de réfuter une réponse -> critique + facteur de doute."""
    name: str
    fn: Callable[[str, str], Tuple[str, float]]      # (réponse d'expert, query) -> (critique, doubt 0..1)

    def critique(self, answer: str, query: str) -> Tuple[str, float]:
        return self.fn(answer, query)


@dataclass
class Judge:
    """Arbitre : pondère experts vs DA, exige un quorum -> verdict (réponse, confiance)."""
    quorum: float = 0.6

    def arbitrate(self, experts: List[Tuple[str, float, float]],
                  critiques: List[Tuple[str, float]]) -> Tuple[Optional[str], float, str]:
        """experts: [(réponse, confiance, weight)] ; critiques: [(critique, doubt)].
        Retourne (verdict, confiance_finale, raison). None si pas de quorum.

        confiance = accord(fraction d'experts d'accord) x (1 - doute moyen des DA).
        Le doute réduit la confiance ABSOLUE (pas juste relative)."""
        import statistics
        doubt = statistics.mean([d for _, d in critiques]) if critiques else 0.0
        mass: Dict[str, float] = {}
        for ans, conf, w in experts:
            mass[ans] = mass.get(ans, 0.0) + w * conf
        total = sum(mass.values()) or 1.0
        best = max(mass, key=mass.get)
        agreement = mass[best] / total                 # fraction de masse d'accord
        conf_fin = agreement * (1.0 - doubt)           # doute réduit la confiance
        if conf_fin < self.quorum:
            return None, conf_fin, f"pas de quorum ({conf_fin:.2f} < {self.quorum}) ; accord={agreement:.2f}, doute DA={doubt:.2f}"
        return best, conf_fin, f"quorum atteint ({conf_fin:.2f}) ; accord={agreement:.2f}, {len(experts)} experts, {len(critiques)} DA"


@dataclass
class MoERouter:
    """Mixture of Experts : route la requête vers les domaines pertinents."""
    domain_keywords: Dict[str, List[str]] = field(default_factory=dict)

    def route(self, query: str, top_k: int = 3) -> List[str]:
        q = query.lower()
        scores = {d: sum(1 for kw in kws if kw in q) for d, kws in self.domain_keywords.items()}
        ranked = sorted([d for d, s in scores.items() if s > 0], key=lambda d: -scores[d])
        return ranked[:top_k] if ranked else list(self.domain_keywords.keys())[:top_k]


@dataclass
class Orchestrator:
    experts: List[ExpertAgent]
    advocates: List[DevAdvocate]
    judge: Judge = field(default_factory=Judge)
    router: Optional[MoERouter] = None
    max_workers: int = 8
    fail_open: bool = False          # False = fail-closed (aucun expert routé -> [])
    MAX_AGENTS: int = 1000           # borne anti-saturation
    MAX_WORKERS: int = 16

    def __post_init__(self):
        # RESOURCE-BOUND : cap le nombre d'agents et de workers (anti-saturation)
        if len(self.experts) > self.MAX_AGENTS:
            raise ValueError(f"trop d'experts ({len(self.experts)} > {self.MAX_AGENTS})")
        if len(self.advocates) > self.MAX_AGENTS:
            raise ValueError(f"trop de DA ({len(self.advocates)} > {self.MAX_AGENTS})")
        self.max_workers = max(1, min(self.max_workers, self.MAX_WORKERS))

    def _select_experts(self, query: str) -> List[ExpertAgent]:
        if self.router is None:
            return self.experts
        domains = self.router.route(query)
        sel = [e for e in self.experts if e.domain in domains]
        if sel or not self.fail_open:
            return sel                                # fail-closed : [] si rien routé
        return self.experts                           # fail_open=True : repli sur tous

    def run(self, query: str) -> Dict:
        """Dispatch parallèle : experts (parallèle) -> DA (parallèle) -> juge -> verdict."""
        selected = self._select_experts(query)
        with ThreadPoolExecutor(max_workers=max(1, self.max_workers)) as ex:
            exp_results = list(ex.map(lambda e: (e.name, *e.answer(query)), selected))
            # chaque réponse d'expert critiquée par tous les DA
            crit_results = []
            for _, ans, _ in exp_results:
                for da in self.advocates:
                    crit_results.append(ex.submit(da.critique, ans, query))
            crit_results = [f.result() for f in crit_results]
        experts_w = [(ans, conf, 1.0) for (_, ans, conf) in exp_results]
        verdict, conf_fin, raison = self.judge.arbitrate(experts_w, crit_results)
        return {
            "query": query, "n_experts": len(selected), "n_advocates": len(self.advocates),
            "expert_answers": exp_results, "critiques": crit_results,
            "verdict": verdict, "confidence": round(conf_fin, 4), "reason": raison,
        }
