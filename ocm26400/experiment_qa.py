#!/usr/bin/env python3
"""
EXPÉRIENCE QA de bout en bout (OCM-26400, cahier des charges §5).

Démontre la capacité « répondre à des questions » en câblant le cycle cognitif
complet sur des concepts NOMMÉS (faits en langage naturel) :

    question compositionnelle ->
      agent.solve_chain (raisonner+vérifier+apprendre) ->
        concept résultat ->
          KnowledgeBase.answer -> réponse textuelle
            OU abstention -> 'Je ne sais pas (mode apprentissage)'

C'est l'intégration user-facing de l'agent (raisononnement) + KnowledgeBase
(retrieval/faits) + abstention (P3). Honnête : les concepts/faits sont abstraits
(op arithmétique sur des ids nommés), pas un LLM dialoguant en langage libre —
mais le CYCLE (comprendre la question -> raisonner -> vérifier -> répondre ou
avouer ignorance -> apprendre) est réel et intégré.
"""
import json, random, time
import torch

from ocm26400.verifier import SymbolicDict, Verifier, P_MOD
from ocm26400.knowledge_base import KnowledgeBase
from ocm26400.cognitive_agent import CognitiveAgent
from ocm26400.experiment_composition import train_binary_block

# Noms symboliques pour rendre le QA lisible (concepts 0..P_MOD-1)
NAMES = ["zero", "un", "deux", "trois", "quatre", "cinq", "six", "sept",
         "huit", "neuf", "dix"]


def main():
    random.seed(0); torch.manual_seed(0)
    d = SymbolicDict(n=P_MOD)
    ver = Verifier(d)
    print(f"OCM-26400 QA DE BOUT EN BOUT | op=(3a+5b) mod {P_MOD} | concepts={NAMES}")

    t0 = time.time()
    blk = train_binary_block(d, ver, n_steps=1500)
    agent = CognitiveAgent(blk, d, ver)
    # KnowledgeBase : chaque concept -> un 'fait' textuel
    kb = KnowledgeBase(d, threshold=0.5)
    for i in range(P_MOD):
        kb.store(i, f"résultat = {NAMES[i]} (#{i})")
    print(f"Agent + KnowledgeBase prêts ({P_MOD} faits nommés).")

    def ask(chain):
        """Pose une question compositionnelle, retourne la réponse textuelle."""
        r, modes = agent.solve_chain(list(chain))
        if r is None:
            return "Je ne sais pas (mode apprentissage) — une étape a abstenu.", modes
        val, conf = kb.answer(d.canonical(r))
        return f"{val}  (conf {conf:.2f}, via {modes})", modes

    # scénarios QA : compositionnels, répétés (=> retrieval), et un OOD
    scenarios = [
        ("combien fait op(deux,cinq) ?", [2, 5]),
        ("combien fait op(op(trois,un),sept) ?", [3, 1, 7]),
        ("re-pose op(deux,cinq) (devrait être retrieved) ?", [2, 5]),
        ("combien fait op(op(op(un,deux),trois),quatre) ?", [1, 2, 3, 4]),
    ]
    print()
    for q, chain in scenarios:
        ans, modes = ask(chain)
        print(f"Q: {q}")
        print(f"R: {ans}\n")

    # question OOD : un concept hors vocabulaire (idx 99) -> abstention
    rOOD, modesOOD = agent.solve_chain([99, 2])
    print(f"Q: op(concept_inconnu, deux) ?")
    print(f"R: {'Je ne sais pas (OOD -> mode apprentissage)' if rOOD is None else kb.answer(d.canonical(rOOD))[0]}  (modes {modesOOD})\n")

    dt = time.time() - t0
    st = agent.stats
    print(f"--- Stats agent ({dt:.1f}s) ---")
    print(f"raisonné+appris: {st['reasoned']} | retrieved: {st['retrieved']} | abstention: {st['abstained']}")
    print(f"accuracy faits: {agent.accuracy()*100:.1f}% | mémoire: {agent.knowledge_size()} faits")
    verdict = "VALIDÉ" if (agent.accuracy() > 0.95 and st["reasoned"] > 0) else "NON VALIDÉ"
    print(f"\nVERDICT (QA de bout en bout : question->raisonner->vérifier->répondre/apprendre) : {verdict}")

    results = {
        "task": "QA de bout en bout (agent + KnowledgeBase + abstention)",
        "n_named_facts": P_MOD,
        "scenarios_run": len(scenarios) + 1,
        "agent_stats": st, "accuracy": round(agent.accuracy(), 4),
        "verdict": verdict, "duration_s": round(dt, 1),
    }
    with open("ocm26400/qa_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nRésultats: ocm26400/qa_results.json")
    return results


if __name__ == "__main__":
    main()
