#!/usr/bin/env python3
"""
EXPÉRIENCE agent cognitif auto-apprenant (OCM-26400, cahier des charges).

Démontre l'INTÉGRATION des briques en un cycle cognitif complet :
  retrieve (mémoire) -> si inconnu, raisonner (block grokké) -> vérifier (gate) ->
  apprendre (stocker) -> désormais 'retrieved'.

Sur op(a,b)=(3a+5b) mod 11. L'agent part d'une mémoire VIDE. On stream des requêtes
(avec répétitions) et on observe :
  - la mémoire qui grossit (apprentissage).
  - la bascule 'raisonné+appris' -> 'retrieved' (les faits sujets deviennent O(1)).
  - l'accuracy des faits stockés (devrait être ~100% car block grokké).
  - l'abstention (devrait être ~0% car block exact).
"""
import json, random, time
import torch

from ocm26400.verifier import SymbolicDict, Verifier, P_MOD
from ocm26400.cognitive_agent import CognitiveAgent
from ocm26400.experiment_composition import train_binary_block


def main():
    random.seed(0); torch.manual_seed(0)
    d = SymbolicDict(n=P_MOD)
    ver = Verifier(d)
    print(f"OCM-26400 AGENT COGNITIF AUTO-APPRENANT | op=(3a+5b) mod {P_MOD} | mémoire vide au départ")

    t0 = time.time()
    blk = train_binary_block(d, ver, n_steps=1500)
    agent = CognitiveAgent(blk, d, ver)

    # stream : 800 requêtes, moitié répétées d'un pool de 40 paires (=> retrieval croît)
    pool = [(random.randrange(P_MOD), random.randrange(P_MOD)) for _ in range(40)]
    n_queries = 800
    curve = []   # (queries écoulées, mémoire, %retrieved)
    retrieved_so_far = 0
    for q in range(n_queries):
        a, b = pool[q % len(pool)] if q % 2 == 0 else (random.randrange(P_MOD), random.randrange(P_MOD))
        _, mode = agent.solve(a, b)
        if mode == "retrieved":
            retrieved_so_far += 1
        if (q + 1) % 200 == 0:
            curve.append({"queries": q + 1, "memory": agent.knowledge_size(),
                          "retrieved_pct": round(retrieved_so_far / (q + 1), 3)})
    dt = time.time() - t0

    st = agent.stats
    print(f"\n--- RÉSULTATS ({dt:.1f}s, {n_queries} requêtes) ---")
    print(f"raisonné+appris : {st['reasoned']:4d}  (nouveaux faits, O(forward)+vérif+store)")
    print(f"retrieved       : {st['retrieved']:4d}  (faits déjà connus, O(1) mémoire)")
    print(f"abstention      : {st['abstained']:4d}  (block incertain -> 'je ne sais pas')")
    print(f"mémoire finale  : {agent.knowledge_size()} faits (sur {P_MOD*P_MOD} paires possibles)")
    print(f"accuracy faits stockés : {agent.accuracy()*100:.1f}%")
    print(f"\nCourbe d'apprentissage (mémoire + part de retrieval) :")
    for c in curve:
        print(f"  {c['queries']:4d} requêtes | mémoire={c['memory']:3d} | retrieved={c['retrieved_pct']*100:4.1f}%")
    verdict = "VALIDÉ" if (agent.accuracy() > 0.95 and st["reasoned"] > 0 and retrieved_so_far > 0) else "NON VALIDÉ"
    print(f"\nVERDICT (cycle retrieve->raisonner->vérifier->apprendre) : {verdict}")

    results = {
        "task": "agent cognitif auto-apprenant (cycle retrieve/raisonner/vérifier/apprendre)",
        "n_queries": n_queries, "pool_repeated": 40,
        "stats": st, "memory_final": agent.knowledge_size(),
        "accuracy_stored_facts": round(agent.accuracy(), 4),
        "learning_curve": curve,
        "verdict": verdict, "duration_s": round(dt, 1),
    }
    with open("ocm26400/agent_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nRésultats: ocm26400/agent_results.json")
    return results


if __name__ == "__main__":
    main()
