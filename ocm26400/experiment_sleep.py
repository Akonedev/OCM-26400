#!/usr/bin/env python3
"""
EXPÉRIENCE sommeil / consolidation (OCM-26400, cahier des charges 'phases de sommeil').

Démontre la consolidation de la mémoire épisodique en mémoire sémantique :
  1. l'agent APPREND des faits épisodiques (op(a,b)=r) au fil des requêtes.
  2. SOMMEIL : on extrait la RÈGLE sous-jacente (r=αa+βb mod n) depuis les faits.
  3. la règle COMPACTE N faits -> 1 règle et GÉNÉRALISE à toutes les paires (vues ou non).

C'est 'réactivation pas réapprentissage' + 'généraliser après compréhension' + 'savings'
du cahier des charges (TESTING.md insights 1-2, E17).
"""
import json, random, time
import torch

from ocm26400.verifier import SymbolicDict, Verifier, P_MOD
from ocm26400.cognitive_agent import CognitiveAgent
from ocm26400.sleep import consolidate, consolidation_stats, rule_predicts
from ocm26400.experiment_composition import train_binary_block


def main():
    random.seed(0); torch.manual_seed(0)
    d = SymbolicDict(n=P_MOD)
    ver = Verifier(d)
    print(f"OCM-26400 SOMMEIL / CONSOLIDATION | op=(3a+5b) mod {P_MOD}")

    t0 = time.time()
    blk = train_binary_block(d, ver, n_steps=1500)
    agent = CognitiveAgent(blk, d, ver)

    # 1. apprentissage épisodique : 30 paires (sur 121 possibles)
    pairs = [(random.randrange(P_MOD), random.randrange(P_MOD)) for _ in range(30)]
    for a, b in pairs:
        agent.solve(a, b)
    n_episodic = agent.knowledge_size()
    print(f"\n1. Mémoire épisodique après apprentissage : {n_episodic} faits (sur {P_MOD*P_MOD} paires possibles)")

    # 2. SOMMEIL : extraction de règle
    rule = consolidate(agent, P_MOD)
    print(f"\n2. SOMMEIL : extraction de règle depuis les {n_episodic} faits...")
    print(f"   règle extraite : r = ({rule[0]}·a + {rule[1]}·b) mod {P_MOD}" if rule else "   aucune règle (mémoire incohérente)")

    # 3. consolidation : compression + généralisation
    stats = consolidation_stats(agent, rule, P_MOD)
    print(f"\n3. Consolidation :")
    print(f"   {stats['episodic_facts']} faits épisodiques -> {stats['compressed_to']} règle sémantique "
          f"(compression x{stats['compression_ratio']})")
    print(f"   cohérente avec les faits appris : {stats['consistent_with_learned']}")
    # vérifie la généralisation sur TOUTES les paires (dont ~91 jamais vues)
    all_ok = all(rule_predicts(rule, a, b, P_MOD) == ver.compose(a, b)
                 for a in range(P_MOD) for b in range(P_MOD))
    print(f"   GÉNÉRALISE à toutes les {P_MOD*P_MOD} paires (dont ~{P_MOD*P_MOD - n_episodic} jamais vues) : {all_ok}")

    dt = time.time() - t0
    verdict = "VALIDÉ" if (rule == (3, 5) and all_ok) else "NON VALIDÉ"
    print(f"\nVERDICT (sommeil : épisodique -> sémantique, compression + généralisation) : {verdict}")

    results = {
        "task": "sommeil / consolidation (spec 'phases de sommeil')",
        "episodic_facts_learned": n_episodic,
        "rule_extracted": list(rule) if rule else None,
        "compression_ratio": stats["compression_ratio"],
        "generalizes_to_all_pairs": all_ok,
        "consistent_with_learned": stats["consistent_with_learned"],
        "verdict": verdict, "duration_s": round(dt, 1),
    }
    with open("ocm26400/sleep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nRésultats: ocm26400/sleep_results.json")
    return results


if __name__ == "__main__":
    main()
