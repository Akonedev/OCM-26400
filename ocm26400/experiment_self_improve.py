#!/usr/bin/env python3
"""
EXPÉRIENCE auto-correction / auto-amélioration (OCM-26400, demande utilisateur).

Le modèle se corrige lui-même : on SIMULE une phase d'apprentissage imparfaite en
seedant la mémoire avec ~30% de faits ERRONÉS, puis on laisse l'agent S'AUTO-CORRIGER
(re-raisonner chaque fait, corriger les conflits). L'auto-amélioration = la justesse
qui monte vers 100% au fil des passes.

Le block grokké (fiable) est le self-check : il re-dérive la bonne réponse et rattrape
les erreurs de mémorisation. Détection INTERNE (re-raisonnement), pas vérité externe.
"""
import json, random, time
import torch

from ocm26400.verifier import SymbolicDict, Verifier, P_MOD
from ocm26400.cognitive_agent import CognitiveAgent
from ocm26400.self_correction import self_correct, self_improve, self_consistency_confidence
from ocm26400.experiment_composition import train_binary_block

FRAC_WRONG = 0.3


def main():
    random.seed(0); torch.manual_seed(0)
    d = SymbolicDict(n=P_MOD); ver = Verifier(d)
    print(f"OCM-26400 AUTO-CORRECTION / AUTO-AMÉLIORATION | op=(3a+5b) mod {P_MOD}")

    t0 = time.time()
    blk = train_binary_block(d, ver, n_steps=1500)        # block grokké (self-check fiable)
    agent = CognitiveAgent(blk, d, ver)

    # simule apprentissage imparfait : mémoire avec FRAC_WRONG de faits erronés
    for a in range(P_MOD):
        for b in range(P_MOD):
            r_true = ver.compose(a, b)
            r = (r_true + (1 if random.random() < FRAC_WRONG else 0)) % P_MOD
            agent.memory[(a, b)] = r

    n_facts = len(agent.memory)
    ok = sum(1 for (a, b), r in agent.memory.items() if r == ver.compose(a, b))
    acc0 = ok / n_facts
    print(f"\nMémoire après 'apprentissage imparfait' : {n_facts} faits, justesse {acc0*100:.1f}% "
          f"(~{FRAC_WRONG*100:.0f}% d'erreurs injectées)")

    # auto-correction
    stats1 = self_correct(agent, ver)
    print(f"\n1re auto-correction : {stats1['corrected']} faits corrigés sur {stats1['checked']} "
          f"-> justesse {stats1['acc_before']*100:.1f}% -> {stats1['acc_after']*100:.1f}%")

    # auto-amélioration (courbe)
    # on ré-injecte des erreurs puis on relance l'amélioration pour montrer la courbe
    for a in range(P_MOD):
        for b in range(P_MOD):
            r_true = ver.compose(a, b)
            r = (r_true + (1 if random.random() < FRAC_WRONG else 0)) % P_MOD
            agent.memory[(a, b)] = r
    curve = self_improve(agent, ver, rounds=5)
    print(f"\nCourbe d'auto-amélioration :")
    for i, s in enumerate(curve):
        print(f"  round {i+1} : corrigés={s['corrected']:3d}  justesse {s['acc_before']*100:5.1f}% -> {s['acc_after']*100:5.1f}%")

    # self-consistency sur quelques faits
    confs = [self_consistency_confidence(agent, a, b, k=5, noise_std=0.3)
             for a in range(P_MOD) for b in [0]]
    print(f"\nSelf-consistency moyenne (k=5, bruit 0.3) : {sum(confs)/len(confs):.2f} (1.0 = certain)")

    dt = time.time() - t0
    final_acc = curve[-1]["acc_after"]
    verdict = "VALIDÉ" if final_acc == 1.0 else "NON VALIDÉ"
    print(f"\nJustesse finale après auto-amélioration : {final_acc*100:.1f}%")
    print(f"VERDICT (auto-correction via re-raisonnement, auto-amélioration -> 100%) : {verdict}")

    results = {
        "task": "auto-correction / auto-amélioration (self-consistency)",
        "n_facts": n_facts, "frac_wrong_injected": FRAC_WRONG,
        "acc_before": round(acc0, 4),
        "first_pass": {"corrected": stats1["corrected"], "acc_after": round(stats1["acc_after"], 4)},
        "improvement_curve": [{"corrected": s["corrected"], "acc_after": round(s["acc_after"], 4)} for s in curve],
        "mean_self_consistency": round(sum(confs) / len(confs), 4),
        "final_acc": round(final_acc, 4), "verdict": verdict, "duration_s": round(dt, 1),
    }
    with open("ocm26400/self_improve_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nRésultats: ocm26400/self_improve_results.json")
    return results


if __name__ == "__main__":
    main()
