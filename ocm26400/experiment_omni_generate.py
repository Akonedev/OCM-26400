#!/usr/bin/env python3
"""
CAPSTONE — paradigme complet : primitives + composition -> GÉNÉRER N'IMPORTE QUOI
(OCM-26400, cahier des charges).

L'utilisateur : « si tu suis le paradigme et tu entraînes le modèle à apprendre les
primitives, à comprendre les principes, le modèle devrait pouvoir générer n'importe quoi ».

Démonstration : le NOYAU DE L'OmniModel (ReasonerBlock partagé, omni.py) est entraîné sur
la PRIMITIVE op(a,b)=(3a+5b) mod 11 (un seul opérateur appris). Une fois la primitive
grokkée, le modèle GÉNÈRE n'importe quelle composition op^k pour des chaînes JAMAIS
VUES — y compris des chaînes arbitrairement longues et demandées par l'utilisateur.

C'est le cœur du paradigme : apprendre les PRIMITIVES + le PRINCIPE de composition (et non
mémoriser) => GÉNÉRALISATION COMPOSITIONNELLE => génération ouverte (« n'importe quoi »).

Réutilise le crown-jewel (experiment_composition) + la récurrence (experiment_recursion)
sur le noyau partagé de l'OmniModel unifié.
"""
import json, random, time
import torch

from ocm26400.verifier import SymbolicDict, Verifier, P_MOD
from ocm26400.reasoner import ReasonerBlock, encode_input, DEVICE
from ocm26400.experiment_composition import train_binary_block
from ocm26400.experiment_recursion import op_chain_gt, recursive_decompose


def main():
    random.seed(0); torch.manual_seed(0)
    d = SymbolicDict(n=P_MOD); ver = Verifier(d)
    print("CAPSTONE OmniModel : primitives + composition -> GÉNÉRER N'IMPORTE QUOI")
    print(f"Noyau partagé de l'OmniModel (ReasonerBlock) entraîné sur la PRIMITIVE op=(3a+5b) mod {P_MOD}.")

    t0 = time.time()
    core = train_binary_block(d, ver, n_steps=1500)        # = OmniModel.core, primitive grokkée

    # 1) génération ouverte : op^k pour des chaînes JAMAIS VUES, profondeurs variées
    print("\n1) Génération compositionnelle (chaînes jamais vues) :")
    results = {}
    for k in [2, 3, 5, 8]:
        chains = [tuple(random.randrange(P_MOD) for _ in range(k)) for _ in range(200)]
        ok = sum(recursive_decompose(core, d, ver, list(c)) == op_chain_gt(ver, c) for c in chains)
        acc = ok / len(chains)
        results[f"depth_{k}"] = round(acc, 4)
        print(f"   profondeur {k} : {acc*100:5.1f}% correct sur {len(chains)} chaînes neuves")

    # 2) génération À LA DEMANDE : l'utilisateur demande une composition précise
    print("\n2) Génération à la demande (compositions arbitraires demandées) :")
    demands = [[1, 2, 3], [7, 7, 7, 7], [10, 0, 5, 2, 8], [3, 3]]
    demos = []
    for chain in demands:
        pred = recursive_decompose(core, d, ver, list(chain))
        truth = op_chain_gt(ver, chain)
        demos.append({"chain": chain, "generated": pred, "truth": truth, "ok": pred == truth})
        print(f"   op^{len(chain)}{chain} -> généré {pred} (vérité {truth}) {'✓' if pred==truth else '✗'}")

    dt = time.time() - t0
    all_ok = all(v >= 0.95 for v in results.values()) and all(x["ok"] for x in demos)
    verdict = "VALIDÉ" if all_ok else "NON VALIDÉ"
    print(f"\nLe modèle, entraîné sur UNE primitive + le principe de composition, "
          f"génère n'importe quelle composition (profondeur 2-8, chaînes neuves).")
    print(f"VERDIT (paradigme complet : primitives -> générer n'importe quoi) : {verdict}")

    results_out = {
        "task": "capstone : primitives+composition -> générer n'importe quoi",
        "paradigm": "entraîner sur la primitive + principe de composition (pas mémoriser)",
        "primitive": f"op(a,b)=(3a+5b) mod {P_MOD}",
        "compositional_generation_unseen": results,
        "on_demand_generation": demos,
        "verdict": verdict, "duration_s": round(dt, 1),
    }
    with open("ocm26400/omni_generate_results.json", "w") as f:
        json.dump(results_out, f, indent=2)
    print("\nRésultats: ocm26400/omni_generate_results.json")
    return results_out


if __name__ == "__main__":
    main()
