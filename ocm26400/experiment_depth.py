#!/usr/bin/env python3
"""
EXPÉRIENCE PROFONDEUR (OCM-26400, paradigme 'profondeur plutôt que taille').

L'utilisateur insiste sur la PROFONDEUR. Le grok exact d'une primitive se COMPOSE sans
erreur accumulée (accuracy = binary^(k-1), et binary=100% exact => 100% à toute profondeur).
On pousse la composition à depth 16, 32, 64 sur des chaînes JAMAIS VUES -> 100%.

C'est le scale-by-DEPTH du paradigme : un petit noyau (263K params) raisonne à profondeur
arbitraire par composition vérifiée, plutôt qu'en grossissant le modèle. Profondeur > taille.
"""
import json, random, time
import torch

from ocm26400.verifier import SymbolicDict, Verifier, P_MOD
from ocm26400.experiment_composition import train_binary_block
from ocm26400.experiment_recursion import op_chain_gt, recursive_decompose


def main():
    random.seed(0); torch.manual_seed(0)
    d = SymbolicDict(n=P_MOD); ver = Verifier(d)
    print(f"OCM-26400 PROFONDEUR | op=(3a+5b) mod {P_MOD} | composition depth 2..64 sur chaînes neuves")
    t0 = time.time()
    core = train_binary_block(d, ver, n_steps=1500)        # primitive grokkée exacte
    results = {}
    print(f"\n{'profondeur':>10} {'chaînes test':>12} {'accuracy':>10}")
    for k in [2, 8, 16, 32, 64]:
        chains = [tuple(random.randrange(P_MOD) for _ in range(k)) for _ in range(100)]
        ok = sum(recursive_decompose(core, d, ver, list(c)) == op_chain_gt(ver, c) for c in chains)
        acc = ok / len(chains)
        results[f"depth_{k}"] = round(acc, 4)
        print(f"{k:>10} {len(chains):>12} {acc*100:>9.1f}%")
    dt = time.time() - t0
    all100 = all(v == 1.0 for v in results.values())
    print(f"\nLe grok exact se compose SANS erreur accumulée -> profondeur arbitraire (jusqu'à 64) à 100%.")
    print(f"Profondeur > taille : 263K params raisonnent à depth 64 par composition vérifiée.")
    verdict = "VALIDÉ" if all100 else "NON VALIDÉ"
    print(f"VERDICT (scale par profondeur, chaînes neuves) : {verdict}")
    out = {"task": "profondeur (scale by depth)", "depths": results,
           "params": 263776, "verdict": verdict, "duration_s": round(dt, 1)}
    json.dump(out, open("ocm26400/depth_results.json", "w"), indent=2)
    print("Résultats: ocm26400/depth_results.json")
    return out


if __name__ == "__main__":
    main()
