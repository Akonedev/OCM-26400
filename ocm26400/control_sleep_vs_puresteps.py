#!/usr/bin/env python3
"""CONTRÔLE D'HONNÊTETÉ — le sommeil est-il spécial, ou = juste +de steps ?

Le 5-cycle sleep fait 2000 steps de replay AU TOTAL (+filtres). Si 2000 steps de PUR
entraînement (sans filtre) grokkent aussi → sommeil = juste accélération du grokking
retardé (pas spécial). Si pur reste stuck → le filtre spectral est ESSENTIEL.

Comparaison stricte même budget total de steps, 3 seeds :
  - eveil 1500 (référence stuck)
  - pur +2000 steps (sans filtre) = 3500 total
  - sommeil 5 cycles (1500 eveil + 2000 replay AVEC filtres)
"""
import torch, torch.nn.functional as F, numpy as np, json
from ocm26400.test_sleep_neural import Model, rule, gen, acc, V, K, L, D
from ocm26400.optimize_sleep import spectral_filter, replay, fresh, EVEIL
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sleep_5cycles(seed, tr, lab_tr, te, lab_te):
    m, opt = fresh(seed); replay(m, opt, tr, lab_tr, EVEIL)
    for _ in range(5):
        spectral_filter(m, 0.5, 'low'); replay(m, opt, tr, lab_tr, 200)
        spectral_filter(m, 0.3, 'high'); replay(m, opt, tr, lab_tr, 200)
    return acc(m, te, lab_te)


def pure_steps(seed, tr, lab_tr, te, lab_te, extra=2000):
    m, opt = fresh(seed); replay(m, opt, tr, lab_tr, EVEIL + extra)  # 3500 total, AUCUN filtre
    return acc(m, te, lab_te)


def main():
    print("="*64); print("CONTRÔLE : sommeil(5 cycles, 2000 replay+FILTRES) vs pur(+2000 steps)"); print("="*64)
    tr, te = gen(400, 1).to(DEVICE), gen(400, 2).to(DEVICE)
    lab_tr, lab_te = rule(tr), rule(te)
    sl, pu = [], []
    for s in [0, 1, 2]:
        sv = sleep_5cycles(s, tr, lab_tr, te, lab_te); sl.append(sv)
        pv = pure_steps(s, tr, lab_tr, te, lab_te); pu.append(pv)
        print(f"  seed {s}: sommeil(5c+F)={sv*100:5.1f}%  pur(+2000, noF)={pv*100:5.1f}%  Δ={ (sv-pv)*100:+5.1f}pt", flush=True)
    print("\n" + "="*64)
    print(f"  SOMMEIL (filtres) : {np.mean(sl)*100:.1f}% ± {np.std(sl)*100:.1f}")
    print(f"  PUR (+2000 steps) : {np.mean(pu)*100:.1f}% ± {np.std(pu)*100:.1f}")
    delta = np.mean(sl) - np.mean(pu)
    print(f"\n  Δ(sommeil − pur) = {delta*100:+.1f}pt")
    if delta > 0.1:
        print("  => SOMMEIL SPÉCIAL ✓ : le filtre spectral fait ce que +de steps ne peut pas.")
        print("     Le grok via sommeil n'est PAS du grokking retardé classique — c'est le filtrage.")
    elif delta > 0:
        print("  => Sommeil légèrement meilleur, mais le pur grok aussi → sommeil = accélérateur.")
    else:
        print("  => Sommeil = juste +de steps (le pur grok autant). Pas spécial.")
    json.dump({"sommeil": sl, "pur_2000": pu, "delta": float(delta)}, open("ocm26400/control_sleep_vs_pure_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    main()
