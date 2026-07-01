#!/usr/bin/env python3
"""CONFIRMATION MULTI-SEED du sommeil neural (5 seeds, mean±std).

Valide le résultat single-run (sommeil +33pt vs baseline) sur plusieurs seeds
pour exclure un coup de chance. Rapporte eveil / sommeil / baseline en mean±std.
"""
import torch, torch.nn as nn, torch.nn.functional as F, json, numpy as np
from ocm26400.test_sleep_neural import Model, rule, gen, spectral_filter, replay, acc, V, K, L, D
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVEIL, REPLAY = 1500, 800


def fresh(seed):
    torch.manual_seed(seed); m = Model().to(DEVICE); return m, torch.optim.Adam(m.parameters(), lr=3e-3)


def one_run(seed):
    tr, te = gen(400, 1).to(DEVICE), gen(400, 2).to(DEVICE)
    lab_tr, lab_te = rule(tr), rule(te)
    # eveil
    m, opt = fresh(seed); replay(m, opt, tr, lab_tr, EVEIL); e = acc(m, te, lab_te)
    # sommeil (depuis meme etat eveil)
    ms, opts = fresh(seed); replay(ms, opts, tr, lab_tr, EVEIL)
    spectral_filter(ms, 0.5, 'low'); replay(ms, opts, tr, lab_tr, 200)
    spectral_filter(ms, 0.3, 'high'); replay(ms, opts, tr, lab_tr, 400)
    replay(ms, opts, tr, lab_tr, 200); s = acc(ms, te, lab_te)
    # baseline (+REPLAY steps)
    mb, optb = fresh(seed); replay(mb, optb, tr, lab_tr, EVEIL); replay(mb, optb, tr, lab_tr, REPLAY); b = acc(mb, te, lab_te)
    return e, s, b


def main():
    print("="*64); print("CONFIRMATION SOMMEIL NEURAL — 5 seeds (mean±std)"); print("="*64)
    E, S, B = [], [], []
    for seed in [0, 1, 2, 3, 4]:
        e, s, b = one_run(seed)
        E.append(e); S.append(s); B.append(b)
        print(f"  seed {seed}: eveil={e*100:5.1f}%  sommeil={s*100:5.1f}%  baseline={b*100:5.1f}%  Δ(sommeil-base)={ (s-b)*100:+5.1f}pt", flush=True)
    print("\n" + "="*64)
    print(f"  ÉVEIL     : {np.mean(E)*100:5.1f}% ± {np.std(E)*100:.1f}")
    print(f"  SOMMEIL   : {np.mean(S)*100:5.1f}% ± {np.std(S)*100:.1f}")
    print(f"  BASELINE  : {np.mean(B)*100:5.1f}% ± {np.std(B)*100:.1f}")
    delta = np.mean(S) - np.mean(B)
    print(f"\n  Δ(sommeil − baseline) = {delta*100:+.1f}pt")
    if delta > 0.05:
        print("  => SOMMEIL AIDE, confirmé sur 5 seeds ✓. Mémoire→compréhension NEURAL robuste.")
    elif delta > 0:
        print("  => Sommeil aide légèrement, mais effet faible/non robuste.")
    else:
        print("  => Sommeil n'aide pas (le single-run était un coup de chance).")
    json.dump({"eveil": E, "sommeil": S, "baseline": B,
               "mean": {"eveil": float(np.mean(E)), "sommeil": float(np.mean(S)), "baseline": float(np.mean(B))}},
              open("ocm26400/test_sleep_neural_seeds_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    main()
