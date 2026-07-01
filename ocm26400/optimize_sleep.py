#!/usr/bin/env python3
"""OPTIMISATION DES PHASES DE SOMMEIL + mesure grok avant/après.

4 questions :
  Q1 ABLATION : quelle phase compte ? (léger seul / profond seul / les deux / replay seul)
  Q2 INTENSITÉ : keep_frac optimal du low-pass (débruit) ?
  Q3 CYCLES : plusieurs nuits aident-elles ? (1/2/3 cycles)
  Q4 GROK AVANT/APRÈS : courbe test à chaque sous-étape → la transition de généralisation
     est-elle déclenchée PAR le sommeil (test bloqué avant, saute après) ?

Setup : class=(seq[3]+seq[7]) mod 5, MLP, EVEIL=1500 (stuck ~25%). On mesure test après chaque config.
"""
import torch, torch.nn as nn, torch.nn.functional as F, json, numpy as np
from ocm26400.test_sleep_neural import Model, rule, gen, acc, V, K, L, D
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVEIL = 1500


def spectral_filter(model, keep_frac, mode):
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                W = p.data.float(); Fr = torch.fft.rfft(W, dim=0)
                k = max(1, int(Fr.shape[0] * keep_frac))
                if mode == 'low': Fr[k:] = 0
                else:             Fr[:k] = 0
                p.data = torch.fft.irfft(Fr, n=W.shape[0], dim=0).to(p.dtype)


def replay(m, opt, seq, lab, steps):
    m.train()
    for _ in range(steps):
        idx = torch.randint(0, len(seq), (64,)); loss = F.cross_entropy(m(seq[idx]), lab[idx])
        opt.zero_grad(); loss.backward(); opt.step()


def fresh(seed):
    torch.manual_seed(seed); m = Model().to(DEVICE); return m, torch.optim.Adam(m.parameters(), lr=3e-3)


def eveil_state(seed, tr, lab_tr):
    m, opt = fresh(seed); replay(m, opt, tr, lab_tr, EVEIL); return m, opt


def run_sleep_config(seed, config, tr, lab_tr, te, lab_te, trace=False):
    """config = dict(phases=[('low',kf,rsteps),('high',kf,rsteps),...], cycles=n).
    Retourne test final (et courbe si trace)."""
    m, opt = eveil_state(seed, tr, lab_tr)
    curve = [("eveil", acc(m, te, lab_te))]
    for cyc in range(config["cycles"]):
        for mode, kf, rsteps in config["phases"]:
            spectral_filter(m, kf, mode); replay(m, opt, tr, lab_tr, rsteps)
            if trace: curve.append((f"c{cyc}/{mode}{kf}", acc(m, te, lab_te)))
    final = acc(m, te, lab_te)
    return (final, curve) if trace else final


def main():
    print("="*64); print("OPTIMISATION SOMMEIL — ablation + intensité + cycles + grok avant/après"); print("="*64)
    tr, te = gen(400, 1).to(DEVICE), gen(400, 2).to(DEVICE)
    lab_tr, lab_te = rule(tr), rule(te)
    results = {}

    # baseline eveil (pour référence)
    e = acc(eveil_state(0, tr, lab_tr)[0], te, lab_te)
    print(f"\n  ÉVEIL (avant sommeil) : test={e*100:.1f}%  (= mémorisation, pas de grok)\n", flush=True)

    # ---- Q1 ABLATION : quelle phase ? ----
    print("--- Q1 ABLATION (quelle phase compte, 200 replay chacune) ---")
    kf_l, kf_h, rs = 0.5, 0.3, 200
    ablations = {
        "replay_seul (pas de filtre)":  {"phases": [("low", 1.0, rs)], "cycles": 1},   # keep_frac=1.0 = pas de filtre
        "léger seul (low-pass)":        {"phases": [("low", kf_l, rs*2)], "cycles": 1},
        "profond seul (high-pass)":     {"phases": [("high", kf_h, rs*2)], "cycles": 1},
        "léger+profond (actuel)":       {"phases": [("low", kf_l, rs), ("high", kf_h, rs)], "cycles": 1},
    }
    abl_res = {}
    for name, cfg in ablations.items():
        t = run_sleep_config(0, cfg, tr, lab_tr, te, lab_te)
        abl_res[name] = t; print(f"  {name:30s}: test={t*100:5.1f}%  (Δ vs eveil {(t-e)*100:+.1f}pt)", flush=True)
    results["Q1_ablation"] = abl_res

    # ---- Q2 INTENSITÉ : keep_frac du low-pass ----
    print("\n--- Q2 INTENSITÉ low-pass (keep_frac, léger+profond, 1 cycle) ---")
    int_res = {}
    for kf in [0.2, 0.3, 0.5, 0.7, 0.9]:
        cfg = {"phases": [("low", kf, rs), ("high", kf_h, rs)], "cycles": 1}
        t = run_sleep_config(0, cfg, tr, lab_tr, te, lab_te)
        int_res[kf] = t; print(f"  keep_frac_low={kf}: test={t*100:5.1f}%", flush=True)
    results["Q2_intensite"] = int_res
    best_kf = max(int_res, key=int_res.get)

    # ---- Q3 CYCLES : 1/2/3 nuits ----
    print(f"\n--- Q3 CYCLES (léger+profond, kf_low={best_kf}, 1/2/3 cycles) ---")
    cyc_res = {}
    for ncyc in [1, 2, 3, 5]:
        cfg = {"phases": [("low", best_kf, rs), ("high", kf_h, rs)], "cycles": ncyc}
        t = run_sleep_config(0, cfg, tr, lab_tr, te, lab_te)
        cyc_res[ncyc] = t; print(f"  {ncyc} cycle(s): test={t*100:5.1f}%", flush=True)
    results["Q3_cycles"] = cyc_res
    best_cyc = max(cyc_res, key=cyc_res.get)

    # ---- Q4 GROK AVANT/APRÈS : courbe (config optimale) ----
    print(f"\n--- Q4 GROK AVANT/APRÈS (courbe test, kf_low={best_kf}, {best_cyc} cycles) ---")
    cfg = {"phases": [("low", best_kf, rs), ("high", kf_h, rs)], "cycles": best_cyc}
    _, curve = run_sleep_config(0, cfg, tr, lab_tr, te, lab_te, trace=True)
    for name, t in curve:
        bar = "█" * int(t*30); print(f"  {name:22s}: {t*100:5.1f}% {bar}", flush=True)
    results["Q4_courbe"] = curve

    # ---- Confirmation 3 seeds de la config optimale ----
    print(f"\n--- CONFIRMATION 3 seeds (config optimale: kf_low={best_kf}, {best_cyc} cycles) ---")
    ts = [run_sleep_config(s, cfg, tr, lab_tr, te, lab_te) for s in [0,1,2]]
    print(f"  optimal: {np.mean(ts)*100:.1f}% ± {np.std(ts)*100:.1f}  (seeds {[f'{x*100:.0f}' for x in ts]})", flush=True)
    results["optimal_3seeds"] = {"mean": float(np.mean(ts)), "std": float(np.std(ts)), "raw": ts}

    print("\n" + "="*64); print("VERDICT :")
    best_phase = max(abl_res, key=abl_res.get)
    print(f"  Phase clé      : {best_phase} ({abl_res[best_phase]*100:.0f}%)")
    print(f"  keep_frac opt  : {best_kf}")
    print(f"  cycles opt     : {best_cyc}  (test {cyc_res[best_cyc]*100:.0f}%)")
    print(f"  optimal 3-seed : {np.mean(ts)*100:.1f}% ± {np.std(ts)*100:.1f}  (eveil {e*100:.0f}%, Δ {(np.mean(ts)-e)*100:+.0f}pt)")
    json.dump(results, open("ocm26400/optimize_sleep_results.json", "w"), indent=2, default=str)
    print("[sauvé]")


if __name__ == "__main__":
    main()
