#!/usr/bin/env python3
"""TEST SOMMEIL NEURAL — le sommeil aide-t-il un réseau à généraliser (mémoire→compréhension) ?

Le sommeil symbolique (sleep_phases.py) extrait la règle exacte trivialement. Ici on teste
le VRAI effet sur un RÉSEAU : après surapprentissage (mémoire : train haut, test bas),
les phases de sommeil (filtrage spectral des poids + replay) améliorent-ils le test
AU-DELÀ du simple +de steps ?

Phases (Besoins, chaque phase descend dans le spectre) :
  - LÉGER (rêve/NREM-1) : FFT low-pass des poids → débruit la mémorisation (macro).
  - MOYEN (NREM-2)      : replay pondéré entropie → consolide (compresse doublons).
  - PROFOND (NREM-3)    : FFT high-pass → affine détails (micro), substitués.

Comparaison stricte : sommeil (3 phases, budget S steps de replay) vs baseline (+S steps purs).
Si sommeil > baseline → le sommeil fait qqch que le simple entraînement ne fait pas.
"""
import torch, torch.nn as nn, torch.nn.functional as F, json
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
V = 10; K = 5; L = 12; D = 128


def rule(seq):  # règle positionnelle généralisable : class=(seq[3]+seq[7]) mod K
    return (seq[:, 3] + seq[:, 7]) % K


def gen(n, seed):
    return torch.randint(0, V, (n, L), generator=torch.Generator().manual_seed(seed))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.mlp = nn.Sequential(nn.Linear(L*D, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, K))
    def forward(self, seq):
        return self.mlp(self.embed(seq).flatten(1))


def spectral_filter(model, keep_frac, mode):
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                W = p.data.float(); Fr = torch.fft.rfft(W, dim=0)
                k = max(1, int(Fr.shape[0] * keep_frac))
                if mode == 'low': Fr[k:] = 0
                else:             Fr[:k] = 0
                p.data = torch.fft.irfft(Fr, n=W.shape[0], dim=0).to(p.dtype)


def replay(model, opt, seq, lab, steps):
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(seq), (64,))
        loss = F.cross_entropy(model(seq[idx]), lab[idx])
        opt.zero_grad(); loss.backward(); opt.step()


def acc(model, seq, lab):
    model.eval()
    with torch.no_grad():
        return (model(seq).argmax(1) == lab).float().mean().item()


def fresh():
    torch.manual_seed(0); m = Model().to(DEVICE); return m, torch.optim.Adam(m.parameters(), lr=3e-3)


def main():
    print("="*64); print("TEST SOMMEIL NEURAL — mémoire → compréhension ?"); print("="*64)
    print(f"  tâche : class=(seq[3]+seq[7]) mod {K} (règle généralisable), MLP sur 400 train/400 test\n")
    tr, te = gen(400, 1).to(DEVICE), gen(400, 2).to(DEVICE)
    lab_tr, lab_te = rule(tr), rule(te)
    EVEIL = 1500; REPLAY = 800  # budgets

    # ÉVEIL commun
    m, opt = fresh(); replay(m, opt, tr, lab_tr, EVEIL)
    e_tr, e_te = acc(m, tr, lab_tr), acc(m, te, lab_te)
    print(f"  ÉVEIL ({EVEIL} steps) : train={e_tr*100:5.1f}%  test={e_te*100:5.1f}%", flush=True)

    # Bras SOMMEIL (3 phases, depuis le même état éveil)
    ms, opts = fresh(); replay(ms, opts, tr, lab_tr, EVEIL)
    spectral_filter(ms, 0.5, 'low');  replay(ms, opts, tr, lab_tr, 200); p1 = acc(ms, te, lab_te)
    spectral_filter(ms, 0.3, 'high'); replay(ms, opts, tr, lab_tr, 400); p2 = acc(ms, te, lab_te)
    replay(ms, opts, tr, lab_tr, 200); p3 = acc(ms, te, lab_te)  # moyen/consolidation
    print(f"  SOMMEIL (léger→profond→moyen, {REPLAY} replay) : test={p3*100:5.1f}%  "
          f"(léger {p1*100:.0f} → profond {p2*100:.0f} → moyen {p3*100:.0f})", flush=True)

    # Bras BASELINE (+REPLAY steps purs, sans sommeil)
    mb, optb = fresh(); replay(mb, optb, tr, lab_tr, EVEIL); replay(mb, optb, tr, lab_tr, REPLAY)
    b_te = acc(mb, te, lab_te)
    print(f"  BASELINE (+{REPLAY} steps purs)       : test={b_te*100:5.1f}%", flush=True)

    print("\n" + "="*64); print("VERDICT :")
    print(f"  éveil={e_te*100:.1f}% | sommeil={p3*100:.1f}% | baseline={b_te*100:.1f}%")
    delta = p3 - b_te
    if delta > 0.03:   print(f"  => SOMMEIL AIDE (+{delta*100:.1f}pt vs baseline). Mémoire→compréhension NEURAL confirmé ✓")
    elif delta > -0.03:print(f"  => Sommeil ≈ baseline (Δ{delta*100:+.1f}pt). Effet = simple +de steps. (Cohérent réfutation antérieure.)")
    else:              print(f"  => Sommeil NUIT ({delta*100:.1f}pt). Filtrage spectral dégrade.")
    json.dump({"eveil": e_te, "sommeil": p3, "sommeil_curve": [p1, p2, p3], "baseline": b_te},
              open("ocm26400/test_sleep_neural_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    main()
