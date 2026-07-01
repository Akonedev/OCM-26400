#!/usr/bin/env python3
"""GROKKING HELD-OUT (canonique) — la recette crown-jewel + VRAI hold-out.

Literature (Gromov, Liu Omnigrok, Power 2022, Exploring Grokking 2412.10898) :
  - Un MLP SIMPLE peut grokker l'arithmétique modulaire (saut de gén. sur held-out).
  - Grok mod-arith = circuits de FOURIER → notre SCB (FFT) est naturellement prédisposé.
  - Conditions : large init + petit wd, assez de steps (transition retardée).

Bugs corrigés vs crown_jewel_grokking_canonical.py :
  - BUG 1 : embed(a)+embed(b) (somme) → collisions → plafond 76.7% train.
            FIX : one-hot CONCATENÉ [canon(a)||canon(b)] (sans collision, comme crown-jewel vectorisé).
  - BUG 2 : recette crown-jewel (prouvée memorize 100%) jamais testée AVEC hold-out.
            FIX : on ajoute un vrai split 60/40 + détection de phase transition.

Tests (op(a,b)=(3a+5b) mod 11, 1-cos, ReasonerBlock) :
  T1 PURE GROK : train 60% des paires, held-out 40%. Le held-out saute-t-il (grok) ?
  T2 DECOMP    : train op(a,b)->m sur TOUT, test op(op(a,b),c)->r sur TRIPLES non-vus.
  T3 LARGE-INIT: variante T1 avec init large (condition "Exploring Grokking").
Vectorisé, CPU/GPU.
"""
import torch, torch.nn as nn, torch.nn.functional as F, json, time
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
P = 11; D = 64; A_COEF, B_COEF = 3, 5


def op(a, b): return (A_COEF * a + B_COEF * b) % P


class ReasonerBlock(nn.Module):
    """Bloc crown-jewel (LayerNorm + FFN résiduel). Init contrôlable (large si want)."""
    def __init__(self, d=D, h=128, large_init=False):
        super().__init__()
        self.norm = nn.LayerNorm(d); self.f1 = nn.Linear(d, h); self.f2 = nn.Linear(h, d)
        std = 0.5 if large_init else 0.02  # ponytail: 0.5 = "large" per Exploring Grokking
        nn.init.normal_(self.f1.weight, std=std); nn.init.normal_(self.f2.weight, std=std)
        nn.init.zeros_(self.f1.bias); nn.init.zeros_(self.f2.bias)
    def forward(self, x):
        h = self.norm(x); h = torch.relu(self.f1(h)); h = self.f2(h); return x + h


def make_canon():
    """one-hot canoniques (P, P) — chaque ID dans son propre slot, orthogonal."""
    c = torch.zeros(P, P, device=DEVICE)
    c[torch.arange(P), torch.arange(P)] = 1.0
    return c


def encode(a, b):
    """AMV sans collision : ent=canon(a) | prop=canon(b). Chaque (a,b) = pattern unique."""
    canon = make_canon(); bs = a.shape[0]
    x = torch.zeros(bs, D, device=DEVICE)
    x[:, 0:P] = canon[a]      # ent slot
    x[:, P:2*P] = canon[b]    # prop slot
    return x


def decode(ent):
    """argmax cosine vs canoniques → ID prédit. VECTORISÉ."""
    canon = make_canon()
    return (ent @ canon.t()).argmax(1)


def run_pure(n_steps=50000, bs=128, lr=3e-3, large_init=False, label="T1"):
    """T1 PURE GROK : train 60% paires, held-out 40%. one-hot-concat + 1-cos."""
    torch.manual_seed(0)
    all_a = torch.arange(P, device=DEVICE).repeat_interleave(P)
    all_b = torch.arange(P, device=DEVICE).repeat(P)
    all_m = op(all_a, all_b)
    perm = torch.randperm(P*P, device=DEVICE); n_tr = int(P*P*0.6)
    tr, te = perm[:n_tr], perm[n_tr:]
    blk = ReasonerBlock(large_init=large_init).to(DEVICE)
    opt = torch.optim.Adam(blk.parameters(), lr=lr)
    canon = make_canon(); t0 = time.time(); grok = None; prev = 0.0; best = 0.0
    for step in range(n_steps+1):
        idx = tr[torch.randint(0, len(tr), (bs,))]
        out = blk(encode(all_a[idx], all_b[idx]))[:, 0:P]
        tgt = canon[all_m[idx]]
        loss = (1 - F.cosine_similarity(out, tgt, dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2500 == 0 or step == n_steps:
            blk.eval()
            with torch.no_grad():
                tr_acc = (decode(blk(encode(all_a[tr], all_b[tr]))[:, 0:P]) == all_m[tr]).float().mean().item()
                te_acc = (decode(blk(encode(all_a[te], all_b[te]))[:, 0:P]) == all_m[te]).float().mean().item()
            blk.train(); best = max(best, te_acc)
            if grok is None and te_acc > 0.5 and prev < 0.5: grok = step
            prev = te_acc
            print(f"  [{label}] step {step:>6} train={tr_acc*100:5.1f}% held={te_acc*100:5.1f}%{'  *** GROK ***' if grok==step else ''} t={time.time()-t0:.0f}s", flush=True)
    return best, grok


def run_decomp(n_steps=2000, bs=128, lr=3e-3):
    """T2 DECOMP : train op(a,b)->m sur TOUT, test op(op(a,b),c)->r sur triples non-vus."""
    torch.manual_seed(0)
    blk = ReasonerBlock().to(DEVICE); opt = torch.optim.Adam(blk.parameters(), lr=lr)
    canon = make_canon(); t0 = time.time()
    # train sur TOUTES les paires (mémorise la primitive 1-pas)
    all_a = torch.arange(P, device=DEVICE).repeat_interleave(P)
    all_b = torch.arange(P, device=DEVICE).repeat(P); all_m = op(all_a, all_b)
    for step in range(n_steps+1):
        idx = torch.randint(0, P*P, (bs,))
        out = blk(encode(all_a[idx], all_b[idx]))[:, 0:P]
        loss = (1 - F.cosine_similarity(out, canon[all_m[idx]], dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    # test composition sur triples non-vus
    blk.eval()
    g = torch.Generator(device=DEVICE).manual_seed(42)
    a = torch.randint(0, P, (1000,), generator=g, device=DEVICE)
    b = torch.randint(0, P, (1000,), generator=g, device=DEVICE)
    c = torch.randint(0, P, (1000,), generator=g, device=DEVICE)
    with torch.no_grad():
        m_pred = decode(blk(encode(a, b))[:, 0:P])          # étape 1
        r_pred = decode(blk(encode(m_pred, c))[:, 0:P])     # étape 2 (cascade)
    r_true = op(op(a, b), c)
    decomp_acc = (r_pred == r_true).float().mean().item()
    print(f"  [T2] decomp sur 1000 triples non-vus = {decomp_acc*100:5.1f}%  t={time.time()-t0:.0f}s", flush=True)
    return decomp_acc


def main():
    print("="*64); print("GROKKING HELD-OUT — crown-jewel + vrai hold-out (bugs corrigés)"); print("="*64)
    print(f"  op(a,b)=(3a+5b) mod {P}, one-hot-concat + 1-cos + ReasonerBlock(d={D})\n")
    results = {}
    print("--- T1 PURE GROK (60% train, 40% held-out) ---")
    a1, g1 = run_pure(label="T1"); results["T1_pure"] = {"held": a1, "grok": g1}
    print("\n--- T3 LARGE-INIT (même chose, init std=0.5) ---")
    a3, g3 = run_pure(large_init=True, label="T3"); results["T3_large_init"] = {"held": a3, "grok": g3}
    print("\n--- T2 DECOMP (primitive mémorisée, composition sur triples non-vus) ---")
    a2 = run_decomp(); results["T2_decomp"] = {"held": a2}

    print("\n" + "="*64); print("VERDICT :")
    for k, v in results.items():
        if "grok" in v:
            gs = f"grok@{v['grok']}" if v["grok"] else "no grok"
            print(f"  {k:16s}: held-out={v['held']*100:5.1f}%  {gs}")
        else:
            print(f"  {k:16s}: held-out(triples)={v['held']*100:5.1f}%")
    if a1 > 0.9 or a3 > 0.9:
        best = "T1" if a1 > a3 else "T3"
        print(f"\n  => GROKKING ATTEINT ({best}) ✓. Le SCB extrapole la règle aux paires NON-VUES.")
        print("     Pont crown-jewel → comprehension NUMÉRIQUE ÉTABLI.")
    else:
        print(f"\n  => Pas de grok pur (max {max(a1,a3)*100:.0f}%). Décomp={a2*100:.0f}%.")
        print("     Le crown-jewel fait COMPOSITION (décomp) mais pas extrapolation pure.")
    json.dump(results, open("ocm26400/grokking_heldout_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    main()
