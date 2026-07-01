#!/usr/bin/env python3
"""COMPLÉTION DE FORMULE — D = k^3.5 · d^-3.55 · T^2.06  (scale = anti-grok).

La vérification précédente (verify_scale_antigrok.py) mesurait le per_step de la PRIMITIVE
→ montrait d AIDER (b=+1.55), réfutant l'anti-grok au niveau primitive.

MAIS le Besoins dit "scale=anti-grok" pour la COMPOSITION one-shot ("élargir cassait le grok, 0.24").
La BONNE grandeur pour la formule = D, la PROFONDEUR COMPOSITIONNELLE gate-certifiée
(combien de cascade l'étape certifiée permet avant que la gate chute).

Ici on mesure D_max(d, T) = profondeur max où la gate (alignement) reste ≥ τ, pour une
primitive grokkée T steps à dim d (imperfectible → la composition dégrade à un certain D).
Fit log(D_max) vs log(d), log(T) → exposants complets de la formule.

Q1 : D_max vs d (anti-grok compositionnel ? exposant prédit -3.55)
Q2 : D_max vs T (exposant prédit +2.06)
Q3 : one-shot composition vs d (le grok one-shot se casse-t-il quand d augmente ?)
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, json, time
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
P = 11  # 2*P=22 ≤ d_min=64 pour l'encodage concat ; 3*P=33 ≤ 64 pour le one-shot


def op(a, b): return (a + b) % P


class ReasonerBlock(nn.Module):
    def __init__(self, d, h=None):
        super().__init__(); h = h or max(d*2, 128)
        self.norm = nn.LayerNorm(d); self.f1 = nn.Linear(d, h); self.f2 = nn.Linear(h, d)
        nn.init.normal_(self.f1.weight, std=0.02); nn.init.normal_(self.f2.weight, std=0.02)
        nn.init.zeros_(self.f1.bias); nn.init.zeros_(self.f2.bias)
    def forward(self, x):
        h = self.norm(x); h = torch.relu(self.f1(h)); h = self.f2(h); return x + h


def canon():
    c = torch.zeros(P, P, device=DEVICE); c[torch.arange(P), torch.arange(P)] = 1.0; return c


def encode(a, b, d):
    x = torch.zeros(len(a), d, device=DEVICE); c = canon()
    x[:, 0:P] = c[a]; x[:, P:2*P] = c[b]; return x


def grok_primitive(d, T, bs=256, lr=3e-3):
    torch.manual_seed(0)
    blk = ReasonerBlock(d).to(DEVICE); opt = torch.optim.Adam(blk.parameters(), lr=lr)
    aa = torch.arange(P, device=DEVICE).repeat_interleave(P)
    bb = torch.arange(P, device=DEVICE).repeat(P); mm = op(aa, bb); c = canon()
    for _ in range(T):
        idx = torch.randint(0, P*P, (bs,))
        out = blk(encode(aa[idx], bb[idx], d))[:, 0:P]
        loss = (1 - F.cosine_similarity(out, c[mm[idx]], dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return blk


def D_max(blk, d, depths=(2, 4, 8, 16, 32, 64), tau=0.99, n=1500):
    """Profondeur max où la gate (alignement au canonique le + proche) reste ≥ τ sur la cascade."""
    c = canon(); blk.eval()
    last_good = 1
    for D in depths:
        g = torch.Generator(device=DEVICE).manual_seed(D*7+1)
        ops = torch.randint(0, P, (n, D+1), generator=g, device=DEVICE)
        true = ops.sum(1) % P
        with torch.no_grad():
            v = ops[:, 0]; gmin = torch.ones(n, device=DEVICE)
            for t in range(D):
                out = blk(encode(v, ops[:, t+1], d))[:, 0:P]
                gs = F.cosine_similarity(out, c[(out @ c.t()).argmax(1)], dim=-1)
                v = (out @ c.t()).argmax(1); gmin = torch.minimum(gmin, gs)
            acc = (v == true).float().mean().item(); gmean = gmin.mean().item()
        if gmean >= tau and acc > 0.9:
            last_good = D
        else:
            break
    return last_good


def oneshot_compose(d, T=1000, n_test=2000):
    """One-shot : entraîne op(a,b,c)→r DIRECTEMENT (pas de scratchpad). Test sur triples non-vus.
    Le Besoins dit : one-shot à grand d = anti-grok (0.24). Vérifions."""
    torch.manual_seed(0)
    blk = ReasonerBlock(d).to(DEVICE); opt = torch.optim.Adam(blk.parameters(), lr=3e-3)
    c = canon()
    g = torch.Generator(device=DEVICE).manual_seed(0)
    a = torch.randint(0, P, (3000,), generator=g, device=DEVICE)
    b = torch.randint(0, P, (3000,), generator=g, device=DEVICE)
    cc = torch.randint(0, P, (3000,), generator=g, device=DEVICE)
    r = op(op(a, b), cc)  # one-shot target
    for _ in range(T):
        idx = torch.randint(0, 3000, (256,))
        # encode (a,b,c) : ent=a, prop=b, op-slot=c
        x = torch.zeros(256, d, device=DEVICE)
        x[:, 0:P] = c[a[idx]]; x[:, P:2*P] = c[b[idx]]
        if 3*P <= d: x[:, 2*P:3*P] = c[cc[idx]]
        out = blk(x)[:, 0:P]
        loss = (1 - F.cosine_similarity(out, c[r[idx]], dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    # test triples non-vus
    blk.eval(); g2 = torch.Generator(device=DEVICE).manual_seed(99)
    ta = torch.randint(0, P, (n_test,), generator=g2, device=DEVICE)
    tb = torch.randint(0, P, (n_test,), generator=g2, device=DEVICE)
    tc = torch.randint(0, P, (n_test,), generator=g2, device=DEVICE)
    with torch.no_grad():
        x = torch.zeros(n_test, d, device=DEVICE)
        x[:, 0:P] = c[ta]; x[:, P:2*P] = c[tb]
        if 3*P <= d: x[:, 2*P:3*P] = c[tc]
        pred = (blk(x)[:, 0:P] @ c.t()).argmax(1)
    return (pred == op(op(ta, tb), tc)).float().mean().item()


def main():
    print("="*64); print("COMPLÉTION FORMULE — D_max (profondeur compositionnelle gate-certifiée) vs d, T"); print("="*64)
    rows = []; t0 = time.time()
    # Q1+Q2 : D_max(d, T)
    print("\n  D_max(d, T) [profondeur où gate ≥ 0.99 sur la cascade] :")
    for d in [64, 128, 256, 512]:
        for T in [40, 80, 160, 320]:
            blk = grok_primitive(d, T); dm = D_max(blk, d)
            rows.append({"d": d, "T": T, "D_max": dm})
            print(f"    d={d:>4} T={T:>4} → D_max={dm}", flush=True)
    # fit log D_max = a + b·log d + c·log T (sur les D_max > 1)
    fr = [r for r in rows if r["D_max"] > 1]
    if len(fr) >= 3:
        A = np.array([[1, np.log(r["d"]), np.log(max(r["T"],1))] for r in fr])
        y = np.log([r["D_max"] for r in fr])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None); a, b, c = coef
    else:
        a = b = c = float('nan')

    # Q3 : one-shot composition vs d (anti-grok ?)
    print("\n  Q3 one-shot composition vs d (anti-grok ?) :")
    os_rows = []
    for d in [64, 128, 256, 512]:
        acc = oneshot_compose(d); os_rows.append({"d": d, "oneshot_acc": acc})
        print(f"    d={d:>4} → one-shot compose acc={acc*100:5.1f}%", flush=True)

    print("\n" + "="*64); print("VERDICT FORMULE :")
    print(f"  Fit D_max = a·d^b · T^c :  b(d)={b:.2f} (prédit -3.55) | c(T)={c:.2f} (prédit +2.06) | a={a:.2f}")
    print(f"  One-shot compose vs d : {'anti-grok CONFIRMÉ' if os_rows[-1]['oneshot_acc'] < os_rows[0]['oneshot_acc']-0.1 else 'pas clair'}")
    print(f"    d=32: {os_rows[0]['oneshot_acc']*100:.0f}% → d=512: {os_rows[-1]['oneshot_acc']*100:.0f}%")
    json.dump({"Dmax_rows": rows, "fit": {"a": a, "b_d": b, "c_T": c}, "oneshot_rows": os_rows},
              open("ocm26400/complete_formula_results.json", "w"), indent=2, default=str)
    print(f"[sauvé] t={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
