#!/usr/bin/env python3
"""VÉRIFIER D = k^3.5 · d^-3.55 · T^2.06  (scale = anti-grok).

Thèse utilisateur : le grok est l'INVERSE du scale.
  - d^-3.55 : augmenter la dimension du modèle DÉTRUIT le grok (anti-grok).
  - T^2.06  : plus de steps améliore le grok (super-linéaire).
  - k^3.5   : effet scratchpad/récurrence (k constant ici, on isole d et T).

Mesure : D = profondeur de raisonnement max = 1/(1 - per_step)  (loi L3 du Besoins).
  per_step = alignement gate moyen (cosinus vers canonique correct) sur toutes les paires,
  à la fin de l'entraînement (primitive crown-jewel op(a,b) mod P, 1-cos).

Sweep d × T → table D(d,T) + fit log-log :
  log D = a + b·log d + c·log T.   Prédit : b ≈ -3.55, c ≈ +2.06.
"""
import torch, torch.nn as nn, torch.nn.functional as F, json, time, numpy as np
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
P = 11; A_COEF, B_COEF = 3, 5


def op(a, b): return (A_COEF * a + B_COEF * b) % P


class ReasonerBlock(nn.Module):
    def __init__(self, d, h=None):
        super().__init__(); h = h or max(d*2, 128)
        self.norm = nn.LayerNorm(d); self.f1 = nn.Linear(d, h); self.f2 = nn.Linear(h, d)
        nn.init.normal_(self.f1.weight, std=0.02); nn.init.normal_(self.f2.weight, std=0.02)
        nn.init.zeros_(self.f1.bias); nn.init.zeros_(self.f2.bias)
    def forward(self, x):
        h = self.norm(x); h = torch.relu(self.f1(h)); h = self.f2(h); return x + h


def make_canon(P):
    c = torch.zeros(P, P, device=DEVICE); c[torch.arange(P), torch.arange(P)] = 1.0; return c


def measure_D(d, T, bs=256, lr=3e-3):
    """Grok op(a,b) à dim d pendant T steps → per_step (gate) → D = 1/(1-per_step)."""
    torch.manual_seed(0)
    canon = make_canon(P)
    blk = ReasonerBlock(d).to(DEVICE); opt = torch.optim.Adam(blk.parameters(), lr=lr)
    all_a = torch.arange(P, device=DEVICE).repeat_interleave(P)
    all_b = torch.arange(P, device=DEVICE).repeat(P); all_m = op(all_a, all_b)
    slot = P
    for _ in range(T):
        idx = torch.randint(0, P*P, (bs,))
        x = torch.zeros(bs, d, device=DEVICE)
        x[:, 0:slot] = canon[all_a[idx]]; x[:, slot:2*slot] = canon[all_b[idx]]
        out = blk(x)[:, 0:slot]
        loss = (1 - F.cosine_similarity(out, canon[all_m[idx]], dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    # per_step = gate (cosinus vers canonique correct) moyen sur toutes les paires
    blk.eval()
    with torch.no_grad():
        x = torch.zeros(P*P, d, device=DEVICE)
        x[:, 0:slot] = canon[all_a]; x[:, slot:2*slot] = canon[all_b]
        out = blk(x)[:, 0:slot]
        per_step = F.cosine_similarity(out, canon[all_m], dim=-1).mean().item()
        acc = (out @ canon.t()).argmax(1).eq(all_m).float().mean().item()
    D = 1.0 / (1.0 - per_step)  # loi L3 (pas de clip : on veut le vrai per_step < 1)
    return per_step, D, acc


def main():
    print("="*64); print("VÉRIFICATION D = k^3.5 · d^-3.55 · T^2.06  (scale=anti-grok)"); print("="*64)
    print(f"  primitive op(a,b)=(3a+5b) mod {P}, per_step=gate(cos), D=1/(1-per_step)")
    print(f"  T FAIBLE (primitive imparfaite) pour faire varier per_step < 1 et mesurer l'exposant\n")
    ds = [32, 64, 128, 256]
    Ts = [40, 80, 160, 320, 640]
    rows = []
    t0 = time.time()
    for d in ds:
        for T in Ts:
            per_step, D, acc = measure_D(d, T)
            rows.append({"d": d, "T": T, "per_step": per_step, "D": D, "acc": acc})
            print(f"  d={d:>4} T={T:>5} | per_step={per_step:.5f} acc={acc*100:5.1f}% | D={D:9.1f}", flush=True)
    # fit log-log : log D = a + b·log d + c·log T
    A = np.array([[1, np.log(r["d"]), np.log(r["T"])] for r in rows])
    y = np.log([r["D"] for r in rows])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b, c = coef
    print("\n" + "="*64); print("FIT log D = a + b·log d + c·log T :")
    print(f"  exposant d : b = {b:.2f}   (prédit -3.55 ; négatif = anti-grok confirmé)")
    print(f"  exposant T : c = {c:.2f}   (prédit +2.06 ; positif = +de steps aide)")
    print(f"  const      : a = {a:.2f}   (= log k^3.5 → k)")
    verdict_d = "CONFIRME (détruit le grok)" if b < -1 else "INFIRMÉ"
    verdict_T = "CONFIRME (+steps aide)" if c > 0.5 else "faible"
    print(f"\n  d: scale=anti-grok {verdict_d}  |  T: {verdict_T}")
    json.dump({"rows": rows, "fit": {"a": a, "b_d": b, "c_T": c},
               "predicted": {"b_d": -3.55, "c_T": 2.06}}, open("ocm26400/verify_scale_antigrok_results.json", "w"), indent=2)
    print(f"[sauvé] t={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
