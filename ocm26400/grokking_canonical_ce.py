#!/usr/bin/env python3
"""GROKKING CANONIQUE CE — le setup exact Gromov/Power, ENFIN sans le bug de collision.

Setup canonique du grokking MLP (Gromov, Liu Omnigrok, Power 2022) :
  - Loss CE (pas 1-cos — le 1-cos mémorise, n'exerce pas de pression vers la gén.)
  - AdamW + weight_decay ≠ 0 (levier clé vers circuits low-norm généralisants)
  - Embeddings DENSES APPRIS, CONCATÉNÉS [e(a)||e(b)] (pas de collision)
  - 2-layer MLP, 60/40 hold-out, transition de phase retardée

Bug précédent corrigé : crown_jewel_grokking_canonical.py faisait embed(a)+embed(b)
(SOMME → collisions → plafond 76.7% train). Ici on CONCATÈNE → info jointe préservée.

Sweep wd ∈ {0, 1e-3, 1e-2, 1e-1}, op(a,b)=(3a+5b) mod 11, 100k steps, hold-out 40%.
"""
import torch, torch.nn as nn, torch.nn.functional as F, json, time
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
P = 11; D = 128; H = 512; A_COEF, B_COEF = 3, 5


def op(a, b): return (A_COEF * a + B_COEF * b) % P


class GrokMLP(nn.Module):
    """2-layer MLP, embeddings denses CONCATÉNÉS (collision-free)."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(P, D)
        self.mlp = nn.Sequential(nn.Linear(2*D, H), nn.GELU(), nn.Linear(H, D), nn.GELU())
        self.head = nn.Linear(D, P)
    def forward(self, a, b):
        x = torch.cat([self.embed(a), self.embed(b)], dim=-1)  # CONCAT (pas somme)
        return self.head(self.mlp(x))


def run(wd, n_steps=100000, bs=128, lr=1e-3):
    torch.manual_seed(0)
    all_a = torch.arange(P, device=DEVICE).repeat_interleave(P)
    all_b = torch.arange(P, device=DEVICE).repeat(P); all_m = op(all_a, all_b)
    perm = torch.randperm(P*P, device=DEVICE); n_tr = int(P*P*0.6)
    tr, te = perm[:n_tr], perm[n_tr:]
    m = GrokMLP().to(DEVICE); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    t0 = time.time(); grok = None; prev = 0.0; best = 0.0
    for step in range(n_steps+1):
        idx = tr[torch.randint(0, len(tr), (bs,))]
        loss = F.cross_entropy(m(all_a[idx], all_b[idx]), all_m[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 5000 == 0 or step == n_steps:
            m.eval()
            with torch.no_grad():
                tr_acc = (m(all_a[tr], all_b[tr]).argmax(1) == all_m[tr]).float().mean().item()
                te_acc = (m(all_a[te], all_b[te]).argmax(1) == all_m[te]).float().mean().item()
            m.train(); best = max(best, te_acc)
            if grok is None and te_acc > 0.5 and prev < 0.5: grok = step
            prev = te_acc
            flag = "  *** GROK ***" if grok == step else ""
            print(f"  [wd={wd}] step {step:>6} train={tr_acc*100:5.1f}% held={te_acc*100:5.1f}%{flag} t={time.time()-t0:.0f}s", flush=True)
    return best, grok


def main():
    print("="*64); print("GROKKING CANONIQUE CE — Gromov/Power setup (collision bug corrigé)"); print("="*64)
    print(f"  op(a,b)=(3a+5b) mod {P}, MLP 2-layer d={D} h={H}, embeddings CONCATÉNÉS, CE+AdamW\n")
    results = {}
    for wd in [0.0, 1e-3, 1e-2, 1e-1]:
        print(f"--- wd={wd} ---")
        best, grok = run(wd)
        results[f"wd_{wd}"] = {"held": best, "grok_step": grok}
    print("\n" + "="*64); print("VERDICT :")
    for k, v in results.items():
        gs = f"grok@{v['grok_step']}" if v["grok_step"] else "no grok"
        print(f"  {k:10s}: held-out={v['held']*100:5.1f}%  {gs}")
    best_wd = max(results.values(), key=lambda v: v["held"])
    if best_wd["held"] > 0.9:
        print(f"\n  => GROKKING CE atteint ✓ (held-out {best_wd['held']*100:.0f}%).")
        print("     Le MLP grok l'arithmétique modulaire quand le wd pousse vers low-norm.")
        print("     Pont : CE+wd = grok pur, 1-cos = composition. Deux mécanismes complémentaires.")
    else:
        print(f"\n  => Pas de grok pur même avec CE+wd (max {best_wd['held']*100:.0f}%).")
        print("     Conclusion : le SCB/crown-jewel fait COMPOSITION (100%), pas extrapolation.")
        print("     La 'comprehension' = primitives mémorisées + cascade, PAS règle apprise.")
    json.dump(results, open("ocm26400/grokking_canonical_ce_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    main()
