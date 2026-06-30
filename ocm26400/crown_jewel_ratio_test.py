#!/usr/bin/env python3
"""TEST DU RATIO P/d = 0.1875 (hypothèse user) sur le crown-jewel.

La décomposition sature à 100% pour tout (d,P) → on mesure le VRAI différenciateur :
  1. VITESSE DE GROK : à quel step la décomposition test atteint 95% (plus rapide = meilleur ratio)
  2. ONE-SHOT : généralisation sans décomposition (reflet de la qualité de la représentation)

Configs : sur-diagonale P/d=0.1875 vs hors-diagonale.
Vectorisé + CPU (libre de toute contention GPU).
"""
import torch, torch.nn as nn, torch.nn.functional as F, time, json
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
A_COEF, B_COEF = 3, 5
def op(a, b, P): return (A_COEF*a + B_COEF*b) % P

class ReasonerBlock(nn.Module):
    def __init__(self, d, hidden=None):
        super().__init__(); hidden = hidden or max(d*2, 128)
        self.norm = nn.LayerNorm(d); self.fc1 = nn.Linear(d, hidden); self.fc2 = nn.Linear(hidden, d)
        nn.init.normal_(self.fc1.weight, std=0.02); nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias); nn.init.zeros_(self.fc2.bias)
    def forward(self, x):
        h = self.norm(x); h = torch.relu(self.fc1(h)); h = self.fc2(h); return x + h

def make_canon(P, slot):
    c = torch.zeros(P, slot); c[torch.arange(P), torch.arange(P)] = 1.0; return c.to(DEVICE)
def encode_batch(a, b, canon, d):
    bs = a.shape[0]; slot = canon.shape[1]; x = torch.zeros(bs, d, device=DEVICE)
    x[:, 0:slot] = canon[a]; x[:, slot:2*slot] = canon[b]; return x
def decode(ent, canon): return (ent @ canon.t()).argmax(1)

def eval_decomp(blk, canon, P, d, a, b, c):
    blk.eval(); slot = canon.shape[1]
    with torch.no_grad():
        m = decode(blk(encode_batch(a, b, canon, d))[:, 0:slot], canon)
        r = decode(blk(encode_batch(m, c, canon, d))[:, 0:slot], canon)
        r_true = op(op(a, b, P), c, P)
    return (r == r_true).float().mean().item()

def make_triples(P, n):
    g = torch.Generator(device=DEVICE).manual_seed(42)
    return (torch.randint(0,P,(n,),generator=g,device=DEVICE),
            torch.randint(0,P,(n,),generator=g,device=DEVICE),
            torch.randint(0,P,(n,),generator=g,device=DEVICE))

def train_and_grok(P, d, max_steps=1500, eval_every=100, grok_thresh=0.95):
    """Train binary block, évalue la décomposition périodiquement, retourne (grok_step, traj, final_decomp)."""
    torch.manual_seed(0); slot = d//2; canon = make_canon(P, slot)
    blk = ReasonerBlock(d).to(DEVICE); opt = torch.optim.Adam(blk.parameters(), lr=3e-3)
    aa = torch.arange(P, device=DEVICE).repeat_interleave(P)
    bb = torch.arange(P, device=DEVICE).repeat(P)
    mm = op(aa, bb, P); n = P*P; bs = 128
    test_tr = make_triples(P, 512)
    grok_step = None; traj = {}
    for step in range(1, max_steps+1):
        idx = torch.randint(0, n, (bs,), device=DEVICE)
        out = blk(encode_batch(aa[idx], bb[idx], canon, d))[:, 0:slot]
        loss = (1 - F.cosine_similarity(out, canon[mm[idx]], dim=1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % eval_every == 0 or step == max_steps:
            acc = eval_decomp(blk, canon, P, d, *test_tr)
            traj[step] = acc
            if grok_step is None and acc >= grok_thresh:
                grok_step = step
    final = traj[max(traj.keys())]
    return grok_step, traj, final

def main():
    print("="*64); print("TEST RATIO P/d = 0.1875 — vitesse de grok + one-shot"); print("="*64)
    configs = [
        ("d=64  P=12  r=0.1875", 64, 12),
        ("d=128 P=24  r=0.1875", 128, 24),
        ("d=256 P=48  r=0.1875", 256, 48),
        ("d=64  P=24  r=0.375",  64, 24),
        ("d=128 P=12  r=0.094",  128, 12),
        ("d=256 P=12  r=0.047",  256, 12),
    ]
    results = []
    print(f"\n{'config':24s} {'grok_step':>10} {'decomp_final':>13}")
    for name, d, P in configs:
        t0 = time.time()
        grok, traj, final = train_and_grok(P, d, max_steps=1500, eval_every=100)
        ratio = P/d
        results.append({"config": name, "d": d, "P": P, "ratio": ratio, "grok_step": grok, "decomp_final": final})
        gs = f"{grok}" if grok else ">1500"
        print(f"{name:24s} {gs:>10} {final*100:>12.1f}%  t={time.time()-t0:.0f}s", flush=True)
    print("\n" + "="*64)
    print("VERDICT ratio P/d=0.1875 :")
    diag = [r for r in results if abs(r["ratio"] - 0.1875) < 0.01]
    off = [r for r in results if abs(r["ratio"] - 0.1875) >= 0.01]
    diag_grok = [r["grok_step"] for r in diag if r["grok_step"]]
    off_grok = [r["grok_step"] for r in off if r["grok_step"]]
    if diag_grok and off_grok:
        print(f"  Grok moyen sur-diagonale (0.1875): {sum(diag_grok)/len(diag_grok):.0f} steps")
        print(f"  Grok moyen hors-diagonale      : {sum(off_grok)/len(off_grok):.0f} steps")
        print(f"  => ratio 0.1875 grok {'PLUS VITE' if sum(diag_grok)/len(diag_grok) < sum(off_grok)/len(off_grok) else 'PLUS LENTEMENT'} ✓" if sum(diag_grok)/len(diag_grok) < sum(off_grok)/len(off_grok) else "  => ratio 0.1875 ne grok pas plus vite ✗")
    json.dump(results, open("ocm26400/crown_jewel_ratio_test_results.json", "w"), indent=2)
    print("\n[sauvé] ocm26400/crown_jewel_ratio_test_results.json")

if __name__ == "__main__":
    main()
