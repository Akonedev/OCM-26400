#!/usr/bin/env python3
"""PROFONDEUR DE RAISONNEMENT — sonde le terme d^-2.38 de la loi.

La décomposition binaire (2-step) sature à 100% pour tout d. Pour sonner le D de la loi
(D = profondeur max fiable ∝ d^-2.38), on teste des CHAÎNES k-step:
  r = op(op(...op(a0,a1),a2)...,ak)  → k étapes de raisonnement.

Hypothèse (loi) : d=64 soutient +d'étapes que d=256/512 (D_64 > D_256 > D_512).
On mesure l'accuracy par étape puis la chaîne : où chaque d casse-t-il (<90%) ?

Sur CPU (libre de toute contention GPU).
"""
import torch, torch.nn as nn, torch.nn.functional as F, time, json
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
A_COEF, B_COEF = 3, 5
def op(a, b, P): return (A_COEF * a + B_COEF * b) % P

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
    bs = a.shape[0]; slot = canon.shape[1]
    x = torch.zeros(bs, d, device=DEVICE)
    x[:, 0:slot] = canon[a]; x[:, slot:2*slot] = canon[b]; return x

def decode(ent, canon):
    return (ent @ canon.t()).argmax(1)

def train_binary(P, d, n_steps=2000, bs=128, lr=3e-3):
    torch.manual_seed(0); slot = d//2; canon = make_canon(P, slot)
    blk = ReasonerBlock(d).to(DEVICE); opt = torch.optim.Adam(blk.parameters(), lr=lr)
    aa = torch.arange(P, device=DEVICE).repeat_interleave(P)
    bb = torch.arange(P, device=DEVICE).repeat(P)
    mm = op(aa, bb, P); n = P*P
    for _ in range(n_steps):
        idx = torch.randint(0, n, (bs,), device=DEVICE)
        out = blk(encode_batch(aa[idx], bb[idx], canon, d))[:, 0:slot]
        tgt = canon[mm[idx]]
        loss = (1 - F.cosine_similarity(out, tgt, dim=1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return blk, canon

def eval_chain(blk, canon, P, d, chain_len, n_test=512):
    """Chaîne k-step VECTORISÉE sur n_test chaînes parallèles."""
    blk.eval(); slot = canon.shape[1]
    g = torch.Generator(device=DEVICE).manual_seed(chain_len*100+7)
    chains = torch.randint(0, P, (n_test, chain_len+1), generator=g, device=DEVICE)
    m = chains[:, 0].clone()
    with torch.no_grad():
        for i in range(1, chain_len+1):
            out = blk(encode_batch(m, chains[:, i], canon, d))[:, 0:slot]
            m = decode(out, canon)         # propage l'erreur étape par étape
        true = chains[:, 0].clone()
        for i in range(1, chain_len+1):
            true = op(true, chains[:, i], P)
    return (m == true).float().mean().item()

def main():
    print("="*64); print("PROFONDEUR DE RAISONNEMENT — sonde d^-2.38 (chaînes k-step)"); print("="*64)
    P = 24
    results = {}
    for d in [64, 256, 512]:
        t0 = time.time()
        blk, canon = train_binary(P, d, n_steps=2000)
        # accuracy par étape (binaire)
        per_step = eval_chain(blk, canon, P, d, chain_len=1, n_test=512)
        chain_accs = {}
        for k in [1, 2, 5, 10, 20, 50]:
            acc = eval_chain(blk, canon, P, d, chain_len=k, n_test=512)
            chain_accs[k] = acc
        results[d] = {"per_step": per_step, "chains": chain_accs}
        # trouver point de rupture (<90%)
        break_k = next((k for k in [1,2,5,10,20,50] if chain_accs[k] < 0.9), ">6")
        bar = " ".join(f"{chain_accs[k]*100:4.0f}" for k in [1,2,5,10,20,50])
        print(f"d={d:>4} | per-step={per_step*100:5.1f}% | chaînes k=1..6: [{bar}] | rupture(<90%): {break_k} | t={time.time()-t0:.0f}s", flush=True)
    print("\n" + "="*64); print("VÉRIFICATION DE LA LOI (D ∝ d^-2.38 → d petit soutient +d'étapes):")
    print("  Prédiction: rupture(d=64) > rupture(d=256) > rupture(d=512)")
    for d in [64, 256, 512]:
        bk = next((k for k in [1,2,5,10,20,50] if results[d]["chains"][k] < 0.9), ">6")
        print(f"  d={d}: rupture à k={bk}  (per-step={results[d]['per_step']*100:.1f}%)")
    json.dump({str(d): {"per_step": r["per_step"], "chains": r["chains"]} for d, r in results.items()},
              open("ocm26400/crown_jewel_depth_probe_results.json", "w"), indent=2)
    print("\n[sauvé] ocm26400/crown_jewel_depth_probe_results.json")

if __name__ == "__main__":
    main()
