#!/usr/bin/env python3
"""D MINIMUM UNIFIÉ — quel d satisfait tout P (3-27) pour toutes modalités ?

Contrainte représentationnelle : IDs one-hot vivent dans slot=d//2 → faut P ≤ d//2.
On teste d ∈ {64,128,256,512} × P ∈ {3,6,9,12,15,18,21,24,27} sur le crown-jewel
(cœur de raisonnement, = ce que tous les lobes sensoriels alimentent).
Métrique : décomposition (généralisation) + one-shot. d minimum = celui où
décomposition = 100% pour TOUT P. Vectorisé, CPU.
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

def train_binary(P, d, n_steps=1500, bs=128, lr=3e-3):
    torch.manual_seed(0)
    slot = min(d//2, 64); slot = max(slot, P)  # slot suffit pour P one-hot
    canon = make_canon(P, slot); blk = ReasonerBlock(d).to(DEVICE)
    opt = torch.optim.Adam(blk.parameters(), lr=lr)
    aa = torch.arange(P, device=DEVICE).repeat_interleave(P)
    bb = torch.arange(P, device=DEVICE).repeat(P)
    mm = op(aa, bb, P); n = P*P
    for _ in range(n_steps):
        idx = torch.randint(0, n, (bs,), device=DEVICE)
        out = blk(encode_batch(aa[idx], bb[idx], canon, d))[:, 0:slot]
        loss = (1 - F.cosine_similarity(out, canon[mm[idx]], dim=1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return blk, canon

def eval_decomp(blk, canon, P, d, a, b, c):
    blk.eval(); slot = canon.shape[1]
    with torch.no_grad():
        m = decode(blk(encode_batch(a, b, canon, d))[:, 0:slot], canon)
        r = decode(blk(encode_batch(m, c, canon, d))[:, 0:slot], canon)
        return (r == op(op(a, b, P), c, P)).float().mean().item()

def make_triples(P, n):
    g = torch.Generator(device=DEVICE).manual_seed(P*7+1)
    return (torch.randint(0,P,(n,),generator=g,device=DEVICE),
            torch.randint(0,P,(n,),generator=g,device=DEVICE),
            torch.randint(0,P,(n,),generator=g,device=DEVICE))

def main():
    print("="*64); print("D MINIMUM UNIFIÉ — sweep d × P (crown-jewel, cœur raisonnement)"); print("="*64)
    Ds = [64, 128, 256, 512]
    Ps = [3, 6, 9, 12, 15, 18, 21, 24, 27]
    test_tr = {P: make_triples(P, 512) for P in Ps}
    grid = {}  # (d,P) -> decomp_acc
    print(f"\n{'P':>4} | " + " ".join(f"d={d:>3}" for d in Ds))
    for P in Ps:
        row = []
        for d in Ds:
            if P > d//2 and P > 64:  # slot capé à 64, mais d//2 pourrait être < P
                # slot = max(d//2, P) capped 64; si P>64 impossible, mais P≤27 ok
                pass
            blk, canon = train_binary(P, d)
            acc = eval_decomp(blk, canon, P, d, *test_tr[P])
            grid[(d,P)] = acc
            row.append(acc)
        print(f"{P:>4} | " + " ".join(f"{a*100:5.1f}" for a in row), flush=True)
    print("\n" + "="*64)
    # d minimum satisfaisant tout P (decomp >= 99%)
    print("D MINIMUM qui satisfait TOUT P (décomp ≥ 99%):")
    for d in Ds:
        all_ok = all(grid[(d,P)] >= 0.99 for P in Ps)
        feas = all(P <= max(d//2, 64) for P in Ps)  # représentation possible
        print(f"  d={d}: slot={d//2}, P≤27 {'✓ représentable' if feas else '✗'}, tout P décomp≥99%: {'✓ OUI' if all_ok else '✗ NON'}")
    min_d = next((d for d in Ds if all(grid[(d,P)] >= 0.99 for P in Ps)), None)
    print(f"\n=> D MINIMUM UNIFIÉ = {min_d} (satisfait tout P 3-27, tout modalité via le cœur)")
    json.dump({f"d{d}_P{P}": grid[(d,P)] for d in Ds for P in Ps},
              open("ocm26400/crown_jewel_min_d_results.json", "w"), indent=2)
    print("[sauvé] ocm26400/crown_jewel_min_d_results.json")

if __name__ == "__main__":
    main()
