#!/usr/bin/env python3
"""CROWN-JEWEL VECTORISÉ — sweep d ∈ {64,256} × P ∈ {12,24} pour vérifier la loi.

Loi: D = k^1.98 × P^1.06 × d^-2.38. Ici on teste les termes P (modulus) et d (dimension)
directement sur le crown-jewel arithmétique : op(a,b)=(3a+5b) mod P, r=op(op(a,b),c).

TOUT VECTORISÉ (batch cosine, plus de boucle Python par-sample) → ~10× plus rapide.
Pour chaque config (d,P) : décomposition (m=a∘b puis r=m∘c) vs one-shot (a,b,c→r).
"""
import torch, torch.nn as nn, torch.nn.functional as F, time, json, itertools
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
A_COEF, B_COEF = 3, 5


def op(a, b, P):  # vectorisé sur tenseurs
    return (A_COEF * a + B_COEF * b) % P


class ReasonerBlock(nn.Module):
    def __init__(self, d, hidden=None):
        super().__init__()
        hidden = hidden or max(d * 2, 128)
        self.norm = nn.LayerNorm(d); self.fc1 = nn.Linear(d, hidden); self.fc2 = nn.Linear(hidden, d)
        nn.init.normal_(self.fc1.weight, std=0.02); nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias); nn.init.zeros_(self.fc2.bias)
    def forward(self, x):
        h = self.norm(x); h = torch.relu(self.fc1(h)); h = self.fc2(h); return x + h


def make_canon(P, slot):
    """one-hot canoniques (P, slot) — vit dans ent_slot dims."""
    c = torch.zeros(P, slot); c[torch.arange(P), torch.arange(P)] = 1.0
    return c.to(DEVICE)


def encode_batch(a, b, canon, d):
    """AMV (bs, d): ent=canon(a) | prop=canon(b) | reste zéro. VECTORISÉ."""
    bs = a.shape[0]; slot = canon.shape[1]
    x = torch.zeros(bs, d, device=DEVICE)
    x[:, 0:slot] = canon[a]            # ent
    x[:, slot:2*slot] = canon[b]       # prop
    return x


def train_binary(P, d, n_steps=1500, bs=128, lr=3e-3):
    """Entraîne ReasonerBlock sur op(a,b)->m. VECTORISÉ (batch cosine)."""
    torch.manual_seed(SEED)
    slot = d // 2
    canon = make_canon(P, slot)
    blk = ReasonerBlock(d).to(DEVICE)
    opt = torch.optim.Adam(blk.parameters(), lr=lr)
    # tous les P² couples pré-calculés
    all_a = torch.arange(P, device=DEVICE).repeat_interleave(P)
    all_b = torch.arange(P, device=DEVICE).repeat(P)
    all_m = op(all_a, all_b, P)                       # (P²,)
    n_pairs = P * P
    for _ in range(n_steps):
        idx = torch.randint(0, n_pairs, (bs,), device=DEVICE)
        x = encode_batch(all_a[idx], all_b[idx], canon, d)
        out = blk(x)
        ent = out[:, 0:slot]
        tgt = canon[all_m[idx]]                       # (bs, slot)
        cos = F.cosine_similarity(ent, tgt, dim=1)    # (bs,) VECTORISÉ
        loss = (1 - cos).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return blk, canon


def decode(ent, canon, P):
    """argmax cosine sur les P canoniques. VECTORISÉ (bs,) -> idx."""
    sims = ent @ canon.t()                            # (bs, P)
    return sims.argmax(1)


def eval_decomp(blk, canon, P, d, triples):
    """Décomposition: m=op(a,b), r=op(m,c). VECTORISÉ."""
    blk.eval(); slot = canon.shape[1]
    a, b, c = triples
    with torch.no_grad():
        x1 = encode_batch(a, b, canon, d)
        m_ent = blk(x1)[:, 0:slot]
        m_pred = decode(m_ent, canon, P)
        x2 = encode_batch(m_pred, c, canon, d)
        r_ent = blk(x2)[:, 0:slot]
        r_pred = decode(r_ent, canon, P)
        r_true = op(op(a, b, P), c, P)
        return (r_pred == r_true).float().mean().item()


def train_oneshot(P, d, train_tr, n_steps=1500, bs=128, lr=3e-3):
    """One-shot: (a,b,c)->r directement. VECTORISÉ."""
    torch.manual_seed(SEED)
    slot = d // 2; canon = make_canon(P, slot)
    blk = ReasonerBlock(d).to(DEVICE); opt = torch.optim.Adam(blk.parameters(), lr=lr)
    a_t, b_t, c_t = train_tr
    r_t = op(op(a_t, b_t, P), c_t, P); n = a_t.shape[0]
    for _ in range(n_steps):
        idx = torch.randint(0, n, (bs,), device=DEVICE)
        # encode (a,b,c): ent=a, prop=b, op-slot=c
        x = torch.zeros(bs, d, device=DEVICE)
        x[:, 0:slot] = canon[a_t[idx]]; x[:, slot:2*slot] = canon[b_t[idx]]
        if 3*slot <= d: x[:, 2*slot:3*slot] = canon[c_t[idx]]
        out = blk(x)[:, 0:slot]
        tgt = canon[r_t[idx]]
        loss = (1 - F.cosine_similarity(out, tgt, dim=1)).mean()   # VECTORISÉ
        opt.zero_grad(); loss.backward(); opt.step()
    return blk, canon


def eval_oneshot(blk, canon, P, d, triples):
    blk.eval(); slot = canon.shape[1]; a, b, c = triples
    with torch.no_grad():
        bs = a.shape[0]; x = torch.zeros(bs, d, device=DEVICE)
        x[:, 0:slot] = canon[a]; x[:, slot:2*slot] = canon[b]
        if 3*slot <= d: x[:, 2*slot:3*slot] = canon[c]
        out = blk(x)[:, 0:slot]
        r_pred = decode(out, canon, P)
        r_true = op(op(a, b, P), c, P)
        return (r_pred == r_true).float().mean().item()


def make_triples(P, n_train, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    a = torch.randint(0, P, (n_train,), generator=g, device=DEVICE)
    b = torch.randint(0, P, (n_train,), generator=g, device=DEVICE)
    c = torch.randint(0, P, (n_train,), generator=g, device=DEVICE)
    return a, b, c


def make_test_triples(P, n_test, train_set, seed=1):
    """triples JAMAIS vus (différents du train)."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    ta, tb, tc = train_set
    train_keys = set(zip(ta.tolist(), tb.tolist(), tc.tolist()))
    a, b, c = [], [], []
    while len(a) < n_test:
        ca = torch.randint(0, P, (1,), generator=g).item()
        cb = torch.randint(0, P, (1,), generator=g).item()
        cc = torch.randint(0, P, (1,), generator=g).item()
        if (ca, cb, cc) not in train_keys:
            a.append(ca); b.append(cb); c.append(cc)
    return torch.tensor(a, device=DEVICE), torch.tensor(b, device=DEVICE), torch.tensor(c, device=DEVICE)


def run_config(d, P, n_train=400, n_test=400):
    t0 = time.time()
    train_tr = make_triples(P, n_train)
    test_tr = make_test_triples(P, n_test, train_tr)
    blk, canon = train_binary(P, d, n_steps=1500)
    decomp_acc = eval_decomp(blk, canon, P, d, test_tr)
    blk_os, canon_os = train_oneshot(P, d, train_tr, n_steps=1500)
    os_acc = eval_oneshot(blk_os, canon_os, P, d, test_tr)
    nparams = sum(p.numel() for p in blk.parameters())
    return {"d": d, "P": P, "params": nparams, "decomp_acc": decomp_acc,
            "oneshot_acc": os_acc, "gap": decomp_acc - os_acc, "t": time.time()-t0}


def main():
    print("="*64); print("CROWN-JEWEL VECTORISÉ — sweep d×P (vérif loi D=k^1.98·P^1.06·d^-2.38)"); print("="*64)
    results = []
    for d, P in itertools.product([64, 256], [12, 24]):
        r = run_config(d, P)
        results.append(r)
        print(f"d={r['d']:>4} P={r['P']:>3} | params={r['params']:>7,} | DÉCOMP={r['decomp_acc']*100:5.1f}% | "
              f"one-shot={r['oneshot_acc']*100:5.1f}% | gap={r['gap']*100:+5.1f}pt | t={r['t']:.0f}s", flush=True)
    print("\n" + "="*64); print("VÉRIFICATION DE LA LOI (D ∝ P^1.06 · d^-2.38, à k constant):")
    # ratio prédit vs ratio observé (decomp_acc comme proxy de D)
    base = results[0]
    print(f"  base: d={base['d']} P={base['P']} decomp={base['decomp_acc']*100:.1f}%")
    for r in results[1:]:
        d_ratio = (r['d']/base['d'])**-2.38
        P_ratio = (r['P']/base['P'])**1.06
        D_pred = d_ratio * P_ratio
        D_obs = r['decomp_acc'] / max(base['decomp_acc'], 1e-6)
        print(f"  d={r['d']} P={r['P']}: D_prédit={D_pred:.3f} | D_observé(decomp)={D_obs:.3f}")
    json.dump(results, open("ocm26400/crown_jewel_sweep_results.json", "w"), indent=2)
    print("\n[sauvé] ocm26400/crown_jewel_sweep_results.json")


if __name__ == "__main__":
    main()
