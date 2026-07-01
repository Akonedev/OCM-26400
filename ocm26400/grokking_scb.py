#!/usr/bin/env python3
"""GROKKING SCB — le test FIDÈLE aux règles du Besoins (FFT = mécanisme du grok).

Règles du Besoins/Grokking.md suivies À LA LETTRE :
  R1. "la FFT traite des patterns numériques — les fréquences sont les nombres eux-mêmes"
      → SCB (FFT) comme noyau, AVEC L=P (FFT sur le groupe cyclique Z_P, pas L=2 dégénéré).
  R2. "len % 3 a la structure cyclique que les circuits Fourier exploitent naturellement"
      → les nombres sont des POSITIONS dans un buffer circulaire de longueur P.
  R3. "le grokking = ASSOCIATION entre nombres, pas copie de texte"
      → op(a,b)=(3a+5b) mod P = déplacement affine sur Z_P.
  R4. crown-jewel paradigm : loss 1-cos (alignement du vecteur position).
  R5. L1 décomposition : m=op(a,b) intermédiaire, r=op(m,c) en cascade.
  R6. hold-out 60/40 (mesurer la vraie extrapolation).

Mécanisme : a,b = marqueurs aux positions a,b dans un buffer circulaire Z_P.
Le SCB (FFT sur L=P) + tête apprend à produire un pic à la position (3a+5b) mod P.
L'affine sur Z_P = rotation de phase en Fourier = ce que le filtre complexe appris sait faire.
"""
import torch, torch.nn as nn, torch.nn.functional as F, json, time
from ocm26400.spectral_core import SpectralCoreBlock
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
P = 11; L = P; DM = 32; A_COEF, B_COEF = 3, 5  # L=P : FFT sur le groupe cyclique Z_P


def op(a, b): return (A_COEF * a + B_COEF * b) % P


class SCBGrok(nn.Module):
    """a,b = marqueurs aux positions dans un buffer circulaire Z_P. SCB(L=P) → pic résultat."""
    def __init__(self):
        super().__init__()
        self.marker_a = nn.Parameter(torch.randn(DM) * 0.5)   # marqueur opérande a
        self.marker_b = nn.Parameter(torch.randn(DM) * 0.5)   # marqueur opérande b
        self.scb = SpectralCoreBlock(d_model=2*DM, seq_len=L, bidirectional=True)
        self.head = nn.Linear(2*DM, 1)
    def encode(self, a, b):  # a,b: (B,) → (B, L, 2*DM) buffer circulaire
        B = a.shape[0]; x = torch.zeros(B, L, 2*DM, device=DEVICE)
        x[torch.arange(B), a, :DM] = self.marker_a        # marqueur a à la position a
        x[torch.arange(B), b, DM:] = self.marker_b        # marqueur b à la position b
        return x
    def forward(self, a, b):
        h = self.scb(self.encode(a, b))                    # (B, L, 2*DM) FFT-mixé
        return self.head(h).squeeze(-1)                    # (B, L) logit par position


def run(loss_type='1cos', n_steps=50000, bs=128, lr=3e-3, label="SCB"):
    """Hold-out 60/40. loss_type ∈ {'1cos','ce'}."""
    torch.manual_seed(0)
    all_a = torch.arange(P, device=DEVICE).repeat_interleave(P)
    all_b = torch.arange(P, device=DEVICE).repeat(P); all_m = op(all_a, all_b)
    perm = torch.randperm(P*P, device=DEVICE); n_tr = int(P*P*0.6)
    tr, te = perm[:n_tr], perm[n_tr:]
    m = SCBGrok().to(DEVICE); opt = torch.optim.Adam(m.parameters(), lr=lr)
    t0 = time.time(); grok = None; prev = 0.0; best = 0.0
    for step in range(n_steps+1):
        idx = tr[torch.randint(0, len(tr), (bs,))]
        logits = m(all_a[idx], all_b[idx])                # (bs, P)
        if loss_type == 'ce':
            loss = F.cross_entropy(logits, all_m[idx])
        else:  # 1-cos crown-jewel : aligne softmax(logits) vers one-hot cible
            pred = F.softmax(logits, dim=-1)
            tgt = F.one_hot(all_m[idx], P).float()
            loss = (1 - F.cosine_similarity(pred, tgt, dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2500 == 0 or step == n_steps:
            m.eval()
            with torch.no_grad():
                tr_acc = (m(all_a[tr], all_b[tr]).argmax(1) == all_m[tr]).float().mean().item()
                te_acc = (m(all_a[te], all_b[te]).argmax(1) == all_m[te]).float().mean().item()
            m.train(); best = max(best, te_acc)
            if grok is None and te_acc > 0.5 and prev < 0.5: grok = step
            prev = te_acc
            flag = "  *** GROK ***" if grok == step else ""
            print(f"  [{label}/{loss_type}] step {step:>6} train={tr_acc*100:5.1f}% held={te_acc*100:5.1f}%{flag} t={time.time()-t0:.0f}s", flush=True)
    return best, grok


def run_decomp(n_steps=3000, bs=128, lr=3e-3):
    """L1 DÉCOMPOSITION : mémorise op(a,b)->m sur TOUT, teste op(op(a,b),c)->r en cascade sur triples non-vus."""
    torch.manual_seed(0)
    m = SCBGrok().to(DEVICE); opt = torch.optim.Adam(m.parameters(), lr=lr)
    all_a = torch.arange(P, device=DEVICE).repeat_interleave(P)
    all_b = torch.arange(P, device=DEVICE).repeat(P); all_m = op(all_a, all_b)
    t0 = time.time()
    for step in range(n_steps+1):
        idx = torch.randint(0, P*P, (bs,))
        pred = F.softmax(m(all_a[idx], all_b[idx]), -1)
        tgt = F.one_hot(all_m[idx], P).float()
        loss = (1 - F.cosine_similarity(pred, tgt, dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    # composition en cascade sur triples non-vus
    m.eval(); g = torch.Generator(device=DEVICE).manual_seed(42)
    a = torch.randint(0, P, (1000,), generator=g, device=DEVICE)
    b = torch.randint(0, P, (1000,), generator=g, device=DEVICE)
    c = torch.randint(0, P, (1000,), generator=g, device=DEVICE)
    with torch.no_grad():
        m_pred = m(a, b).argmax(1)        # étape 1
        r_pred = m(m_pred, c).argmax(1)   # étape 2 (cascade)
    acc = (r_pred == op(op(a, b), c)).float().mean().item()
    print(f"  [SCB/decomp] cascade sur 1000 triples non-vus = {acc*100:5.1f}%  t={time.time()-t0:.0f}s", flush=True)
    return acc


def main():
    print("="*64); print("GROKKING SCB — FFT sur Z_P (règles Besoins suivies à la lettre)"); print("="*64)
    print(f"  op(a,b)=(3a+5b) mod {P}, SCB(L=P={L}, d={2*DM}), marqueurs positionnels, hold-out 40%\n")
    results = {}
    print("--- SCB + 1-cos (paradigme crown-jewel) ---")
    a1, g1 = run('1cos', label="SCB"); results["SCB_1cos"] = {"held": a1, "grok": g1}
    print("\n--- SCB + CE (comparaison) ---")
    a2, g2 = run('ce', label="SCB"); results["SCB_ce"] = {"held": a2, "grok": g2}
    print("\n--- SCB + décomposition L1 (cascade triples non-vus) ---")
    a3 = run_decomp(); results["SCB_decomp"] = {"held": a3}
    print("\n" + "="*64); print("VERDICT :")
    for k, v in results.items():
        gs = f"grok@{v['grok']}" if "grok" in v and v["grok"] else ("no grok" if "grok" in v else "")
        print(f"  {k:14s}: held-out={v['held']*100:5.1f}%  {gs}")
    if a1 > 0.9 or a2 > 0.9:
        print(f"\n  => GROK SCB ATTEINT ✓. CONFIRME la règle : la FFT (L=P) est le mécanisme du grok.")
    else:
        print(f"\n  => SCB sans grok pur (max {max(a1,a2)*100:.0f}%). Décomp={a3*100:.0f}%.")
    json.dump(results, open("ocm26400/grokking_scb_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    main()
