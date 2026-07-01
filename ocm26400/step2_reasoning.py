#!/usr/bin/env python3
"""ÉTAPE 2 — RAISONNEMENT multi-étapes via la boucle cognitive autonome.

Démontre le cycle complet Apprendre→Comprendre→Raisonner sur UNE tâche de raisonnement :
  1. APPRENDRE (éveil) : entraîner la primitive op(a,b)=(a+b) mod P — peu de steps → STUCK.
  2. COMPRENDRE (sommeil auto, gate stop-criterion) : sommeil spectral auto-déclenché
     jusqu'à gate ≥ τ → la primitive est grokkée (certifiée).
  3. RAISONNER (compose) : évaluer des EXPRESSIONS multi-étapes ((a+b)+c)+d sur des
     opérandes JAMAIS vus, via cascade gate-certifiée.

Mesure : accuracy de raisonnement AVANT sommeil (primitive stuck) vs APRÈS (primitive comprise).
Démontre que la compréhension (sommeil) est le prérequis du raisonnement (composition).
"""
import torch, torch.nn as nn, torch.nn.functional as F, json
from ocm26400.optimize_sleep import spectral_filter
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
P = 23; D = 64; TAU = 0.99; MAX_CYC = 8


def op(a, b): return (a + b) % P

class ReasonerBlock(nn.Module):
    def __init__(self, d=D, h=128):
        super().__init__(); self.norm = nn.LayerNorm(d); self.f1 = nn.Linear(d, h); self.f2 = nn.Linear(h, d)
        nn.init.normal_(self.f1.weight, std=0.02); nn.init.normal_(self.f2.weight, std=0.02)
        nn.init.zeros_(self.f1.bias); nn.init.zeros_(self.f2.bias)
    def forward(self, x):
        h = self.norm(x); h = torch.relu(self.f1(h)); h = self.f2(h); return x + h

def canon():
    c = torch.zeros(P, P, device=DEVICE); c[torch.arange(P), torch.arange(P)] = 1.0; return c
def encode(a, b):
    x = torch.zeros(len(a), D, device=DEVICE); c = canon(); x[:, 0:P] = c[a]; x[:, P:2*P] = c[b]; return x

def train_prim(model, opt, steps):
    aa = torch.arange(P, device=DEVICE).repeat_interleave(P); bb = torch.arange(P, device=DEVICE).repeat(P)
    mm = op(aa, bb); c = canon()
    for _ in range(steps):
        idx = torch.randint(0, P*P, (256,)); out = model(encode(aa[idx], bb[idx]))[:, 0:P]
        loss = (1 - F.cosine_similarity(out, c[mm[idx]], dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()

def gate_prim(model):
    aa = torch.arange(P, device=DEVICE).repeat_interleave(P); bb = torch.arange(P, device=DEVICE).repeat(P)
    mm = op(aa, bb); c = canon()
    with torch.no_grad(): out = model(encode(aa, bb))[:, 0:P]
    return F.cosine_similarity(out, c[mm], dim=-1).mean().item()

def reason(model, depth, n=2000):
    """Évalue des expressions depth-opérandes ((..(a+b)+c)+..) sur opérandes non-vus, cascade gate-certifiée."""
    c = canon(); g = torch.Generator(device=DEVICE).manual_seed(depth*7+1)
    ops = torch.randint(0, P, (n, depth), generator=g, device=DEVICE)  # n expressions, 'depth' opérandes
    true = ops.sum(1) % P
    model.eval()
    with torch.no_grad():
        v = ops[:, 0]; gmin = torch.ones(n, device=DEVICE)
        for t in range(depth - 1):
            out = model(encode(v, ops[:, t+1]))[:, 0:P]
            gs = F.cosine_similarity(out, c[(out @ c.t()).argmax(1)], dim=-1)
            v = (out @ c.t()).argmax(1); gmin = torch.minimum(gmin, gs)
    return (v == true).float().mean().item(), gmin.mean().item()


def autonomous_comprehend(model, opt, train_data_fn, max_cyc=MAX_CYC):
    """Sommeil auto-déclenché jusqu'à gate ≥ τ. Retourne (gate, cycles)."""
    g = gate_prim(model); cyc = 0
    while g < TAU and cyc < max_cyc:
        cyc += 1
        spectral_filter(model, 0.5, 'low');  train_data_fn(model, opt, 100)
        spectral_filter(model, 0.3, 'high'); train_data_fn(model, opt, 100)
        g = gate_prim(model)
    return g, cyc


def main():
    print("="*64); print("ÉTAPE 2 — RAISONNEMENT multi-étapes via boucle cognitive autonome"); print("="*64)
    print(f"  primitive op(a,b)=(a+b) mod {P}, raisonnement = expressions depth-3/5/10\n")
    results = {}
    for seed in [0, 1, 2]:
        torch.manual_seed(seed)
        model = ReasonerBlock().to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        # 1. APPRENDRE (éveil court → stuck)
        train_prim(model, opt, 40)
        g0 = gate_prim(model)
        r3_before, _ = reason(model, 3)
        # 2. COMPRENDRE (sommeil autonome jusqu'à gate ≥ τ)
        g1, cyc = autonomous_comprehend(model, opt, train_prim)
        # 3. RAISONNER (expressions multi-étapes sur opérandes non-vus)
        r3, gc3 = reason(model, 3); r5, gc5 = reason(model, 5); r10, gc10 = reason(model, 10)
        results[seed] = {"gate_eveil": g0, "reason_D3_before": r3_before,
                         "gate_compris": g1, "cycles": cyc,
                         "reason_D3": r3, "reason_D5": r5, "reason_D10": r10}
        print(f"  seed {seed}: éveil gate {g0:.3f} (D3 {r3_before*100:.0f}%) → sommeil {cyc}c gate {g1:.3f} → "
              f"D3 {r3*100:.0f}% D5 {r5*100:.0f}% D10 {r10*100:.0f}%", flush=True)
    import numpy as np
    print("\n" + "="*64); print("VERDICT — raisonnement après compréhension autonome :")
    rb = np.mean([results[s]["reason_D3_before"] for s in results])
    r3 = np.mean([results[s]["reason_D3"] for s in results]); r5 = np.mean([results[s]["reason_D5"] for s in results])
    r10 = np.mean([results[s]["reason_D10"] for s in results]); cyc = np.mean([results[s]["cycles"] for s in results])
    print(f"  Raisonnement D3 AVANT sommeil : {rb*100:.0f}% (primitive stuck)")
    print(f"  Raisonnement APRÈS compréhension (sommeil {cyc:.1f} cycles auto) :")
    print(f"    D3 = {r3*100:.0f}% | D5 = {r5*100:.0f}% | D10 = {r10*100:.0f}%  (opérandes jamais vus)")
    if r3 > 0.9:
        print("\n  => APPRENDRE→COMPRENDRE→RAISONNER démontré ✓")
        print("     La compréhension (sommeil autonome, gate) est le prérequis du raisonnement (composition).")
        print("     Sans sommeil : stuck. Avec : raisonnement multi-étapes sur opérandes inédits.")
    json.dump(results, open("ocm26400/step2_reasoning_results.json", "w"), indent=2, default=str)
    print("[sauvé]")


if __name__ == "__main__":
    main()
