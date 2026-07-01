#!/usr/bin/env python3
"""VÉRIFICATION L10 / HONNÊTETÉ de la capture simultanée (anti-régression).

L'audit DA+juges flaggue L10 (mixer FFT bidirectionnel → fuite champs futurs) comme critique
pour la capture multi-lobes. Ma capture (test_capture_simultanee.py) utilise L=1 (SOMME des
ents en 1 AMV, pas séquence multi-positions) → structurellement SANS fuite bidirectionnelle
(FFT sur L=1 = triviale, pas de mélange cross-position).

On le VÉRIFIE par contrôle de mélange (shuffle) :
  - HONNÊTE : superposition de 3 modalités du MÊME concept → acc (référence).
  - SHUFFLE  : 1 modalité sur 3 vient d'un concept DIFFÉRENT (brisure d'association).
    Si le cœur est honnête (combine les modalités, pas de raccourci/fuite) → acc chute.
    Si leak/shortcut (ignore les modalités, triche positionnelle) → acc reste haute.
  - CHECK L=1 : confirme qu'aucun mélange cross-position n'a lieu (capture = somme, L=1).

Verdict : si shuffle → chute nette, la capture est L10-safe + honnête (pas de raccourci).
"""
import torch, torch.nn as nn, torch.nn.functional as F, json
from ocm26400.spectral_core import SpectralCoreBlock
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
P = 10; PART = 64; D_MODEL = 256


def canon():
    g = torch.Generator(device=DEVICE).manual_seed(42)
    return torch.linalg.qr(torch.randn(P, PART, device=DEVICE, generator=g).T)[0].T  # (P,PART) figé


class SynthLobe(nn.Module):  # 3 "modalités" = projections différentes du concept
    def __init__(self, seed):
        super().__init__(); g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(torch.randn(PART, PART, generator=g) * 0.3)
    def forward(self, c):  # c (B,) → ent (B,PART) : projection bruitée du canon[c]
        return canon()[c] @ self.W + 0.4 * torch.randn(len(c), PART, device=DEVICE)


def superpose(ents):  # L=1 : somme des ents (PAS de séquence multi-positions → pas de fuite L10)
    return sum(ents)


def main():
    print("="*64); print("VÉRIFICATION L10 / HONNÊTETÉ — capture simultanée (L=1 somme)"); print("="*64)
    C = canon()
    lobe1, lobe2, lobe3 = SynthLobe(1).to(DEVICE), SynthLobe(2).to(DEVICE), SynthLobe(3).to(DEVICE)
    core = SpectralCoreBlock(d_model=D_MODEL, seq_len=1).to(DEVICE)
    opt = torch.optim.Adam(core.parameters(), lr=3e-3)
    # entraînement : superposition 3 modalités (même concept) → concept (1-cos), L=1
    for _ in range(1500):
        c = torch.randint(0, P, (64,), device=DEVICE)
        sup = superpose([lobe1(c), lobe2(c), lobe3(c)])
        amv = torch.zeros(64, D_MODEL, device=DEVICE); amv[:, 0:PART] = sup
        out = core(amv.unsqueeze(1)).squeeze(1)[:, 0:PART]
        loss = (1 - F.cosine_similarity(out, C[c], dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    core.eval()
    def predict(ents):
        sup = superpose(ents); amv = torch.zeros(len(sup), D_MODEL, device=DEVICE); amv[:, 0:PART] = sup
        with torch.no_grad(): out = core(amv.unsqueeze(1)).squeeze(1)[:, 0:PART]
        return (out @ C.t()).argmax(1)
    n = 1000; c = torch.randint(0, P, (n,), device=DEVICE)
    # 1. HONNÊTE : 3 modalités même concept
    acc_h = (predict([lobe1(c), lobe2(c), lobe3(c)]) == c).float().mean().item()
    # 2. SHUFFLE : 1 modalité (lobe2) d'un concept ALÉATOIRE (brisure)
    cw = torch.randint(0, P, (n,), device=DEVICE)  # concept faux pour lobe2
    acc_s = (predict([lobe1(c), lobe2(cw), lobe3(c)]) == c).float().mean().item()
    # 3. CONTRÔLE L=1 : si on permutait en séquence L=3 (bidirectionnelle), fuite possible ?
    #    On vérifie juste que la capture actuelle (L=1 somme) ne dépend pas de l'ordre/position.
    acc_rev = (predict([lobe3(c), lobe2(c), lobe1(c)]) == c).float().mean().item()  # ordre permuté

    print(f"\n  HONNÊTE (3 modalités même concept)   : {acc_h*100:5.1f}%", flush=True)
    print(f"  SHUFFLE (1 modalité concept faux)    : {acc_s*100:5.1f}%   ← doit chuter si honnête", flush=True)
    print(f"  ORDRE permuté (L=1 = invariant)      : {acc_rev*100:5.1f}%   ← doit = honnête (pas de fuite position)", flush=True)
    drop = acc_h - acc_s
    print("\n" + "="*64); print("VERDICT L10/honnêteté :")
    honest = drop > 0.2
    l1safe = abs(acc_h - acc_rev) < 0.03
    print(f"  Shuffle fait chuter l'acc de {drop*100:.0f}pt → {'HONNÊTE (pas de raccourci/fuite) ✓' if honest else 'fuite/shortcut suspecté ✗'}")
    print(f"  L=1 invariant à l'ordre (Δ{abs(acc_h-acc_rev)*100:.0f}pt) → {'L10-safe (pas de mélange cross-position) ✓' if l1safe else 'dépend de la position ✗'}")
    if honest and l1safe:
        print("\n  => CAPTURE L10-SAFE & HONNÊTE ✓ : L=1 somme, pas de fuite bidirectionnelle,")
        print("     pas de raccourci. L10 (masque futur) restera à appliquer au DOSC multi-champ L>1 futur.")
    json.dump({"honest": acc_h, "shuffled": acc_s, "reordered": acc_rev, "drop": drop},
              open("ocm26400/verify_l10_capture_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    main()
