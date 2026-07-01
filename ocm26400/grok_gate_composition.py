#!/usr/bin/env python3
"""GROK + GATE → COMPOSITION ARBITRAIRE (test du mécanisme de l'utilisateur).

Thèse utilisateur : "La gate fait grokker — la capacité compositionnelle ARBITRAIRE
émerge d'UNE primitive grokkée + la gate qui certifie."

Mécanisme (Besoins + ACSP/LSRA) :
  1. Grok UNE primitive op(a,b) à 100% (gate ≥ 0.99 sur la primitive).
  2. La GATE = alignement (cosinus) de la sortie au dictionnaire canonique (L_align).
     Si alignement ≥ τ → étape CERTIFIÉE (correcte) ; sinon → incertaine.
  3. Composer en cascade : r = op(op(...op(a,b),c),...) à profondeur D.
  4. Sans gate : l'erreur s'accumule (acc ≈ per_step^D). Avec primitive parfaitement
     grokkée + gate certifiant chaque étape → composition EXACTE à profondeur arbitraire.

Test : primitive op(a,b)=(a+b) mod P grokkée sur TOUTES les paires (1-cos, crown-jewel).
Puis cascade à D ∈ {1,2,3,5,10,20,50}. On mesure acc(D) ET le gate (alignement moyen).
Prédiction utilisateur : acc reste ~100% à D arbitraire (chaque étape certifiée).
"""
import torch, torch.nn as nn, torch.nn.functional as F, json, time
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
P = 23; D_MODEL = 64


def op(a, b): return (a + b) % P  # addition mod P (associative, cascade propre)


class ReasonerBlock(nn.Module):
    def __init__(self, d=D_MODEL, h=128):
        super().__init__()
        self.norm = nn.LayerNorm(d); self.f1 = nn.Linear(d, h); self.f2 = nn.Linear(h, d)
        nn.init.normal_(self.f1.weight, std=0.02); nn.init.normal_(self.f2.weight, std=0.02)
        nn.init.zeros_(self.f1.bias); nn.init.zeros_(self.f2.bias)
    def forward(self, x):
        h = self.norm(x); h = torch.relu(self.f1(h)); h = self.f2(h); return x + h


def make_canon():
    c = torch.zeros(P, P, device=DEVICE); c[torch.arange(P), torch.arange(P)] = 1.0; return c


def encode(a, b, canon):  # one-hot CONCAT (sans collision) : ent=canon(a) | prop=canon(b)
    bs = a.shape[0]; x = torch.zeros(bs, D_MODEL, device=DEVICE)
    x[:, 0:P] = canon[a]; x[:, P:2*P] = canon[b]; return x


def gate_score(ent, canon):  # LA GATE : alignement (cosinus) au dictionnaire canonique
    """Max cosine de la sortie vers les P canoniques = confiance que l'étape est valide."""
    sims = ent @ canon.t()                       # (bs, P)
    max_sim = sims.max(dim=1).values             # (bs,) = alignement au canonique le + proche
    return max_sim, sims.argmax(1)               # gate + ID prédit


def grok_primitive(n_steps=4000, bs=256, lr=3e-3):
    """Grok op(a,b) sur TOUTES les P² paires (primitive déterministe)."""
    torch.manual_seed(0)
    canon = make_canon()
    blk = ReasonerBlock().to(DEVICE); opt = torch.optim.Adam(blk.parameters(), lr=lr)
    all_a = torch.arange(P, device=DEVICE).repeat_interleave(P)
    all_b = torch.arange(P, device=DEVICE).repeat(P); all_m = op(all_a, all_b)
    t0 = time.time()
    for step in range(n_steps+1):
        idx = torch.randint(0, P*P, (bs,))
        out = blk(encode(all_a[idx], all_b[idx], canon))[:, 0:P]
        loss = (1 - F.cosine_similarity(out, canon[all_m[idx]], dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0 or step == n_steps:
            blk.eval()
            with torch.no_grad():
                _, pred = gate_score(blk(encode(all_a, all_b, canon))[:, 0:P], canon)
                acc = (pred == all_m).float().mean().item()
            blk.train()
            print(f"  [grok primitive] step {step:>4} acc={acc*100:5.1f}% t={time.time()-t0:.0f}s", flush=True)
    return blk, canon


def compose_cascade(blk, canon, depth, n=2000):
    """Cascade profondeur 'depth' : r = op(op(...op(a,b),c),...). Measure acc + gate.
    'depth' = nombre d'opérations appliquées (depth=1 → 1 op, 2 opérandes)."""
    blk.eval()
    g = torch.Generator(device=DEVICE).manual_seed(depth * 7 + 1)
    operands = torch.randint(0, P, (n, depth + 1), generator=g, device=DEVICE)  # n suites de depth+1 ops
    # résultat vrai = somme des opérandes mod P
    true = operands.sum(dim=1) % P
    with torch.no_grad():
        v = operands[:, 0]                       # état courant (ID)
        gate_min = torch.ones(n, device=DEVICE)  # gate minim sur la cascade
        for t in range(depth):                   # depth opérations
            nxt = operands[:, t + 1]
            out = blk(encode(v, nxt, canon))[:, 0:P]
            gs, v = gate_score(out, canon)       # gate certifie + avance l'état
            gate_min = torch.minimum(gate_min, gs)
    acc = (v == true).float().mean().item()
    gate_mean = gate_min.mean().item()           # gate le + faible sur la cascade (maillon faible)
    return acc, gate_mean


def main():
    print("="*64); print("GROK + GATE → COMPOSITION ARBITRAIRE (thèse utilisateur)"); print("="*64)
    print(f"  primitive op(a,b)=(a+b) mod {P}, ReasonerBlock(d={D_MODEL}), grokkée sur TOUTES les paires\n")
    blk, canon = grok_primitive()
    print("\n  --- Cascade à profondeur D (prédiction : acc ~100% via gate certifiant chaque étape) ---")
    results = {}
    for depth in [1, 2, 3, 5, 10, 20, 50]:
        acc, gate = compose_cascade(blk, canon, depth)
        results[depth] = {"acc": acc, "gate_min": gate}
        cert = "CERTIFIÉ" if gate > 0.99 else f"gate={gate:.3f}"
        print(f"  D={depth:>3} : acc={acc*100:5.1f}%  gate(min cascade)={gate:.4f}  {cert}", flush=True)
    print("\n" + "="*64); print("VERDICT :")
    d50 = results.get(50, {})
    if d50.get("acc", 0) > 0.95:
        print(f"  => COMPOSITION ARBITRAIRE ATTEINTE ✓ (D=50: {d50['acc']*100:.1f}%, gate {d50['gate_min']:.3f}).")
        print("  => CONFIRME la thèse : 1 primitive grokkée + gate qui certifie → capacité compositionnelle arbitraire.")
        print("  => Le grok = la gate, PAS le scale. Raisonner = enchaîner des étapes certifiées.")
    else:
        print(f"  => Composition dégrade à D=50 ({d50.get('acc',0)*100:.1f}%). Erreur accumulée malgré primitive grokkée.")
        print("  => La gate seule ne suffit pas si la primitive n'est pas EXACTE (per_step < 1.0).")
    json.dump(results, open("ocm26400/grok_gate_composition_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    main()
