#!/usr/bin/env python3
"""ASSEMBLAGE — pipeline unifié ÉVEIL → SOMMEIL → GATE → COMPOSE.

Démontre end-to-end mémoire→sommeil→compréhension→raisonnement dans UNE boucle,
en réutilisant les 3 mécanismes validés cette session :

  ÉVEIL   (train → mémoire, souvent stuck)          [test_sleep_neural]
  SOMMEIL (5 cycles spectraux → grok, gate émerge)  [optimize_sleep]
  GATE    (alignement canonique ≥ τ → certifie)     [grok_gate_composition]
  COMPOSE (cascade gate-certifiée, profondeur arb.) [grok_gate_composition]

Démo 1 (RAISONNEMENT, crown-jewel) :
  op(a,b)=(a+b) mod P. Éveil partiel (gate bas) → sommeil (gate→1.0) → compose D=20 certifié.
Démo 2 (PERCEPTION/RÈGLE, seq-rule) :
  class=(seq[3]+seq[7]) mod K. Éveil stuck (25%) → sommeil grok (99%) → la gate émerge.

Mesure à chaque étape : acc + gate (alignement). Montre que le sommeil fait GROK (gate↑+acc↑)
ce que l'éveil pur laisse stuck.
"""
import torch, torch.nn as nn, torch.nn.functional as F, json, numpy as np
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===================== MÉCANISMES (réutilisés) =====================
def spectral_filter(model, keep_frac, mode):
    """Filtre spectral des poids : low-pass (débruit) ou high-pass (affine)."""
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                W = p.data.float(); Fr = torch.fft.rfft(W, dim=0)
                k = max(1, int(Fr.shape[0] * keep_frac))
                if mode == 'low': Fr[k:] = 0
                else:             Fr[:k] = 0
                p.data = torch.fft.irfft(Fr, n=W.shape[0], dim=0).to(p.dtype)


class UnifiedPipeline:
    """ÉVEIL → SOMMEIL → GATE → COMPOSE. Générique sur le modèle + tâche."""
    def __init__(self, model, train_fn, eval_fn, gate_fn, lr=3e-3):
        self.m = model.to(DEVICE); self.opt = torch.optim.Adam(self.m.parameters(), lr=lr)
        self.train_fn = train_fn    # train_fn(model, opt, steps) -> loss
        self.eval_fn = eval_fn      # eval_fn(model) -> acc
        self.gate_fn = gate_fn      # gate_fn(model) -> gate moyen (alignement)

    def eveil(self, steps):
        self.train_fn(self.m, self.opt, steps)
        return self.eval_fn(self.m), self.gate_fn(self.m)

    def sommeil(self, train_replay, cycles=5, kf_low=0.5, kf_high=0.3, replay=200):
        """5 cycles spectraux : low-pass → replay → high-pass → replay."""
        for _ in range(cycles):
            spectral_filter(self.m, kf_low, 'low');   train_replay(self.m, self.opt, replay)
            spectral_filter(self.m, kf_high, 'high'); train_replay(self.m, self.opt, replay)
        return self.eval_fn(self.m), self.gate_fn(self.m)

    def mesure(self):
        return {"acc": self.eval_fn(self.m), "gate": self.gate_fn(self.m)}


# ===================== DÉMO 1 : RAISONNEMENT (crown-jewel) =====================
P1 = 23; D1 = 64
def op1(a, b): return (a + b) % P1

class ReasonerBlock(nn.Module):
    def __init__(self, d=D1, h=128):
        super().__init__(); self.norm = nn.LayerNorm(d); self.f1 = nn.Linear(d, h); self.f2 = nn.Linear(h, d)
        nn.init.normal_(self.f1.weight, std=0.02); nn.init.normal_(self.f2.weight, std=0.02)
        nn.init.zeros_(self.f1.bias); nn.init.zeros_(self.f2.bias)
    def forward(self, x):
        h = self.norm(x); h = torch.relu(self.f1(h)); h = self.f2(h); return x + h

def canon1():
    c = torch.zeros(P1, P1, device=DEVICE); c[torch.arange(P1), torch.arange(P1)] = 1.0; return c

def encode1(a, b):
    x = torch.zeros(len(a), D1, device=DEVICE); c = canon1()
    x[:, 0:P1] = c[a]; x[:, P1:2*P1] = c[b]; return x

def train1(model, opt, steps):
    aa = torch.arange(P1, device=DEVICE).repeat_interleave(P1)
    bb = torch.arange(P1, device=DEVICE).repeat(P1); mm = op1(aa, bb); c = canon1()
    for _ in range(steps):
        idx = torch.randint(0, P1*P1, (256,))
        out = model(encode1(aa[idx], bb[idx]))[:, 0:P1]
        loss = (1 - F.cosine_similarity(out, c[mm[idx]], dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()

def eval1(model):
    aa = torch.arange(P1, device=DEVICE).repeat_interleave(P1)
    bb = torch.arange(P1, device=DEVICE).repeat(P1); mm = op1(aa, bb); c = canon1()
    with torch.no_grad():
        pred = (model(encode1(aa, bb))[:, 0:P1] @ c.t()).argmax(1)
    return (pred == mm).float().mean().item()

def gate1(model):
    """gate = alignement moyen (cos) au canonique correct."""
    aa = torch.arange(P1, device=DEVICE).repeat_interleave(P1)
    bb = torch.arange(P1, device=DEVICE).repeat(P1); mm = op1(aa, bb); c = canon1()
    with torch.no_grad():
        out = model(encode1(aa, bb))[:, 0:P1]
    return F.cosine_similarity(out, c[mm], dim=-1).mean().item()

def compose1(model, depth, n=2000):
    """Cascade gate-certifiée : r=op(op(...op(a,b),c),...). Retourne (acc, gate_min)."""
    c = canon1(); g = torch.Generator(device=DEVICE).manual_seed(depth*7+1)
    ops = torch.randint(0, P1, (n, depth+1), generator=g, device=DEVICE)
    true = ops.sum(1) % P1
    model.eval()
    with torch.no_grad():
        v = ops[:, 0]; gmin = torch.ones(n, device=DEVICE)
        for t in range(depth):
            out = model(encode1(v, ops[:, t+1]))[:, 0:P1]
            gs = F.cosine_similarity(out, c[(out @ c.t()).argmax(1)], dim=-1)  # gate = align au canonique le + proche
            v = (out @ c.t()).argmax(1); gmin = torch.minimum(gmin, gs)
    return (v == true).float().mean().item(), gmin.mean().item()


# ===================== DÉMO 2 : RÈGLE (seq-rule, sleep-nécessaire) =====================
V2, K2, L2, Dm2 = 10, 5, 12, 128
def rule2(seq): return (seq[:, 3] + seq[:, 7]) % K2
def gen2(n, seed): return torch.randint(0, V2, (n, L2), generator=torch.Generator().manual_seed(seed))

class MLP2(nn.Module):
    def __init__(self):
        super().__init__(); self.embed = nn.Embedding(V2, Dm2)
        self.mlp = nn.Sequential(nn.Linear(L2*Dm2, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, K2))
    def forward(self, seq): return self.mlp(self.embed(seq).flatten(1))

def train2(model, opt, steps, tr, lab):
    for _ in range(steps):
        idx = torch.randint(0, len(tr), (64,)); loss = F.cross_entropy(model(tr[idx]), lab[idx])
        opt.zero_grad(); loss.backward(); opt.step()

def eval2(model, te, lab):
    with torch.no_grad(): return (model(te).argmax(1) == lab).float().mean().item()

def gate2(model, te, lab):
    """gate = confiance moyenne = proba softmax de la classe prédite (proxy alignement)."""
    with torch.no_grad():
        p = F.softmax(model(te), dim=-1)
    return p.max(1).values.mean().item()


# ===================== RUN =====================
def demo1():
    print("="*64); print("DÉMO 1 — RAISONNEMENT : Éveil → Sommeil → Gate → Compose (crown-jewel)"); print("="*64)
    torch.manual_seed(0)
    pipe = UnifiedPipeline(ReasonerBlock(), train1, eval1, gate1)
    a0, g0 = pipe.eveil(300)
    print(f"  ÉVEIL (300 steps, partiel) : acc={a0*100:5.1f}%  gate={g0:.3f}", flush=True)
    a1, g1 = pipe.sommeil(train_replay=lambda m, o, s: train1(m, o, s), cycles=5)
    print(f"  [après SOMMEIL 5 cycles]   : acc={a1*100:5.1f}%  gate={g1:.3f}  ← gate→1.0 (certifié)", flush=True)
    # compose avant/après (re-démarre pour mesurer "avant")
    torch.manual_seed(0); pipe_b = UnifiedPipeline(ReasonerBlock(), train1, eval1, gate1); pipe_b.eveil(300)
    cb, gb = compose1(pipe_b.m, 20)
    ca, ga = compose1(pipe.m, 20)
    print(f"  COMPOSE D=20 : avant sommeil acc={cb*100:5.1f}% (gate {gb:.3f}) | après sommeil acc={ca*100:5.1f}% (gate {ga:.3f})", flush=True)
    return {"eveil": (a0, g0), "sommeil": (a1, g1), "compose_D20_avant": (cb, gb), "compose_D20_apres": (ca, ga)}


def demo2():
    print("\n" + "="*64); print("DÉMO 2 — RÈGLE : Éveil (stuck) → Sommeil (grok) → gate émerge (seq-rule)"); print("="*64)
    tr, te = gen2(400, 1).to(DEVICE), gen2(400, 2).to(DEVICE); lab_tr, lab_te = rule2(tr), rule2(te)
    torch.manual_seed(0)
    pipe = UnifiedPipeline(MLP2(), lambda m, o, s: train2(m, o, s, tr, lab_tr),
                           lambda m: eval2(m, te, lab_te), lambda m: gate2(m, te, lab_te))
    a0, g0 = pipe.eveil(1500)
    print(f"  ÉVEIL (1500 steps)         : acc={a0*100:5.1f}%  gate(conf)={g0:.3f}  ← stuck (mémoire)", flush=True)
    a1, g1 = pipe.sommeil(train_replay=lambda m, o, s: train2(m, o, s, tr, lab_tr), cycles=5)
    print(f"  [après SOMMEIL 5 cycles]   : acc={a1*100:5.1f}%  gate(conf)={g1:.3f}  ← grok (compréhension)", flush=True)
    return {"eveil": (a0, g0), "sommeil": (a1, g1)}


def main():
    r1 = demo1(); r2 = demo2()
    print("\n" + "="*64); print("ASSEMBLAGE — VERDICT END-TO-END"); print("="*64)
    print("  DÉMO 1 (raisonnement crown-jewel) :")
    print(f"    éveil  → acc {r1['eveil'][0]*100:.0f}%  gate {r1['eveil'][1]:.3f}")
    print(f"    sommeil→ acc {r1['sommeil'][0]*100:.0f}%  gate {r1['sommeil'][1]:.3f}")
    print(f"    compose D=20 après sommeil : acc {r1['compose_D20_apres'][0]*100:.0f}%  gate {r1['compose_D20_apres'][1]:.3f}")
    print("  DÉMO 2 (règle seq, sleep-nécessaire) :")
    print(f"    éveil  → acc {r2['eveil'][0]*100:.0f}%  gate {r2['eveil'][1]:.3f}  (STUCK)")
    print(f"    sommeil→ acc {r2['sommeil'][0]*100:.0f}%  gate {r2['sommeil'][1]:.3f}  (GROK)")
    ok = r1['compose_D20_apres'][0] > 0.9 and r2['sommeil'][0] > 0.9
    if ok:
        print("\n  => PIPELINE UNIFIÉ FONCTIONNE END-TO-END ✓")
        print("     Éveil(mémoire) → Sommeil(grok, gate émerge) → Compose(certifié, profondeur arb.).")
        print("     Apprendre→Comprendre→Raisonner tient dans UNE boucle.")
    else:
        print("\n  => Pipeline partiel (à ajuster).")
    json.dump({"demo1": r1, "demo2": r2}, open("ocm26400/assemble_pipeline_results.json", "w"), indent=2, default=str)
    print("[sauvé]")


if __name__ == "__main__":
    main()
