#!/usr/bin/env python3
"""ÉTAPE 1 — BRANCHER LES LOBES RÉELS au pipeline unifié (AMV-256).

L'agent Explore a identifié le gap de câblage : les lobes (audio/image) émettent 64-d,
pas partitionnés en AMV-256 [ent|prop|op|meta]. Seul ConceptModel (texte) émet l'AMV.
OmniModel (omni.py) intègre audio+image mais UnifiedPipeline n'y est pas branché.

Ici on livre la pièce manquante :
  1. AMVAdapter : embedding lobe (64-d) → AMV-256 (ent = embedding, prop/op/meta = 0).
  2. Vrai audio (SpeechCommands, petit sous-ensemble) → AudioEncoder → AMV → cœur SCB → classifie.
  3. Boucle autonome éveil→gate→sommeil sur de la VRAIE perception.

⚠️ HONNÊTE (split perception/raisonnement) :
  - Perception (audio, CE) : grok normalement par entraînement — le sommeil y est OPTIONNEL.
  - Raisonnement (core, 1-cos) : c'est LÀ que sommeil+gate sont essentiels (étapes 2/3).
  La gate reste un signal de suivi valide dans les deux cas. On démontre le CÂBLAGE + la boucle
  sur vrai audio, sans prétendre que le sommeil aide la perception (il aide le raisonnement).
"""
import torch, torch.nn as nn, torch.nn.functional as F, json, os, glob, numpy as np
from ocm26400.amv import D_MODEL, PART
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.multimodal_encoders import AudioEncoder
from ocm26400.optimize_sleep import spectral_filter
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"


class AMVAdapter(nn.Module):
    """Lobe embedding (PART=64) → AMV-256 (ent=embedding, prop/op/meta=0). Pièce de câblage manquante."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(PART, PART)  # affine l'embedding lobe dans le slot ent
    def forward(self, lobe_emb):  # (B, PART) → (B, D_MODEL=256)
        amv = torch.zeros(lobe_emb.shape[0], D_MODEL, device=lobe_emb.device)
        amv[:, 0:PART] = self.proj(lobe_emb)   # ent partition
        return amv


class WiredPipeline(nn.Module):
    """Audio (réel) → AudioEncoder → AMV → SCB core → classifie. Le cœur spectral raisonne sur l'AMV."""
    def __init__(self, n_classes):
        super().__init__()
        self.lobe = AudioEncoder(out_dim=PART)          # signal → 64-d (perception, CE)
        self.adapter = AMVAdapter()                      # 64-d → AMV-256 (ent)
        self.core = SpectralCoreBlock(d_model=D_MODEL, seq_len=1, bidirectional=True)  # cœur SCB sur AMV
        self.head = nn.Linear(D_MODEL, n_classes)        # classifie (CE, perception)
    def forward(self, wav):
        emb = self.lobe(wav)                  # (B, 64)
        amv = self.adapter(emb).unsqueeze(1)  # (B, 1, 256) — AMV comme séquence L=1
        out = self.core(amv).squeeze(1)       # (B, 256)
        return self.head(out)                 # (B, n_classes)


def load_tiny_audio(words, n_per=15):
    """Petit sous-ensemble SpeechCommands (rapide) : n_per wav par mot."""
    import soundfile as sf
    X, Y = [], []
    for wi, w in enumerate(words):
        files = sorted(glob.glob(os.path.join(SC, w, "*.wav")))[:n_per]
        for p in files:
            y, sr = sf.read(p); y = y.astype(np.float32)
            if y.ndim > 1: y = y.mean(1)
            if len(y) < 16000: y = np.pad(y, (0, 16000 - len(y)))
            else: y = y[:16000]
            X.append(y); Y.append(wi)
    return torch.tensor(np.stack(X)), torch.tensor(Y)


def main():
    print("="*64); print("ÉTAPE 1 — Brancher le lobe audio RÉEL → AMV-256 → cœur SCB (boucle autonome)"); print("="*64)
    words = ["yes", "no", "up", "down", "left"]  # 5 mots, sous-ensemble tiny
    print(f"  Audio réel SpeechCommands : {len(words)} mots × 15 wav = 75 échantillons\n")
    X, Y = load_tiny_audio(words)
    # split 60/40
    perm = torch.randperm(len(X)); ntr = int(len(X)*0.6)
    tr_i, te_i = perm[:ntr], perm[ntr:]
    Xtr, Ytr = X[tr_i].to(DEVICE), Y[tr_i].to(DEVICE)
    Xte, Yte = X[te_i].to(DEVICE), Y[te_i].to(DEVICE)

    torch.manual_seed(0)
    model = WiredPipeline(len(words)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def acc():
        model.eval()
        with torch.no_grad(): return (model(Xte).argmax(1) == Yte).float().mean().item()
    def gate():
        model.eval()
        with torch.no_grad(): return F.softmax(model(Xte), -1).max(1).values.mean().item()
    def train(steps):
        model.train()
        for _ in range(steps):
            idx = torch.randint(0, len(Xtr), (32,)); loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    print("  --- VÉRIF CÂBLAGE : audio → AMV → SCB core tourne ---", flush=True)
    train(200)
    print(f"  ÉVEIL 200 steps (CE) : acc={acc()*100:5.1f}%  gate={gate():.3f}", flush=True)
    # sommeil (spectral) — honnête : sur perception, observer si ça change qch
    a0, g0 = acc(), gate()
    for c in range(3):
        spectral_filter(model, 0.5, 'low'); train(50)
        spectral_filter(model, 0.3, 'high'); train(50)
    a1, g1 = acc(), gate()
    print(f"  Après 3 cycles sommeil : acc={a1*100:5.1f}% (Δ{(a1-a0)*100:+.1f}pt)  gate={g1:.3f}", flush=True)

    print("\n" + "="*64); print("VERDICT — câblage lobe réel :")
    print(f"  Audio → AudioEncoder(64-d) → AMVAdapter(→256) → SCB core → classifie : ✓ fonctionne")
    print(f"  acc {a0*100:.0f}% → {a1*100:.0f}%, gate {g0:.3f} → {g1:.3f}")
    print("  → CÂBLAGE livré (AMVAdapter = pièce manquante). La boucle tourne sur VRAI audio.")
    print("  ⚠️ Honnête : la perception (CE) grok par entraînement ; le sommeil spectral aide le")
    print("     RAISONNEMENT (1-cos, étapes 2/3), pas la perception. Ici Δ petit = attendu.")
    json.dump({"words": len(words), "n": len(X), "acc_eveil": a0, "acc_sommeil": a1,
               "gate_eveil": g0, "gate_sommeil": g1}, open("ocm26400/step1_wire_lobes_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    main()
