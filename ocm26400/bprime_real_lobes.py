#!/usr/bin/env python3
"""B' sur VRAIS LOBES — chiffres parlés (SpeechCommands) + chiffres écrits (scikit digits).

Même architecture B' que bprime_implementation.py (lobes séparés → AMV canonique figé → cœur SCB),
mais sur DONNÉES RÉELLES avec concepts alignés (10 chiffres : 0-9 parlé ↔ 0-9 écrit).

  AudioLobe : AudioEncoder (Mel STFT → Conv) sur SpeechCommands "zero".."nine" → ent canonique (1-cos).
  ImageLobe : MLP patch sur scikit digits (8×8) → ent canonique (1-cos).
  Cœur SCB  : crown-jewel op sur l'AMV canonique (entraîné UNE fois).

Validations B' sur vrai données :
  1. Lobes perçoivent le chiffre (acc classification via canon).
  2. Cross-modal : chiffre parlé(c) ↔ chiffre écrit(c) → même AMV canonique (cos).
  3. REMPLAÇABILITÉ : cœur (non retrainé) raisonne sur AMV audio OU image.
"""
import torch, torch.nn as nn, torch.nn.functional as F, json, os, glob, numpy as np
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.multimodal_encoders import AudioEncoder
from ocm26400.audio_unified_m5scb import load_wav
from sklearn.datasets import load_digits
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
P = 10; PART = 64; D_MODEL = 256; T = 16000
SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
DIGITS = ["zero","one","two","three","four","five","six","seven","eight","nine"]


def canonical_dict():
    g = torch.Generator(device=DEVICE).manual_seed(42)
    C = torch.randn(P, PART, device=DEVICE, generator=g)
    return torch.linalg.qr(C.T)[0].T   # (P, PART) orthogonal, figé


def load_audio_digits(n_per=120):
    """n_per wav par chiffre → (N, T), (N,) labels."""
    X, Y = [], []
    for di, d in enumerate(DIGITS):
        for p in sorted(glob.glob(os.path.join(SC, d, "*.wav")))[:n_per]:
            X.append(load_wav(p, T).numpy()); Y.append(di)
    return torch.tensor(np.stack(X)), torch.tensor(Y)

def load_image_digits(n_per=150):
    """scikit digits → (N, 64), (N,). n_per par classe."""
    X, Y = load_digits(return_X_y=True); Xl, Yl = [], []
    for di in range(P):
        idx = np.where(Y == di)[0][:n_per]
        for i in idx: Xl.append(X[i]); Yl.append(di)
    return torch.tensor(np.stack(Xl)).float(), torch.tensor(Yl)


# ---- lobes (front-ends RÉELS, émettent l'ent canonique) ----
class AudioLobe(nn.Module):    # AudioEncoder(PART) = Mel STFT → Conv → ent (PART)
    def __init__(self): super().__init__(); self.enc = AudioEncoder(out_dim=PART)
    def forward(self, wav): return self.enc(wav)              # (B, T) → (B, PART)

class ImageLobe(nn.Module):    # MLP sur 8×8 digits → ent (PART)
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, PART))
    def forward(self, img): return self.net(img)              # (B, 64) → (B, PART)


def train_lobe(lobe, X, Y, canon, steps=1200, lr=3e-3, bs=64):
    opt = torch.optim.Adam(lobe.parameters(), lr=lr); N = len(X)
    for _ in range(steps):
        idx = torch.randint(0, N, (bs,))
        ent = lobe(X[idx]); loss = (1 - F.cosine_similarity(ent, canon[Y[idx]], dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()

def lobe_acc(lobe, X, Y, canon):
    lobe.eval()
    with torch.no_grad(): pred = (lobe(X) @ canon.t()).argmax(1)
    lobe.train(); return (pred == Y).float().mean().item()

def cross_modal(audio_l, image_l, Xa, Ya, Xi, Yi, canon, n=200):
    """chiffre parlé(c) ↔ chiffre écrit(c) → même AMV canonique ? cos moyen."""
    audio_l.eval(); image_l.eval()
    with torch.no_grad():
        # pour chaque classe c, compare l'AMV moyen audio(c) vs image(c)
        cos = []
        for c in range(P):
            ea = audio_l(Xa[Ya == c])[:20].mean(0)
            ei = image_l(Xi[Yi == c])[:20].mean(0)
            cos.append(F.cosine_similarity(ea, ei, dim=0).item())
    audio_l.train(); image_l.train(); return float(np.mean(cos))

def core_reason(core, ent_a, ent_b):
    amv = torch.zeros(ent_a.shape[0], D_MODEL, device=DEVICE)
    amv[:, 0:PART] = ent_a; amv[:, PART:2*PART] = ent_b
    return core(amv.unsqueeze(1)).squeeze(1)[:, 0:PART]

def train_core(core, canon, steps=1500, lr=3e-3):
    opt = torch.optim.Adam(core.parameters(), lr=lr)
    for _ in range(steps):
        a = torch.randint(0, P, (64,), device=DEVICE); b = torch.randint(0, P, (64,), device=DEVICE)
        out = core_reason(core, canon[a], canon[b]); loss = (1 - F.cosine_similarity(out, canon[(a+b)%P], dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()

def replaceability(core, audio_l, image_l, Xa, Ya, Xi, Yi, canon, n=400, gated=False):
    """Cœur non-retrainé raisonne sur AMV du lobe audio OU image.
    gated=False : AMV brut (bruité) → l'op amplifie l'erreur.
    gated=True  : GATE = snap l'AMV au canonique le + proche (argmax cos) AVANT le raisonnement
                  → le cœur raisonne sur l'AMV PROPRE (certifié). L'op n'amplifie plus le bruit."""
    core.eval(); audio_l.eval(); image_l.eval()
    a = torch.randint(0, len(Xa), (n,), device=DEVICE); b = torch.randint(0, len(Xa), (n,), device=DEVICE)
    ca, cb = Ya[a], Ya[b]; true = (ca + cb) % P
    def gate(ent):  # snap au canonique le + proche (= certification du concept)
        return canon[(ent @ canon.t()).argmax(1)] if gated else ent
    with torch.no_grad():
        ea, eb = audio_l(Xa[a]), audio_l(Xa[b])
        ra = (core_reason(core, gate(ea), gate(eb)) @ canon.t()).argmax(1)
        ia = torch.randint(0, len(Xi), (n,), device=DEVICE); ib = torch.randint(0, len(Xi), (n,), device=DEVICE)
        ria, rib = Yi[ia], Yi[ib]; itrue = (ria + rib) % P
        ei_a, ei_b = image_l(Xi[ia]), image_l(Xi[ib])
        ri = (core_reason(core, gate(ei_a), gate(ei_b)) @ canon.t()).argmax(1)
    core.train(); audio_l.train(); image_l.train()
    return (ra == true).float().mean().item(), (ri == itrue).float().mean().item()


def main():
    print("="*64); print("B' sur VRAIS LOBES — chiffres parlés + écrits → AMV canonique → cœur"); print("="*64)
    canon = canonical_dict()
    print("  Chargement données réelles...", flush=True)
    Xa, Ya = load_audio_digits(120); Xi, Yi = load_image_digits(150)
    print(f"  Audio (SpeechCommands digits) : {Xa.shape} | Image (scikit digits) : {Xi.shape}\n", flush=True)
    Xa, Ya = Xa.to(DEVICE), Ya.to(DEVICE); Xi, Yi = Xi.to(DEVICE), Yi.to(DEVICE)

    audio_l = AudioLobe().to(DEVICE); image_l = ImageLobe().to(DEVICE)
    print("  Entraînement lobes SÉPARÉMENT (1-cos vers canonique)...", flush=True)
    train_lobe(audio_l, Xa, Ya, canon); train_lobe(image_l, Xi, Yi, canon)
    aa, ia = lobe_acc(audio_l, Xa, Ya, canon), lobe_acc(image_l, Xi, Yi, canon)
    print(f"  Lobe AUDIO (parlé) perçoit le chiffre : {aa*100:5.1f}%", flush=True)
    print(f"  Lobe IMAGE (écrit) perçoit le chiffre  : {ia*100:5.1f}%", flush=True)

    xm = cross_modal(audio_l, image_l, Xa, Ya, Xi, Yi, canon)
    print(f"  Cross-modal parlé↔écrit (cos moyen)    : {xm:.3f}", flush=True)

    core = SpectralCoreBlock(d_model=D_MODEL, seq_len=1).to(DEVICE)
    print("  Entraînement cœur SCB SEUL sur AMV canonique...", flush=True)
    train_core(core, canon)
    # UNGATED (AMV brut bruité — l'op amplifie l'erreur)
    ra_u, ri_u = replaceability(core, audio_l, image_l, Xa, Ya, Xi, Yi, canon, gated=False)
    # GATED (snap au canonique avant raisonnement — le cœur raisonne sur AMV propre)
    ra_g, ri_g = replaceability(core, audio_l, image_l, Xa, Ya, Xi, Yi, canon, gated=True)
    print(f"  [REMPLAÇABILITÉ] cœur non-retrainé raisonne sur :", flush=True)
    print(f"    UNGATED (AMV brut)    audio {ra_u*100:5.1f}% | image {ri_u*100:5.1f}%", flush=True)
    print(f"    GATED   (snap canon)  audio {ra_g*100:5.1f}% | image {ri_g*100:5.1f}%  ← gate fiabilise", flush=True)

    print("\n" + "="*64); print("VERDICT B' (vrais lobes) :")
    print(f"  Lobes séparés : audio {aa*100:.0f}% / image {ia*100:.0f}% | cross-modal {xm:.3f}")
    print(f"  Remplaçabilité GATED : cœur raisonne audio {ra_g*100:.0f}% ET image {ri_g*100:.0f}% (non retrainé)")
    ok = min(ra_g, ri_g) > 0.7 and aa > 0.7 and ia > 0.7
    status = "B' VALIDÉE sur vrais lobes ✓ (gate fiabilise)" if ok else "B' partielle"
    print(f"  => {status}")
    print(f"  Gate effect : audio {ra_u*100:.0f}→{ra_g*100:.0f}% (Δ+{(ra_g-ra_u)*100:.0f}pt) | image {ri_u*100:.0f}→{ri_g*100:.0f}% (Δ+{(ri_g-ri_u)*100:.0f}pt)")
    json.dump({"audio_acc": aa, "image_acc": ia, "cross_modal": xm,
               "ungated": {"audio": ra_u, "image": ri_u}, "gated": {"audio": ra_g, "image": ri_g}},
              open("ocm26400/bprime_real_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    main()
