#!/usr/bin/env python3
"""CAPTURE SIMULTANÉE multi-lobes en 1 passe → ASSOCIATIONS (L6) — test+validation.

Règle primordiale (Besoins) : capturer TOUT en une fois, en même temps (toutes modalités +
primitives), pour optimiser les ASSOCIATIONS. Le cœur SCB reçoit la SUPERPOSITION spectrale
des IDs/AMV de tous les lobes en UNE passe → associations (L6 : 1-source direct, multi-source).

Ici : un concept (chiffre) est capturé par 3 lobes en PARALLÈLE (texte mot + audio Mel +
image patch). Chacun émet son AMV canonique. La capture simultanée = superposition (somme)
des AMV → le cœur SCB en 1 passe → associe (= retrouve le concept).

Validations :
  1. Superposition (3 modalités) ≥ meilleure modalité seule → les associations aident.
  2. COMPLÉTION cross-modale : capture partielle (2/3 modalités) → le cœur retrouve le concept
     (association/comble l'info manquante).
  3. Pas de régression : la superposition ne dégrade jamais vs la meilleure modalité seule.
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
    return torch.linalg.qr(C.T)[0].T   # (P, PART) figé


def load_audio(n_per=120):
    X, Y = [], []
    for di, d in enumerate(DIGITS):
        for p in sorted(glob.glob(os.path.join(SC, d, "*.wav")))[:n_per]:
            X.append(load_wav(p, T).numpy()); Y.append(di)
    return torch.tensor(np.stack(X)), torch.tensor(Y)

def load_image(n_per=150):
    X, Y = load_digits(return_X_y=True); Xl, Yl = [], []
    for di in range(P):
        for i in np.where(Y == di)[0][:n_per]: Xl.append(X[i]); Yl.append(di)
    return torch.tensor(np.stack(Xl)).float(), torch.tensor(Yl)


# ---- 3 lobes (texte mot + audio + image), chacun émet l'ent canonique ----
class TextLobe(nn.Module):    # mot → ent canonique (le texte est déjà discret)
    def __init__(self): super().__init__(); self.emb = nn.Embedding(P, PART)
    def forward(self, word_id): return self.emb(word_id)

class AudioLobe(nn.Module):
    def __init__(self): super().__init__(); self.enc = AudioEncoder(out_dim=PART)
    def forward(self, wav): return self.enc(wav)

class ImageLobe(nn.Module):
    def __init__(self): super().__init__(); self.net = nn.Sequential(nn.Linear(64,128), nn.ReLU(), nn.Linear(128,PART))
    def forward(self, img): return self.net(img)


def train_lobe(lobe, X, Y, canon, steps=1200, lr=3e-3, bs=64):
    opt = torch.optim.Adam(lobe.parameters(), lr=lr); N = len(X)
    for _ in range(steps):
        idx = torch.randint(0, N, (bs,))
        loss = (1 - F.cosine_similarity(lobe(X[idx]), canon[Y[idx]], dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()

def align_acc(lobe, X, Y, canon):
    lobe.eval()
    with torch.no_grad(): pred = (lobe(X) @ canon.t()).argmax(1)
    lobe.train(); return (pred == Y).float().mean().item()


def superpose_and_associate(core, ents, canon):
    """Capture simultanée : superposition (somme) des ent de plusieurs lobes → cœur 1 passe → concept.
    ents = liste de (B, PART). Retourne le concept prédit (argmax vs canon)."""
    sup = sum(ents)                               # superposition spectrale des AMV (L6)
    amv = torch.zeros(sup.shape[0], D_MODEL, device=DEVICE); amv[:, 0:PART] = sup
    out = core(amv.unsqueeze(1)).squeeze(1)[:, 0:PART]
    return (out @ canon.t()).argmax(1)

def train_core_association(core, lobes, Xs, Y, canon, steps=1500, lr=3e-3):
    """Entraîne le cœur à ASSOCIER : superposition des AMV (3 modalités) → concept canonique.
    1-cos (cœur raisonnement). Le cœur apprend à fusionner la superposition en 1 passe."""
    opt = torch.optim.Adam(core.parameters(), lr=lr); N = len(Y)
    for _ in range(steps):
        idx = torch.randint(0, N, (64,))
        ents = [lobe(X[idx]) for lobe, X in zip(lobes, Xs)]   # 3 ent en parallèle
        sup = sum(ents); amv = torch.zeros(64, D_MODEL, device=DEVICE); amv[:, 0:PART] = sup
        out = core(amv.unsqueeze(1)).squeeze(1)[:, 0:PART]
        loss = (1 - F.cosine_similarity(out, canon[Y[idx]], dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()


def main():
    print("="*64); print("CAPTURE SIMULTANÉE multi-lobes (1 passe) → ASSOCIATIONS (L6)"); print("="*64)
    canon = canonical_dict()
    print("  Chargement (texte mot + audio + image)...", flush=True)
    Xa, Ya = load_audio(120); Xi, Yi = load_image(150); Xt = torch.arange(P, device=DEVICE)
    n = min(len(Xa), len(Xi)); Xa, Ya, Xi, Yi = Xa[:n].to(DEVICE), Ya[:n].to(DEVICE), Xi[:n].to(DEVICE), Yi[:n].to(DEVICE)
    print(f"  {n} exemples × 3 modalités\n", flush=True)

    text_l, audio_l, image_l = TextLobe().to(DEVICE), AudioLobe().to(DEVICE), ImageLobe().to(DEVICE)
    print("  Entraînement lobes SÉPARÉMENT...", flush=True)
    train_lobe(audio_l, Xa, Ya, canon); train_lobe(image_l, Xi, Yi, canon)
    # texte : mappage direct mot→canon (déjà discret)
    with torch.no_grad(): text_l.emb.weight.copy_(canon)
    ta, aa, ia = align_acc(text_l, Xt, Xt, canon), align_acc(audio_l, Xa, Ya, canon), align_acc(image_l, Xi, Yi, canon)
    print(f"  Lobe seul : texte {ta*100:.0f}% | audio {aa*100:.0f}% | image {ia*100:.0f}%", flush=True)

    # cœur : associer la superposition (3 modalités) → concept
    core = SpectralCoreBlock(d_model=D_MODEL, seq_len=1).to(DEVICE)
    Xtext_per_ex = Ya  # l'id du mot = le concept (texte parfait)
    print("  Entraînement cœur (associer superposition 3-modalités → concept, 1 passe)...", flush=True)
    train_core_association(core, [text_l, audio_l, image_l], [Xtext_per_ex, Xa, Xi], Ya, canon)

    # ---- TESTS capture simultanée ----
    nte = 400; idx = torch.randperm(n, device=DEVICE)[:nte]
    Yte = Ya[idx]
    with torch.no_grad():
        et = text_l(Yte); ea = audio_l(Xa[idx]); ei = image_l(Xi[idx])
    core.eval()
    with torch.no_grad():
        # chaque modalité seule (via le cœur, mais 1 seul ent)
        r_text = (superpose_and_associate(core, [et], canon) == Yte).float().mean().item()
        r_aud = (superpose_and_associate(core, [ea], canon) == Yte).float().mean().item()
        r_img = (superpose_and_associate(core, [ei], canon) == Yte).float().mean().item()
        # superposition 3 modalités (CAPTURE SIMULTANÉE)
        r_all = (superpose_and_associate(core, [et, ea, ei], canon) == Yte).float().mean().item()
        # complétion : 2/3 modalités (partial → association)
        r_ta = (superpose_and_associate(core, [et, ea], canon) == Yte).float().mean().item()
        r_ti = (superpose_and_associate(core, [et, ei], canon) == Yte).float().mean().item()
        r_ai = (superpose_and_associate(core, [ea, ei], canon) == Yte).float().mean().item()
    print(f"\  [CAPTURE] modalité seule : texte {r_text*100:.0f}% | audio {r_aud*100:.0f}% | image {r_img*100:.0f}%", flush=True)
    print(f"  [CAPTURE SIMULTANÉE 3 modalités] : {r_all*100:.0f}%  ← superposition 1 passe", flush=True)
    print(f"  [COMPLÉTION 2/3] : texte+audio {r_ta*100:.0f}% | texte+image {r_ti*100:.0f}% | audio+image {r_ai*100:.0f}%", flush=True)

    best_single = max(r_text, r_aud, r_img)
    print("\n" + "="*64); print("VERDICT capture simultanée + associations :")
    print(f"  Meilleure modalité seule : {best_single*100:.0f}%")
    print(f"  Superposition 3 modalités : {r_all*100:.0f}%  (Δ{(r_all-best_single)*100:+.0f}pt)")
    no_reg = r_all >= best_single - 0.02
    assoc_helps = r_all > best_single + 0.02
    print(f"  Pas de régression : {'✓' if no_reg else '✗'} | Associations aident : {'✓' if assoc_helps else '≈'}")
    if no_reg and assoc_helps:
        print("  => CAPTURE SIMULTANÉE VALIDÉE ✓ : la superposition 1-passe → associations (+ proof).")
    elif no_reg:
        print("  => Capture validée (pas de régression). Association neutre ici (modalités déjà fortes).")
    else:
        print("  => Régression ! La superposition dégrade — à corriger.")
    json.dump({"single": {"text": r_text, "audio": r_aud, "image": r_img}, "all3": r_all,
               "partial": {"ta": r_ta, "ti": r_ti, "ai": r_ai}}, open("ocm26400/capture_simultanee_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    main()
