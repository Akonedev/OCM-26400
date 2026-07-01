#!/usr/bin/env python3
"""IMPLÉMENTATION B' — lobes SÉPARÉS → AMV-256 canonique partagé → cœur SCB unifié.

Architecture CANONIQUE OCM-26400 (décision experts + Shukor 2025 + biologie) :
  [MODALITÉ] → [LOBE spectral mince SÉPARÉ] → [AMV-256 canonique: ent|prop|op|meta] → [CŒUR SCB]
                                          ↕ interface AMV STANDARD = lobes REMPLAÇABLES sans retrain du cœur

Démontre les 3 propriétés B' (vs Frankenstein/OmniModel-joint) :
  1. Lobes SÉPARÉS : 2 front-ends DIFFÉRENTS (audio Conv1d + image patch), chacun entraîné SEUL.
  2. AMV-256 STANDARD : les 2 lobes émettent le MÊME AMV canonique (ent = vecteur canonique du concept,
     dictionnaire figé orthogonal partagé). Alignement 1-cos vers le canonique.
  3. REMPLAÇABILITÉ : le cœur (entraîné UNE fois sur l'AMV canonique) raisonne correctement
     quel que soit le lobe qui a émis l'AMV (audio OU image) — SANS retrain. C'est le test décisif.

De +, alignement CROSS-MODAL : audio(c) et image(c) → même AMV canonique (cos ≈ 1) → le cœur
est agnostique à la modalité.
"""
import torch, torch.nn as nn, torch.nn.functional as F, json
from ocm26400.spectral_core import SpectralCoreBlock
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
P = 11; PART = 64; D_MODEL = 256; AUD_LEN = 800; IMG = 16


# ===================== AMV-256 CANONIQUE (figé, partagé) =====================
def canonical_dict():
    """Dictionnaire AMV canonique figé : P vecteurs orthogonaux dans le slot ent [0:PART].
    PARTAGÉ par tous les lobes = l'interface standard B'."""
    g = torch.Generator(device=DEVICE).manual_seed(42)
    C = torch.randn(P, PART, device=DEVICE, generator=g)
    C = torch.linalg.qr(C.T)[0].T   # orthogonalisation → P vecteurs ortho dans PART dims
    return C  # (P, PART) — figé (aucun grad)


def to_amv(ent_part):
    """ent_part (B, PART) → AMV-256 [ent|prop|op|meta] (ent rempli, reste 0)."""
    amv = torch.zeros(ent_part.shape[0], D_MODEL, device=ent_part.device)
    amv[:, 0:PART] = ent_part
    return amv


# ===================== DONNÉES SYNTHÉTIQUES (2 modalités, P concepts) =====================
def gen_audio(c, n):
    """Concept c → signature audio (sine à freq c + bruit). (n, AUD_LEN)"""
    t = torch.arange(AUD_LEN, device=DEVICE).float() / 100       # (AUD_LEN,)
    freq = (2.0 + 0.8 * c.float()).unsqueeze(1)                  # (n, 1)
    base = torch.sin(2 * 3.14159 * freq * t.unsqueeze(0))        # (n, AUD_LEN)
    return base + 0.3 * torch.randn(n, AUD_LEN, device=DEVICE)

def gen_image(c, n):
    """Concept c → signature image (motif c). (n, 1, IMG, IMG)"""
    x = torch.arange(IMG, device=DEVICE).float()
    grid = x.unsqueeze(0) + x.unsqueeze(1)                      # (IMG, IMG)
    cf = (0.6 * c.float()).view(-1, 1, 1)                       # (n, 1, 1)
    pattern = torch.sin(cf * grid).unsqueeze(1)                 # (n, 1, IMG, IMG)
    return pattern + 0.3 * torch.randn(n, 1, IMG, IMG, device=DEVICE)


# ===================== LOBES SÉPARÉS (front-ends DIFFÉRENTS, MÊME sortie AMV) =====================
class AudioLobe(nn.Module):
    """Front-end audio Conv1d (mince) → AMV head → ent canonique. Entraîné SEUL (1-cos vers canon)."""
    def __init__(self):
        super().__init__()
        self.front = nn.Sequential(nn.Conv1d(1, 32, 80, stride=16), nn.ReLU(), nn.Conv1d(32, 32, 3), nn.ReLU())
        self.head = nn.Linear(32 * ((AUD_LEN - 80) // 16 - 1), PART)
    def forward(self, wav):  # (B, AUD_LEN) → ent (B, PART)
        h = self.front(wav.unsqueeze(1)).flatten(1)
        return self.head(h)

class ImageLobe(nn.Module):
    """Front-end image patch (mince) → AMV head → ent canonique. Entraîné SEUL (1-cos vers canon)."""
    def __init__(self):
        super().__init__()
        self.front = nn.Sequential(nn.Flatten(), nn.Linear(IMG*IMG, 128), nn.ReLU())
        self.head = nn.Linear(128, PART)
    def forward(self, img):  # (B,1,IMG,IMG) → ent (B, PART)
        return self.head(self.front(img))


# ===================== CŒUR UNIFIÉ (SCB, raisonne sur l'AMV) =====================
def core_reason(core, ent_a, ent_b):
    """Cœur SCB sur l'AMV : (ent_a, ent_b) → ent résultat. crown-jewel sur AMV canonique."""
    amv = torch.zeros(ent_a.shape[0], D_MODEL, device=DEVICE)
    amv[:, 0:PART] = ent_a; amv[:, PART:2*PART] = ent_b   # ent + prop slots
    out = core(amv.unsqueeze(1)).squeeze(1)[:, 0:PART]    # ent du résultat
    return out


# ===================== ENTRAÎNEMENTS SÉPARÉS =====================
def train_lobe(lobe, gen_fn, canon, steps=800, lr=3e-3):
    """Entraîne le lobe SÉPARÉMENT : percevoir le concept → émettre son ent canonique (1-cos)."""
    opt = torch.optim.Adam(lobe.parameters(), lr=lr)
    for _ in range(steps):
        c = torch.randint(0, P, (64,), device=DEVICE)
        x = gen_fn(c, 64)
        ent = lobe(x)
        tgt = canon[c]
        loss = (1 - F.cosine_similarity(ent, tgt, dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()

def train_core(core, canon, steps=1500, lr=3e-3):
    """Entraîne le cœur SEUL sur l'AMV canonique : op(ent_a, ent_b) = canon[(a+b)%P]. crown-jewel 1-cos."""
    opt = torch.optim.Adam(core.parameters(), lr=lr)
    for _ in range(steps):
        a = torch.randint(0, P, (64,), device=DEVICE); b = torch.randint(0, P, (64,), device=DEVICE)
        out = core_reason(core, canon[a], canon[b])
        tgt = canon[(a + b) % P]
        loss = (1 - F.cosine_similarity(out, tgt, dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()


# ===================== TESTS B' =====================
def lobe_acc(lobe, gen_fn, canon, n=500):
    """Lobe perçoit-il le bon concept ? (argmax cosine vs canon)."""
    lobe.eval()
    with torch.no_grad():
        c = torch.arange(P, device=DEVICE).repeat_interleave(n // P + 1)[:n]
        ent = lobe(gen_fn(c, n))
        pred = (ent @ canon.t()).argmax(1)
    lobe.train()
    return (pred == c).float().mean().item()

def cross_modal_align(audio_l, image_l, canon, n=500):
    """audio(c) et image(c) → même AMV canonique ? (cos entre les 2 ent)."""
    audio_l.eval(); image_l.eval()
    with torch.no_grad():
        c = torch.arange(P, device=DEVICE).repeat_interleave(n // P + 1)[:n]
        ea = audio_l(gen_audio(c, n)); ei = image_l(gen_image(c, n))
        cos = F.cosine_similarity(ea, ei, dim=-1).mean().item()
    audio_l.train(); image_l.train()
    return cos

def replaceability_test(core, audio_l, image_l, canon, n=1000):
    """TEST DÉCISIF B' : le cœur (non retrainé) raisonne-t-il sur l'AMV émis par l'AUTRE lobe ?
    On nourrit le cœur avec l'AMV du lobe audio OU image → le résultat (op) doit être correct
    dans les 2 cas (le cœur est agnostique au front-end)."""
    core.eval(); audio_l.eval(); image_l.eval()
    g = torch.Generator(device=DEVICE).manual_seed(7)
    a = torch.randint(0, P, (n,), generator=g, device=DEVICE); b = torch.randint(0, P, (n,), generator=g, device=DEVICE)
    true = (a + b) % P
    with torch.no_grad():
        # lobe audio émet les AMV de a et b → cœur raisonne
        ea = audio_l(gen_audio(a, n)); eb = audio_l(gen_audio(b, n))
        ra = (core_reason(core, ea, eb) @ canon.t()).argmax(1)
        # lobe image émet les AMV → cœur (MÊME, non retrainé) raisonne
        ei_a = image_l(gen_image(a, n)); ei_b = image_l(gen_image(b, n))
        ri = (core_reason(core, ei_a, ei_b) @ canon.t()).argmax(1)
    core.train(); audio_l.train(); image_l.train()
    return (ra == true).float().mean().item(), (ri == true).float().mean().item()


def main():
    print("="*64); print("IMPLÉMENTATION B' — lobes séparés → AMV canonique → cœur unifié"); print("="*64)
    print(f"  {P} concepts, 2 front-ends DIFFÉRENTS (audio Conv1d + image patch), AMV-256 canonique figé\n")
    canon = canonical_dict()

    # 1. Entraîner les lobes SÉPARÉMENT
    audio_l = AudioLobe().to(DEVICE); image_l = ImageLobe().to(DEVICE)
    print("  Entraînement lobes SÉPARÉMENT (1-cos vers canonique)...", flush=True)
    train_lobe(audio_l, gen_audio, canon)
    train_lobe(image_l, gen_image, canon)
    aa, ia = lobe_acc(audio_l, gen_audio, canon), lobe_acc(image_l, gen_image, canon)
    print(f"  Lobe audio perçoit le concept : {aa*100:5.1f}%", flush=True)
    print(f"  Lobe image perçoit le concept  : {ia*100:5.1f}%", flush=True)

    # 2. Alignement cross-modal (audio et image → même AMV ?)
    xm = cross_modal_align(audio_l, image_l, canon)
    print(f"  Alignement cross-modal audio↔image (cos) : {xm:.3f}  (1.0 = parfait)", flush=True)

    # 3. Entraîner le cœur SEUL sur l'AMV canonique (crown-jewel)
    core = SpectralCoreBlock(d_model=D_MODEL, seq_len=1).to(DEVICE)
    print("  Entraînement cœur SCB SEUL sur AMV canonique (crown-jewel op)...", flush=True)
    train_core(core, canon)

    # 4. TEST DÉCISIF : remplaçabilité
    ra, ri = replaceability_test(core, audio_l, image_l, canon)
    print("  [TEST REMPLAÇABILITÉ] cœur non-retrainé raisonne sur :", flush=True)
    print(f"    AMV du lobe AUDIO : op correct à {ra*100:5.1f}%", flush=True)
    print(f"    AMV du lobe IMAGE : op correct à {ri*100:5.1f}%  (MÊME cœur, lobe permuté)", flush=True)

    print("\n" + "="*64); print("VERDICT B' :")
    print(f"  1. Lobes SÉPARÉS (audio {aa*100:.0f}% / image {ia*100:.0f}% perçus) ✓")
    print(f"  2. AMV canonique STANDARD : cross-modal align = {xm:.3f} {'✓' if xm>0.9 else '≈'}")
    print(f"  3. REMPLAÇABILITÉ : cœur (non retrainé) raisonne audio={ra*100:.0f}% ET image={ri*100:.0f}% {'✓' if min(ra,ri)>0.7 else '≈'}")
    if min(ra, ri) > 0.7 and xm > 0.85:
        print("\n  => ARCHITECTURE B' VALIDÉE ✓")
        print("     Lobes séparés (front-ends différents) → AMV canonique partagé → cœur unifié.")
        print("     Le cœur est agnostique au lobe (remplaçable sans retrain). PAS de Frankenstein.")
    else:
        print("\n  => B' partiel (alignement/replaceabilité à renforcer).")
    json.dump({"audio_acc": aa, "image_acc": ia, "cross_modal_cos": xm,
               "reason_via_audio": ra, "reason_via_image": ri},
              open("ocm26400/bprime_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    main()
