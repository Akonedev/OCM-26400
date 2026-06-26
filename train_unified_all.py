#!/usr/bin/env python3
"""CAPTURE TOUT EN SIMULTANÉ — UN modèle, UN passage, TOUTES modalités en même temps.

PRINCIPE ABSOLU : capturer TOUT en MÊME TEMPS pour les associations.
- texte + phonétique + audio + image + génération
- UN SEUL SpectralCoreBlock partagé par TOUTES les vues
- UN SEUL optimizer, UN SEUL passage d'entraînement
- TOUTES les pertes sommées simultanément (texte + phon + audio + image + gen)
- Les associations cross-modales émergent de la capture simultanée

C'est ÇA "capturer en une fois" : pas N trainings séparés, UN training unifié.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import glob, os, numpy as np, time, json
import soundfile as sf
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.multimodal_encoders import AudioEncoder
from ocm26400.amv import D_MODEL, PART
from ocm26400.learned_vocab import LearnedVocab

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
T_AUDIO = 8000
IMG_DIM = 48  # 4x4x3


def text_feat(word):
    v = np.zeros(PART, dtype=np.float32)
    for c in word.lower():
        v[(ord(c) * 167) % PART] += 1.0
    return v


def phon_feat(word):
    w = word.lower()
    vw = sum(1 for c in w if c in "aeiou")
    cs = len(w) - vw
    pat = "".join("v" if c in "aeiou" else "c" for c in w)[:8]
    v = np.zeros(PART, dtype=np.float32)
    for c in pat: v[(ord(c) * 167) % PART] += 1.0
    v[(vw * 7) % PART] += 1.0
    v[(cs * 11 + PART // 2) % PART] += 1.0
    return v


def load_wav(p):
    y, sr = sf.read(p); y = y.astype(np.float32)
    if y.ndim > 1: y = y.mean(1)
    if len(y) < T_AUDIO: y = np.pad(y, (0, T_AUDIO - len(y)))
    else: y = y[:T_AUDIO]
    return torch.tensor(y)


def img_patches(path):
    im = Image.open(path).convert("RGB").resize((8, 8))
    a = np.array(im, dtype=np.float32) / 255.0
    return np.mean([a[:4,:4].flatten(), a[:4,4:].flatten(),
                    a[4:,:4].flatten(), a[4:,4:].flatten()], axis=0)


class UnifiedModel(nn.Module):
    """UN SEUL modèle — SpectralCoreBlock PARTAGÉ par toutes les modalités.
    Texte + phonétique + audio + image → même core → compréhension unifiée.
    Flow decoder : concept → GÉNÉRER le signal (audio/image)."""
    def __init__(self, n_concepts):
        super().__init__()
        # projections vers l'espace AMV (toutes vers D_MODEL)
        self.text_proj = nn.Linear(PART, D_MODEL)
        self.phon_proj = nn.Linear(PART, D_MODEL)
        self.audio_enc = AudioEncoder(out_dim=D_MODEL)
        self.img_proj = nn.Linear(IMG_DIM, D_MODEL)
        # NOYAU SPECTRAL PARTAGÉ (le cœur unifié)
        self.core = SpectralCoreBlock(d_model=D_MODEL, seq_len=1)
        # tête commune (output → ent partition pour 1-cos)
        self.head = nn.Linear(D_MODEL, PART)
        # flow decoders (génération : concept → créer audio/image)
        self.audio_gen = nn.Sequential(
            nn.Linear(PART + 32 + 1, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 32))
        self.img_gen = nn.Sequential(
            nn.Linear(PART + IMG_DIM + 1, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(), nn.Linear(128, IMG_DIM))

    def forward_view(self, feat, proj):
        """N'importe quelle vue → core → ent (PART)."""
        return self.head(self.core(proj(feat).unsqueeze(1)).squeeze(1))

    def gen_sample(self, decoder, cond, x_dim, n_steps=20, batch_size=1):
        """Flow-matching : concept AMV → intégrer bruit→signal."""
        x = torch.randn(batch_size, x_dim, device=cond.device)
        for i in range(n_steps):
            t = torch.full((batch_size, 1), i / n_steps, device=cond.device)
            v = decoder(torch.cat([cond, x, t], dim=-1))
            x = x + v / n_steps
        return x


def train_unified():
    SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
    IMG_DIR = "/media/akone/SAVENVME2/Datasets/vision_tinyimagenet"

    # --- préparer DONNÉES ---
    # audio (SpeechCommands)
    words = sorted([w for w in os.listdir(SC)
                    if os.path.isdir(os.path.join(SC, w)) and not w.startswith("_")])[:35]
    NW = len(words)
    audio_by_word = {}
    for wi, w in enumerate(words):
        wavs = [load_wav(p) for p in glob.glob(os.path.join(SC, w, "*.wav"))[:200]]
        audio_by_word[wi] = torch.stack(wavs).to(device)
    text_audio = torch.tensor([text_feat(w) for w in words]).to(device)
    phon_audio = torch.tensor([phon_feat(w) for w in words]).to(device)

    # images (tinyimagenet, clustering)
    img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))[:2000]
    all_patches = np.array([img_patches(p) for p in img_paths], dtype=np.float64)
    km = MiniBatchKMeans(n_clusters=10, batch_size=256, random_state=0, n_init=3)
    km.fit(all_patches)
    img_labels = km.predict(all_patches)
    all_patches_f32 = all_patches.astype(np.float32)
    cat_names = ["cat_" + chr(65 + i) for i in range(10)]
    text_img = torch.tensor([text_feat(c) for c in cat_names]).to(device)
    patches_t = torch.tensor(all_patches_f32).to(device)
    labels_t = torch.tensor(img_labels, dtype=torch.long).to(device)

    # espace concept unifié (audio + image)
    N_TOTAL = NW + 10  # 15 audio + 10 image = 25 concepts
    cv = LearnedVocab(n=N_TOTAL, dim=PART, init="ortho" if N_TOTAL <= PART else "random", seed=0)
    cv.freeze()
    canon = cv._matrix().to(device)

    model = UnifiedModel(N_TOTAL).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    # splits
    # audio: 80/20 par mot
    audio_tr, audio_te = {}, {}
    for wi in range(NW):
        n = len(audio_by_word[wi])
        p = torch.randperm(n); n_te = max(1, n // 5)
        audio_te[wi] = p[:n_te]; audio_tr[wi] = p[n_te:]
    # image: 80/20
    iperm = np.random.permutation(len(img_paths)); in_tr = int(len(img_paths) * 0.8)
    img_tr, img_te = iperm[:in_tr], iperm[in_tr:]

    print(f"[UNIFIED] {NW} mots audio + 10 clusters image = {N_TOTAL} concepts", flush=True)
    print(f"  VUES: texte + phonétique + audio + image + génération — TOUT SIMULTANÉ", flush=True)
    print(f"  UN SpectralCoreBlock partagé, UN optimizer, UN passage\n", flush=True)

    t0 = time.time()
    for step in range(20000):
        total_loss = torch.tensor(0.0, device=device)
        n_terms = 0

        # --- VUE AUDIO (texte + phonétique + audio → concept) ---
        wi_batch = torch.randint(0, NW, (24,))
        tgt_a = canon[wi_batch]
        out_ta = model.forward_view(text_audio[wi_batch], model.text_proj)
        out_pa = model.forward_view(phon_audio[wi_batch], model.phon_proj)
        # sample audio
        wavs = []
        for wi in wi_batch.tolist():
            samples = audio_tr[wi]
            j = samples[torch.randint(0, len(samples), (1,)).item()]
            wavs.append(audio_by_word[wi][j])
        out_au = model.forward_view(model.audio_enc(torch.stack(wavs)),
                                   nn.Identity().to(device))  # audio déjà projeté
        total_loss = total_loss + (1 - F.cosine_similarity(out_ta, tgt_a).clamp(-1, 1)).mean()
        total_loss = total_loss + (1 - F.cosine_similarity(out_pa, tgt_a).clamp(-1, 1)).mean()
        total_loss = total_loss + (1 - F.cosine_similarity(out_au, tgt_a).clamp(-1, 1)).mean()
        n_terms += 3

        # --- VUE IMAGE (image + texte catégorie → concept) ---
        ci_batch = torch.randint(0, 10, (24,)) + NW  # offset: image concepts après audio
        ii_batch = torch.tensor([img_tr[np.random.choice(np.where(img_labels[img_tr] == (c - NW))[0])]
                                 if len(np.where(img_labels[img_tr] == (c - NW))[0]) > 0 else img_tr[0]
                                 for c in ci_batch.tolist()])
        tgt_i = canon[ci_batch]
        out_ii = model.forward_view(patches_t[ii_batch], model.img_proj)
        out_ti = model.forward_view(text_img[ci_batch - NW], model.text_proj)
        total_loss = total_loss + (1 - F.cosine_similarity(out_ii, tgt_i).clamp(-1, 1)).mean()
        total_loss = total_loss + (1 - F.cosine_similarity(out_ti, tgt_i).clamp(-1, 1)).mean()
        n_terms += 2

        # --- GÉNÉRATION IMAGE (concept → flow-matching → image créée) ---
        gen_ci = torch.randint(0, 10, (16,)) + NW
        gen_cond = canon[gen_ci]
        gen_idx = torch.tensor([img_tr[np.random.choice(np.where(img_labels[img_tr] == (c - NW))[0])]
                                if len(np.where(img_labels[img_tr] == (c - NW))[0]) > 0 else img_tr[0]
                                for c in gen_ci.tolist()])
        x_real = patches_t[gen_idx]  # cibles réelles
        x_0 = torch.randn_like(x_real)
        t_gen = torch.rand(16, 1, device=device)
        x_t = (1 - t_gen) * x_0 + t_gen * x_real
        v_tgt = x_real - x_0
        v_pred = model.img_gen(torch.cat([gen_cond, x_t, t_gen], dim=-1))
        total_loss = total_loss + F.mse_loss(v_pred, v_tgt)
        n_terms += 1

        # --- GÉNÉRATION AUDIO (concept → flow-matching → audio créé) ---
        gen_wi = torch.randint(0, NW, (16,))
        gen_cond_a = canon[gen_wi]
        # target: Mel features (32-dim) du vrai audio
        with torch.no_grad():
            tmp = model.audio_enc(torch.stack([audio_by_word[wi.item()][audio_tr[wi.item()][0]]
                                               for wi in gen_wi]))
        x_real_a = tmp[:, :32] if tmp.shape[1] >= 32 else F.pad(tmp, (0, 32 - tmp.shape[1]))
        x_0a = torch.randn_like(x_real_a)
        t_gen_a = torch.rand(16, 1, device=device)
        x_ta = (1 - t_gen_a) * x_0a + t_gen_a * x_real_a
        v_tgt_a = x_real_a - x_0a
        v_pred_a = model.audio_gen(torch.cat([gen_cond_a, x_ta, t_gen_a], dim=-1))
        total_loss = total_loss + F.mse_loss(v_pred_a, v_tgt_a)
        n_terms += 1

        loss = total_loss / n_terms
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 5000 == 0:
            model.eval()
            with torch.no_grad():
                # audio classification
                ok_a = sum(1 for wi in range(NW) for j in audio_te[wi][:2]
                           if (model.forward_view(model.audio_enc(audio_by_word[wi][j:j+1]),
                                                  nn.Identity().to(device)) @ canon.t()).argmax(1).item() == wi)
                n_a = sum(len(audio_te[wi][:2]) for wi in range(NW))
                # image classification
                ok_i = sum(1 for i in img_te[:200]
                           if (model.forward_view(patches_t[i:i+1], model.img_proj) @ canon.t()).argmax(1).item() == img_labels[i] + NW)
                # image generation verification
                gen_ok = 0
                for ci in range(10):
                    cond = canon[ci + NW:ci + NW + 1].expand(3, -1)
                    gen = model.gen_sample(model.img_gen, cond, IMG_DIM, n_steps=20, batch_size=3)
                    pred = (model.forward_view(gen, model.img_proj) @ canon.t()).argmax(1)
                    gen_ok += (pred == (ci + NW)).sum().item()
            print(f"  step {step} loss={loss.item():.4f} | audio={ok_a}/{n_a} "
                  f"img={ok_i}/100 gen={gen_ok}/30 t={time.time()-t0:.0f}s", flush=True)
            model.train()

    # éval finale
    model.eval()
    with torch.no_grad():
        ok_a = sum(1 for wi in range(NW) for j in audio_te[wi]
                   if (model.forward_view(model.audio_enc(audio_by_word[wi][j:j+1]),
                                          nn.Identity().to(device)) @ canon.t()).argmax(1).item() == wi)
        n_a = sum(len(audio_te[wi]) for wi in range(NW))
        ok_i = sum(1 for i in img_te
                   if (model.forward_view(patches_t[i:i+1], model.img_proj) @ canon.t()).argmax(1).item() == img_labels[i] + NW)
        gen_ok = sum(1 for ci in range(10)
                     for _ in range(5)
                     if (model.forward_view(
                         model.gen_sample(model.img_gen, canon[ci+NW:ci+NW+1], IMG_DIM, 20, 1),
                         model.img_proj) @ canon.t()).argmax(1).item() == ci + NW)
    print(f"\n{'='*60}")
    print(f"CAPTURE TOUT EN SIMULTANÉ — RÉSULTATS UNIFIÉS")
    print(f"{'='*60}")
    print(f"  AUDIO classification: {ok_a}/{n_a} = {ok_a/max(n_a,1)*100:.1f}%")
    print(f"  IMAGE classification: {ok_i}/{len(img_te)} = {ok_i/max(len(img_te),1)*100:.1f}%")
    print(f"  IMAGE génération (concept→créé, vérifié): {gen_ok}/50 = {gen_ok/50*100:.0f}%")
    print(f"  temps: {time.time()-t0:.0f}s, 5000 steps")
    print(f"  méthode: UN modèle, UN core, TOUT simultané (texte+phon+audio+image+génération)")

    ckpt = "/media/akone/SAVENVME2/Datasets/ocm26400/unified_all_trained.pt"
    torch.save({"model_state": model.state_dict(), "canon": canon,
                "audio_acc": ok_a/max(n_a,1), "img_acc": ok_i/max(len(img_te),1),
                "gen_acc": gen_ok/50,
                "words": words, "method": "unified simultaneous all modalities"}, ckpt)
    print(f"  [SAUVÉ] {ckpt}")


if __name__ == "__main__":
    train_unified()
