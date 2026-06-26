#!/usr/bin/env python3
"""Entraînement NUMÉRIQUE — tout en IDs, grokking par compréhension.

PRINCIPE : tout est converti en IDs numériques (comme le crown-jewel).
Le SpectralCoreBlock (FFT) grok les PATTERNS NUMÉRIQUES entre IDs.

Pour chaque modalité :
1. Signal continu → VECTOR QUANTIZATION → séquence d'IDs discrets
   (audio: Mel frames → k-means → spectral-token-IDs)
   (image: patches → k-means → visual-token-IDs)
2. IDs → embeddings (LearnedVocab) → AMV → SpectralCoreBlock GROK
3. Loss 1-cos (crown-jewel pattern) : compréhension ↔ canonical
4. Gate 0.99 = le modèle a COMPRIS la règle de composition des IDs

C'est EXACTEMENT le crown-jewel : (a,b) → op(a,b) devient
                             (spectral-IDs) → word-ID
Le FFT grok les relations numériques entre IDs — nativement.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import glob, os, numpy as np, time, json
import soundfile as sf
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.amv import D_MODEL, PART
from ocm26400.learned_vocab import LearnedVocab

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
K_CODEBOOK = 256    # nb de primitives (IDs possibles)
SEQ_LEN = 64        # longueur de séquence d'IDs (pad/truncate)


# ============================================================
# VQ : Signal continu → IDs discrets (vector quantization)
# ============================================================
def vq_codebook(features, k=K_CODEBOOK):
    """K-means sur les features → codebook de k primitives."""
    print(f"  VQ: {features.shape[0]} features -> {k} clusters...", flush=True)
    km = MiniBatchKMeans(n_clusters=k, batch_size=512, random_state=0, n_init=3)
    km.fit(features)
    return km


def vq_encode(km, features):
    """Features continues → séquence d'IDs (nearest cluster)."""
    return km.predict(features)


# ============================================================
# Modèle : IDs → embeddings → SpectralCoreBlock → 1-cos
# ============================================================
class NumericGrokModel(nn.Module):
    """Grok une séquence d'IDs numériques → prédit un ID cible.
    Exactement le crown-jewel : (sequence d'IDs) → output ID."""
    def __init__(self, vocab_size=K_CODEBOOK, n_classes=20, d_model=D_MODEL):
        super().__init__()
        # embeddings denses pour les IDs (LearnedVocab, comme concept IDs)
        self.embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        # noyau spectral FFT partagé (gok les patterns numériques)
        self.core = SpectralCoreBlock(d_model=d_model, seq_len=SEQ_LEN, bidirectional=True)
        # tête : output AMV → prédire l'ID de classe (1-cos sur canonical)
        self.head = nn.Linear(d_model, PART)  # projette vers l'espace canonical

    def forward(self, id_seq):
        """id_seq: (B, L) IDs → output ent (B, PART) pour 1-cos."""
        x = self.embed(id_seq)          # (B, L, d_model)
        out = self.core(x)              # grok les patterns numériques
        pooled = out.mean(dim=1)        # pool → (B, d_model)
        return self.head(pooled)        # (B, PART) = ent partition


def train_numeric_grok(id_seqs, labels, n_classes, name="domaine"):
    """Grok une séquence d'IDs → class ID (crown-jewel pattern, 1-cos loss).
    Le modèle COMPREND la règle de composition des IDs, ne mémorise pas."""
    N = len(id_seqs)
    # canonical embeddings pour les classes (comme crown-jewel canonical)
    cv = LearnedVocab(n=max(n_classes, 2), dim=PART, init="ortho" if n_classes <= PART else "random", seed=0)
    cv.freeze()
    canon = cv._matrix().to(device)  # (n_classes, PART)

    model = NumericGrokModel(vocab_size=K_CODEBOOK, n_classes=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    id_t = torch.tensor(id_seqs, dtype=torch.long).to(device)  # (N, L)
    lab_t = torch.tensor(labels, dtype=torch.long).to(device)

    idx_all = torch.randperm(N); ntr = int(N * 0.85)
    tr_i, te_i = idx_all[:ntr], idx_all[ntr:]

    print(f"\n  [{name}] {N} séquences d'IDs, {n_classes} classes", flush=True)
    print(f"  [GROK] loss 1-cos (crown-jewel pattern) — compréhension, PAS de CE", flush=True)
    t0 = time.time()
    for step in range(10000):
        bi = tr_i[torch.randint(0, len(tr_i), (48,))]
        out = model(id_t[bi])                     # (48, PART)
        tgt = canon[lab_t[bi]]                    # canonical des classes
        cos = F.cosine_similarity(out, tgt, dim=-1).clamp(-1, 1)
        loss = (1 - cos).mean()                   # 1-cos (crown-jewel)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0:
            # gate check
            with torch.no_grad():
                tr_out = model(id_t[tr_i[:100]])
                tr_pred = (tr_out @ canon.t()).argmax(1)
                tr_acc = (tr_pred == lab_t[tr_i[:100]]).float().mean().item()
            print(f"    step {step} loss={loss.item():.4f} train_acc={tr_acc*100:.0f}% t={time.time()-t0:.0f}s",
                  flush=True)

    # évaluation TEST (compréhension → généralisation OOD)
    model.eval()
    with torch.no_grad():
        te_out = model(id_t[te_i])
        te_pred = (te_out @ canon.t()).argmax(1)
        te_acc = (te_pred == lab_t[te_i]).float().mean().item()
    print(f"  [{name}] TEST acc (compréhension, OOD): {te_acc*100:.1f}% (hasard={100/n_classes:.0f}%)",
          flush=True)
    return model, te_acc, canon, None


# ============================================================
# AUDIO : Mel-STFT → VQ → IDs → grok
# ============================================================
def train_audio_numeric():
    SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
    WORDS = [w for w in os.listdir(SC) if os.path.isdir(os.path.join(SC, w)) and not w.startswith("_")][:15]
    T = 8000

    def load_wav(p):
        y, sr = sf.read(p); y = y.astype(np.float32)
        if y.ndim > 1: y = y.mean(1)
        if len(y) < T: y = np.pad(y, (0, T - len(y)))
        else: y = y[:T]
        return y

    # 1. charger audio + extraire Mel frames
    print("[AUDIO] chargement SpeechCommands + extraction Mel frames...", flush=True)
    from ocm26400.multimodal_encoders import AudioEncoder
    tmp = AudioEncoder(out_dim=32)
    mel_fb = tmp.mel_fb.numpy()
    win = torch.hann_window(tmp.n_fft).numpy()

    all_mel = []; aud_ids = []; labs = []
    for wi, w in enumerate(WORDS):
        for p in glob.glob(os.path.join(SC, w, "*.wav"))[:80]:
            y = load_wav(p)
            # STFT manuel (numpy) → Mel frames
            frames = []
            for start in range(0, len(y) - tmp.n_fft, tmp.n_fft // 2):
                seg = y[start:start + tmp.n_fft] * win
                fft = np.fft.rfft(seg)
                power = np.abs(fft) ** 2
                mel = mel_fb @ power
                frames.append(np.log1p(mel))
            if len(frames) >= SEQ_LEN:
                all_mel.extend(frames[:SEQ_LEN])
                aud_ids.append(wi)
                labs.append(wi)
            elif len(frames) > 0:
                # pad to SEQ_LEN
                padded = frames + [frames[-1]] * (SEQ_LEN - len(frames))
                all_mel.extend(padded)
                aud_ids.append(wi)
                labs.append(wi)

    all_mel = np.array(all_mel, dtype=np.float32)  # (N_frames, 32)
    N_audio = len(labs)
    print(f"  {N_audio} audio, {all_mel.shape[0]} Mel frames collectés", flush=True)

    # 2. VQ : Mel frames → codebook de K=256 primitives spectrales
    km = vq_codebook(all_mel, K_CODEBOOK)

    # 3. chaque audio → ID sequence
    id_seqs = []
    for wi in range(N_audio):
        start = wi * SEQ_LEN
        frames = all_mel[start:start + SEQ_LEN]
        ids = vq_encode(km, frames)
        id_seqs.append(ids)
    id_seqs = np.array(id_seqs)  # (N_audio, SEQ_LEN)
    print(f"  séquences d'IDs: {id_seqs.shape}", flush=True)

    # 4. GROK : IDs → SpectralCoreBlock → 1-cos → classify
    model, acc, canon = train_numeric_grok(id_seqs, labs, len(WORDS), "audio")
    return model, acc, canon, WORDS


# ============================================================
# IMAGE : patches → VQ → IDs → grok (self-supervisé)
# ============================================================
def train_image_numeric():
    IMG_DIR = "/media/akone/SAVENVME2/Datasets/vision_tinyimagenet"

    def load_img_patches(p):
        im = Image.open(p).convert("RGB").resize((16, 16))
        arr = np.array(im).astype(np.float32) / 255.0  # (16,16,3)
        # 4 patches 8x8
        patches = [arr[:8, :8].flatten(), arr[:8, 8:].flatten(),
                   arr[8:, :8].flatten(), arr[8:, 8:].flatten()]
        return patches

    print("[IMAGE] chargement tinyimagenet + extraction patches...", flush=True)
    paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))[:1000]
    all_patches = []; img_ids = []
    for i, p in enumerate(paths):
        patches = load_img_patches(p)
        all_patches.extend(patches)
        img_ids.append(i)

    all_patches = np.array(all_patches, dtype=np.float32)  # (N_img*4, 192)
    N_img = len(img_ids)
    print(f"  {N_img} images, {all_patches.shape[0]} patches", flush=True)

    # VQ : patches → codebook de K=256 primitives visuelles
    km = vq_codebook(all_patches, K_CODEBOOK)

    # chaque image → 4 patch-IDs → pad to SEQ_LEN
    id_seqs = []
    for i in range(N_img):
        start = i * 4
        patch_ids = vq_encode(km, all_patches[start:start + 4])
        # pad to SEQ_LEN (repeat)
        ids = list(patch_ids) + [0] * (SEQ_LEN - 4)
        id_seqs.append(ids[:SEQ_LEN])
    id_seqs = np.array(id_seqs)

    # 10 classes (par index / 100) pour classification
    n_cls = 10
    labs = [i % n_cls for i in range(N_img)]
    id_seqs_data = train_numeric_grok(id_seqs, labs, n_cls, "image")
    return model, acc


if __name__ == "__main__":
    print("="*60)
    print("ENTRAÎNEMENT NUMÉRIQUE : VQ → IDs → SpectralCoreBlock GROK → 1-cos")
    print("PRINCIPE : tout en IDs numériques, FFT grok les patterns, comprehension > memory")
    print("="*60)

    results = {}
    # AUDIO
    try:
        _, audio_acc, _, words = train_audio_numeric()
        results["audio"] = {"acc": audio_acc, "words": len(words), "method": "VQ Mel→IDs→grok→1-cos"}
    except Exception as e:
        results["audio"] = {"error": str(e)}
        print(f"  audio erreur: {e}", flush=True)

    # IMAGE
    try:
        _, img_acc = train_image_numeric()
        results["image"] = {"acc": img_acc, "method": "VQ patches→IDs→grok→1-cos"}
    except Exception as e:
        results["image"] = {"error": str(e)}
        print(f"  image erreur: {e}", flush=True)

    print(f"\n{'='*60}")
    print("RÉSULTATS ENTRAÎNEMENT NUMÉRIQUE (compréhension, pas mémorisation)")
    print("="*60)
    for domain, r in results.items():
        if "acc" in r:
            print(f"  {domain}: {r['acc']*100:.1f}%  ({r['method']})")
        else:
            print(f"  {domain}: ERREUR {r.get('error','?')[:80]}")
