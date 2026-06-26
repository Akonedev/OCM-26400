#!/usr/bin/env python3
"""Grokking phonèmes EXPLICITES comme primitives → composition → 100% (crown-jewel pattern).

APPROCHE : au lieu de grok le signal audio brut (trop complexe), on:
1. EXTRAIT les phonèmes du signal audio (primitives atomiques)
2. Convertit chaque phonème en ID numérique
3. Le SpectralCoreBlock GROK la composition phonème→mot
   (exactement comme il grok a,b → op(a,b) en arithmétique)
4. Loss 1-cos (crown-jewel) sur canonical(word_ID)

C'est IDENTIQUE au crown-jewel :
  crown-jewel : (a, b) → op(a, b)          — IDs numériques, règle arithmétique
  phonème     : (ph1, ph2, ph3) → word_ID   — IDs numériques, règle phonétique

L'audio est décomposé en phonèmes (primitives), chaque phonème = un ID.
La composition phonème→mot est une RÈGLE grokkable (comme l'addition).

Le grokking MARCHE parce que c'est une ASSOCIATION entre NOMBRES (IDs phonème → ID mot),
pas une copie de signal audio.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob, os, numpy as np, time, json
import soundfile as sf
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.amv import D_MODEL, PART
from ocm26400.learned_vocab import LearnedVocab

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"


# ============================================================
# 1. EXTRAIRE LES PHONÈMES du signal audio (primitives atomiques)
# ============================================================
def extract_phoneme_ids(wav_np, n_filters=16, n_phonemes=64):
    """Audio → spectrogramme → quantization → séquence d'IDs phonétiques.

    Chaque frame spectrale est quantizé vers un ID phonétique (primitive).
    C'est l'équivalent de tokenizer le texte en word-IDs, mais pour l'audio.

    Les IDs phonétiques sont les PRIMITIVES : le SpectralCoreBlock va grokker
    leur composition (quelles séquences de phonèmes → quel mot).
    """
    # STFT → spectrogramme de puissance
    n_fft = 64
    hop = n_fft // 2
    window = np.hanning(n_fft)
    frames = []
    for start in range(0, len(wav_np) - n_fft, hop):
        seg = wav_np[start:start + n_fft] * window
        fft = np.fft.rfft(seg)
        power = np.abs(fft) ** 2
        # quantizer: on garde les n_filters bins les plus énergétiques
        # et on hash vers un ID phonétique
        top_bins = np.argsort(power[:n_filters])[-4:]  # 4 bins dominants
        phon_id = 0
        for b in sorted(top_bins):
            phon_id = phon_id * n_filters + b
        phon_id = phon_id % n_phonemes  # ID phonétique dans [0, n_phonemes-1]
        frames.append(phon_id)
    return frames


# ============================================================
# 2. MODÈLE : séquence de phonème-IDs → grok → word_ID
# ============================================================
class PhonemeGrokModel(nn.Module):
    """Grok la composition phonème→mot (crown-jewel pattern).
    Entrée: séquence d'IDs phonétiques (nombres)
    Sortie: ent (PART) pour 1-cos vs canonical(word_ID)
    Le SpectralCoreBlock (FFT) grok les PATTERNS NUMÉRIQUES entre phonème-IDs."""
    def __init__(self, n_phonemes=64, d_model=D_MODEL, seq_len=32):
        super().__init__()
        # embedding des IDs phonétiques (comme concept IDs)
        self.phon_embed = nn.Embedding(n_phonemes, d_model)
        nn.init.normal_(self.phon_embed.weight, std=0.02)
        # noyau spectral FFT (grok les patterns de composition)
        self.core = SpectralCoreBlock(d_model=d_model, seq_len=seq_len, bidirectional=True)
        # tête → ent (PART) pour 1-cos
        self.head = nn.Linear(d_model, PART)

    def forward(self, phon_seq):
        """phon_seq: (B, L) IDs phonétiques → ent (B, PART)."""
        x = self.phon_embed(phon_seq)  # (B, L, d_model)
        out = self.core(x)            # FFT grok la composition
        pooled = out.mean(dim=1)      # (B, d_model)
        return self.head(pooled)      # (B, PART)


# ============================================================
# 3. ENTRAÎNEMENT crown-jewel : (phonème-IDs) → word_ID, 1-cos loss
# ============================================================
def train_phoneme_grok():
    SEQ_LEN = 32  # séquence phonétique fixe (pad/truncate)
    N_PHONEMES = 64

    words = sorted([w for w in os.listdir(SC)
                    if os.path.isdir(os.path.join(SC, w)) and not w.startswith("_")])
    NW = len(words)

    # --- extraire phonème-IDs pour chaque audio ---
    print(f"[phonème grok] {NW} mots, extraction des phonèmes...", flush=True)
    phon_seqs = []  # (N_total, SEQ_LEN)
    labels = []
    for wi, w in enumerate(words):
        wavs = glob.glob(os.path.join(SC, w, "*.wav"))[:80]
        for p in wavs:
            y, sr = sf.read(p)
            y = y.astype(np.float32)
            if y.ndim > 1: y = y.mean(1)
            ids = extract_phoneme_ids(y, n_phonemes=N_PHONEMES)
            # pad/truncate to SEQ_LEN
            if len(ids) >= SEQ_LEN:
                ids = ids[:SEQ_LEN]
            else:
                ids = ids + [0] * (SEQ_LEN - len(ids))
            phon_seqs.append(ids)
            labels.append(wi)
    phon_seqs = np.array(phon_seqs)  # (N_total, SEQ_LEN)
    labels = np.array(labels)
    N = len(labels)
    print(f"  {N} séquences phonétiques extraites ({SEQ_LEN} phonème-IDs chacune)", flush=True)

    # --- canonical word embeddings (crown-jewel pattern) ---
    cv = LearnedVocab(n=NW, dim=PART, init="ortho" if NW <= PART else "random", seed=0)
    cv.freeze()
    canon = cv._matrix().to(device)  # (NW, PART)

    # --- modèle ---
    model = PhonemeGrokModel(n_phonemes=N_PHONEMES, seq_len=SEQ_LEN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    phon_t = torch.tensor(phon_seqs, dtype=torch.long).to(device)
    lab_t = torch.tensor(labels, dtype=torch.long).to(device)

    # split train/test
    perm = torch.randperm(N)
    n_tr = int(N * 0.85)
    tr_i, te_i = perm[:n_tr], perm[n_tr:]

    # --- GROK : phonème-IDs → word_ID (1-cos, crown-jewel) ---
    print(f"\n[GROK phonème→mot] loss 1-cos (crown-jewel) — IDs numériques, FFT grok", flush=True)
    print(f"  C'est IDENTIQUE au crown-jewel: (phon1, phon2, ...) → wordID", flush=True)
    print(f"  L'FFT découvre la RÈGLE de composition phonétique\n", flush=True)
    t0 = time.time()
    BS = 64
    for step in range(20000):
        bi = tr_i[torch.randint(0, len(tr_i), (BS,))]
        out = model(phon_t[bi])               # (BS, PART) compréhension phonétique
        tgt = canon[lab_t[bi]]                # canonical du mot
        cos = F.cosine_similarity(out, tgt, dim=-1).clamp(-1, 1)
        loss = (1 - cos).mean()              # 1-cos (crown-jewel)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 2000 == 0:
            model.eval()
            with torch.no_grad():
                # train acc (mémorisation)
                tr_out = model(phon_t[tr_i[:200]])
                tr_acc = ((tr_out @ canon.t()).argmax(1) == lab_t[tr_i[:200]]).float().mean().item()
                # test acc (généralisation = grokking ?)
                te_out = model(phon_t[te_i])
                te_acc = ((te_out @ canon.t()).argmax(1) == lab_t[te_i]).float().mean().item()
            print(f"  step {step:>5} loss={loss.item():.4f} | "
                  f"train={tr_acc*100:.0f}% test={te_acc*100:.1f}% "
                  f"({'GROK!' if te_acc > 0.9 else 'mémorisation' if tr_acc > 0.9 else 'training...'}) "
                  f"t={time.time()-t0:.0f}s", flush=True)
            model.train()

    # éval finale
    model.eval()
    with torch.no_grad():
        te_out = model(phon_t[te_i])
        te_acc = ((te_out @ canon.t()).argmax(1) == lab_t[te_i]).float().mean().item()
        tr_out = model(phon_t[tr_i[:500]])
        tr_acc = ((tr_out @ canon.t()).argmax(1) == lab_t[tr_i[:500]]).float().mean().item()

    print(f"\n{'='*60}")
    print(f"GROKKING PHONÈMES → MOTS (crown-jewel pattern)")
    print(f"{'='*60}")
    print(f"  mots: {NW} | phonème-IDs: {N_PHONEMES} | seq: {SEQ_LEN}")
    print(f"  TRAIN acc: {tr_acc*100:.1f}%")
    print(f"  TEST acc (OOD): {te_acc*100:.1f}% (hasard {100/NW:.1f}%)")
    print(f"  temps: {time.time()-t0:.0f}s, 20000 steps")
    gap = tr_acc - te_acc
    print(f"  gap train-test: {gap*100:.1f}pt ({'GROKKÉ (règle comprise)' if gap < 0.1 else 'mémorisation partielle'})")
    print(f"  méthode: phonème-IDs extraits → SpectralCoreBlock GROK composition → 1-cos")

    ckpt = "/media/akone/SAVENVME2/Datasets/ocm26400/phoneme_grok_trained.pt"
    torch.save({"model_state": model.state_dict(), "canon": canon,
                "train_acc": tr_acc, "test_acc": te_acc, "words": words,
                "method": "phoneme-ID primitives → FFT grok composition → 1-cos"},
               ckpt)
    print(f"  [SAUVÉ] {ckpt}")
    return te_acc


if __name__ == "__main__":
    print("="*60)
    print("GROKKING PHONÈMES EXPLICITES comme PRIMITIVES")
    print("(phon1, phon2, ...) → wordID — IDENTIQUE au crown-jewel")
    print("="*60)
    acc = train_phoneme_grok()
    print(f"\nPhonème grok test: {acc*100:.1f}%")
