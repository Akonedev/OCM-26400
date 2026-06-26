#!/usr/bin/env python3
"""Capture SIMULTANÉE cross-modale BATCHÉ — texte + phonétique + audio → même ID.

Version vectorisée : 3 forward passes batchés (au lieu de 96 individuels) = 30× plus rapide.
Permet 6000+ pas dans le même temps → le grokking va plus loin.

PRINCIPE : tout numérique, tout simultané, compréhension cross-modale.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import glob, os, numpy as np, time, json
import soundfile as sf
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.multimodal_encoders import AudioEncoder
from ocm26400.amv import D_MODEL, PART
from ocm26400.learned_vocab import LearnedVocab

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
T_AUDIO = 8000


def text_features(word):
    v = np.zeros(PART, dtype=np.float32)
    for c in word.lower():
        v[(ord(c) * 167) % PART] += 1.0
    return v


def phon_features(word):
    w = word.lower()
    vowels = sum(1 for c in w if c in "aeiou")
    consonants = len(w) - vowels
    pattern = "".join("v" if c in "aeiou" else "c" for c in w)[:8]
    syllables = max(1, vowels)
    v = np.zeros(PART, dtype=np.float32)
    for c in pattern:
        v[(ord(c) * 167) % PART] += 1.0
    v[(vowels * 7) % PART] += 1.0
    v[(consonants * 11 + PART // 2) % PART] += 1.0
    v[(syllables * 13) % PART] += 1.0
    return v


def load_wav(p):
    y, sr = sf.read(p); y = y.astype(np.float32)
    if y.ndim > 1: y = y.mean(1)
    if len(y) < T_AUDIO: y = np.pad(y, (0, T_AUDIO - len(y)))
    else: y = y[:T_AUDIO]
    return torch.tensor(y)


class CrossModalGrokModel(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.text_proj = nn.Linear(PART, d_model)
        self.phon_proj = nn.Linear(PART, d_model)
        self.audio_enc = AudioEncoder(out_dim=d_model)
        self.core = SpectralCoreBlock(d_model=d_model, seq_len=1)
        self.head = nn.Linear(d_model, PART)

    def forward_view(self, feat, proj):
        return self.head(self.core(proj(feat).unsqueeze(1)).squeeze(1))

    def forward_audio(self, wav):
        return self.head(self.core(self.audio_enc(wav).unsqueeze(1)).squeeze(1))


def train():
    SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
    words = sorted([w for w in os.listdir(SC)
                    if os.path.isdir(os.path.join(SC, w)) and not w.startswith("_")])
    NW = len(words)

    # précompute TOUTES les features en tenseurs (pour batching rapide)
    text_all = torch.tensor([text_features(w) for w in words]).to(device)  # (NW, PART)
    phon_all = torch.tensor([phon_features(w) for w in words]).to(device)
    # audio : liste de tenseurs par mot
    audio_by_word = {}
    for wi, w in enumerate(words):
        wavs = [load_wav(p) for p in glob.glob(os.path.join(SC, w, "*.wav"))[:50]]
        audio_by_word[wi] = torch.stack(wavs).to(device)  # (n_samples, T)
    print(f"[cross-modal batché] {NW} mots, audio + phonétique + texte", flush=True)

    cv = LearnedVocab(n=NW, dim=PART, init="ortho" if NW <= PART else "random", seed=0)
    cv.freeze()
    canon = cv._matrix().to(device)

    model = CrossModalGrokModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    # test set : indices d'audio par mot (20% pour test)
    te_indices = []
    tr_word_samples = {wi: [] for wi in range(NW)}
    for wi in range(NW):
        n = len(audio_by_word[wi])
        perm = torch.randperm(n)
        n_te = max(1, n // 5)
        for j in perm[:n_te]:
            te_indices.append((wi, j))
        for j in perm[n_te:]:
            tr_word_samples[wi].append(j)
    print(f"  {sum(len(v) for v in tr_word_samples.values())} train, {len(te_indices)} test", flush=True)

    def sample_audio_batch(word_indices):
        """Batch d'audio : un sample aléatoire par mot."""
        wavs = []
        for wi in word_indices:
            samples = tr_word_samples[wi]
            j = samples[torch.randint(0, len(samples), (1,)).item()]
            wavs.append(audio_by_word[wi][j])
        return torch.stack(wavs)  # (B, T)

    print(f"\n[CAPTURE SIMULTANÉE batchée — 3 vues en MÊME TEMPS, forward vectorisé]", flush=True)
    t0 = time.time()
    BS = 48
    for step in range(6000):
        wi_batch = torch.randint(0, NW, (BS,))  # batch de word-IDs
        tgt = canon[wi_batch]                    # (BS, PART) canonical
        # 3 vues BATCHÉES (3 forward, pas 96)
        out_t = model.forward_view(text_all[wi_batch], model.text_proj)
        out_p = model.forward_view(phon_all[wi_batch], model.phon_proj)
        aud_batch = sample_audio_batch(wi_batch.tolist())
        out_a = model.forward_audio(aud_batch)
        # loss 1-cos (crown-jewel) sur les 3 vues
        loss = ((1 - F.cosine_similarity(out_t, tgt).clamp(-1, 1)).mean() +
                (1 - F.cosine_similarity(out_p, tgt).clamp(-1, 1)).mean() +
                (1 - F.cosine_similarity(out_a, tgt).clamp(-1, 1)).mean())
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            model.eval()
            with torch.no_grad():
                ok_a = ok_t = ok_p = 0
                for wi, j in te_indices[:100]:
                    wav = audio_by_word[wi][j].unsqueeze(0)
                    if (model.forward_audio(wav) @ canon.t()).argmax(1).item() == wi: ok_a += 1
                for wi in range(NW):
                    if (model.forward_view(text_all[wi:wi+1], model.text_proj) @ canon.t()).argmax(1).item() == wi: ok_t += 1
                    if (model.forward_view(phon_all[wi:wi+1], model.phon_proj) @ canon.t()).argmax(1).item() == wi: ok_p += 1
            print(f"  step {step} loss={loss.item():.4f} | audio={ok_a}% texte={ok_t}/{NW} phon={ok_p}/{NW} t={time.time()-t0:.0f}s", flush=True)
            model.train()

    # éval finale
    model.eval()
    with torch.no_grad():
        ok_a = sum(1 for wi, j in te_indices if (model.forward_audio(audio_by_word[wi][j].unsqueeze(0)) @ canon.t()).argmax(1).item() == wi)
        ok_t = sum(1 for wi in range(NW) if (model.forward_view(text_all[wi:wi+1], model.text_proj) @ canon.t()).argmax(1).item() == wi)
        ok_p = sum(1 for wi in range(NW) if (model.forward_view(phon_all[wi:wi+1], model.phon_proj) @ canon.t()).argmax(1).item() == wi)
    print(f"\n=== RÉSULTAT FINAL (capture simultanée cross-modale) ===")
    print(f"  AUDIO  (test OOD): {ok_a}/{len(te_indices)} = {ok_a/max(len(te_indices),1)*100:.1f}% (hasard {100/NW:.0f}%)")
    print(f"  TEXTE  : {ok_t}/{NW} = {ok_t/NW*100:.0f}%")
    print(f"  PHONÉT : {ok_p}/{NW} = {ok_p/NW*100:.0f}%")
    print(f"  temps: {time.time()-t0:.0f}s, 6000 steps")

    # save checkpoint
    ckpt = "/media/akone/SAVENVME2/Datasets/ocm26400/crossmodal_trained.pt"
    torch.save({"model_state": model.state_dict(), "words": words,
                "audio_acc": ok_a/max(len(te_indices),1), "text_acc": ok_t/NW, "phon_acc": ok_p/NW,
                "method": "simultaneous cross-modal (text+phonetic+audio→same ID, 1-cos)"}, ckpt)
    print(f"  [SAUVÉ] {ckpt}")
    return ok_a/max(len(te_indices),1)


if __name__ == "__main__":
    acc = train()
