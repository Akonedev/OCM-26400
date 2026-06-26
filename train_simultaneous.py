#!/usr/bin/env python3
"""Capture SIMULTANÉE cross-modale — texte + phonétique + audio → même ID numérique.

PRINCIPE : tout capturer EN MÊME TEMPS pour les associations cross-modales.
Le phonème est le pont texte↔audio. Le texte (déjà grokké) ancre l'audio.

Chaque mot a plusieurs VUES qui mappent au MÊME ID numérique (concept) :
  - vue texte : word → features → ID
  - vue phonétique : phoneme_pattern + vowels + consonants → features → même ID
  - vue audio : SpeechCommands wav → AudioEncoder → même ID

Le SpectralCoreBlock grok : toutes les vues → même concept = COMPRÉHENSION CROSS-MODALE.
Loss 1-cos (crown-jewel) : chaque vue → canonical(word_ID).

La compréhension textuelle/phonétique (grokkée) TRANSFÈRE à l'audio via l'ID partagé.
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


def phonetic_features(word):
    """Dérive les features phonétiques d'un mot (le pont texte↔audio)."""
    w = word.lower()
    vowels = sum(1 for c in w if c in "aeiou")
    consonants = len(w) - vowels
    # pattern cv simplifié
    pattern = "".join("v" if c in "aeiou" else "c" for c in w)[:8]
    syllables = max(1, vowels)
    return pattern, vowels, consonants, syllables


def text_features(word):
    """Features textuelles → numérique (hash des chars, pas de texte brut)."""
    v = np.zeros(PART)
    for c in word.lower():
        v[(ord(c) * 167) % PART] += 1.0
    return v


def phon_to_features(pattern, vowels, consonants, syllables):
    """Features phonétiques → numérique."""
    v = np.zeros(PART)
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
    """Grok cross-modal : texte/phonétique/audio → même concept ID.
    Le SpectralCoreBlock (FFT) est PARTAGÉ par toutes les vues."""
    def __init__(self, n_words, d_model=D_MODEL):
        super().__init__()
        # projections des vues vers l'espace AMV
        self.text_proj = nn.Linear(PART, d_model)
        self.phon_proj = nn.Linear(PART, d_model)
        self.audio_enc = AudioEncoder(out_dim=d_model)
        # noyau spectral partagé (grok cross-modal)
        self.core = SpectralCoreBlock(d_model=d_model, seq_len=1)
        # tête : output AMV → ent (PART) pour 1-cos
        self.head = nn.Linear(d_model, PART)

    def forward_text(self, feat):
        return self.head(self.core(self.text_proj(feat).unsqueeze(1)).squeeze(1))

    def forward_phon(self, feat):
        return self.head(self.core(self.phon_proj(feat).unsqueeze(1)).squeeze(1))

    def forward_audio(self, wav):
        return self.head(self.core(self.audio_enc(wav).unsqueeze(1)).squeeze(1))


def train_simultaneous():
    SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
    words = sorted([w for w in os.listdir(SC)
                    if os.path.isdir(os.path.join(SC, w)) and not w.startswith("_")])
    NW = len(words)
    print(f"[cross-modal] {NW} mots avec audio + phonétique dérivée", flush=True)

    # préparer les vues pour chaque mot
    text_feats = {}  # word_idx -> (PART,) numpy
    phon_feats = {}
    audio_samples = {}  # word_idx -> list of wav tensors

    for wi, w in enumerate(words):
        pat, v, c, s = phonetic_features(w)
        text_feats[wi] = text_features(w)
        phon_feats[wi] = phon_to_features(pat, v, c, s)
        wavs = [load_wav(p) for p in glob.glob(os.path.join(SC, w, "*.wav"))[:40]]
        audio_samples[wi] = wavs

    # canonical embeddings (crown-jewel pattern)
    cv = LearnedVocab(n=NW, dim=PART, init="ortho" if NW <= PART else "random", seed=0)
    cv.freeze()
    canon = cv._matrix().to(device)

    model = CrossModalGrokModel(NW).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    # split train/test par SAMPLE (mêmes mots, samples différents)
    all_data = []
    for wi in range(NW):
        for wav in audio_samples[wi]:
            all_data.append((wi, wav))
    random_idx = torch.randperm(len(all_data))
    n_tr = int(len(all_data) * 0.8)
    tr = [all_data[i] for i in random_idx[:n_tr]]
    te = [all_data[i] for i in random_idx[n_tr:]]
    print(f"  {len(tr)} train, {len(te)} test samples", flush=True)

    # ENTRAÎNEMENT SIMULTANÉ : texte + phonétique + audio → même ID
    print(f"\n[CAPTURE SIMULTANÉE cross-modale — texte + phonétique + audio en MÊME TEMPS]", flush=True)
    print(f"  loss 1-cos (crown-jewel) sur TOUTES les vues → même canonical(ID)", flush=True)
    t0 = time.time()
    for step in range(4000):
        bi = torch.randint(0, len(tr), (32,))
        total_loss = torch.tensor(0.0, device=device)
        n_views = 0
        for i in bi:
            wi, wav = tr[i]
            wav_dev = wav.unsqueeze(0).to(device)
            tgt = canon[wi].unsqueeze(0)  # canonical du mot
            # vue audio (always available)
            out_a = model.forward_audio(wav_dev)
            total_loss = total_loss + (1 - F.cosine_similarity(out_a, tgt, dim=-1).clamp(-1, 1)).mean()
            n_views += 1
            # vue texte (always available, NUMÉRIQUE — pas de texte brut)
            tf = torch.tensor(text_feats[wi], dtype=torch.float32).unsqueeze(0).to(device)
            out_t = model.forward_text(tf)
            total_loss = total_loss + (1 - F.cosine_similarity(out_t, tgt, dim=-1).clamp(-1, 1)).mean()
            n_views += 1
            # vue phonétique (le pont, always available)
            pf = torch.tensor(phon_feats[wi], dtype=torch.float32).unsqueeze(0).to(device)
            out_p = model.forward_phon(pf)
            total_loss = total_loss + (1 - F.cosine_similarity(out_p, tgt, dim=-1).clamp(-1, 1)).mean()
            n_views += 1
        loss = total_loss / n_views
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            # test audio classification (vue audio seule)
            model.eval()
            with torch.no_grad():
                ok = 0
                for wi, wav in te[:100]:
                    out = model.forward_audio(wav.unsqueeze(0).to(device))
                    pred = (out @ canon.t()).argmax(1).item()
                    ok += (pred == wi)
            print(f"  step {step} loss={loss.item():.4f} audio_test={ok}/100 t={time.time()-t0:.0f}s", flush=True)
            model.train()

    # évaluation finale
    model.eval()
    with torch.no_grad():
        # AUDIO classification (vue audio seule — test cross-modal transfer)
        ok_a = sum(1 for wi, wav in te if (model.forward_audio(wav.unsqueeze(0).to(device)) @ canon.t()).argmax(1).item() == wi)
        # TEXT classification (vue texte)
        ok_t = 0
        for wi in range(NW):
            out = model.forward_text(torch.tensor(text_feats[wi], dtype=torch.float32).unsqueeze(0).to(device))
            if (out @ canon.t()).argmax(1).item() == wi: ok_t += 1
        # PHONETIC classification
        ok_p = 0
        for wi in range(NW):
            out = model.forward_phon(torch.tensor(phon_feats[wi], dtype=torch.float32).unsqueeze(0).to(device))
            if (out @ canon.t()).argmax(1).item() == wi: ok_p += 1

    print(f"\n=== RÉSULTATS CAPTURE SIMULTANÉE CROSS-MODALE ===")
    print(f"  AUDIO classification (test OOD): {ok_a}/{len(te)} = {ok_a/max(len(te),1)*100:.1f}% (hasard {100/NW:.0f}%)")
    print(f"  TEXTE classification: {ok_t}/{NW} = {ok_t/NW*100:.0f}%")
    print(f"  PHONÉTIQUE classification: {ok_p}/{NW} = {ok_p/NW*100:.0f}%")
    print(f"  temps: {time.time()-t0:.0f}s")
    print(f"  méthode: texte + phonétique + audio simultanés → même ID (cross-modal)")
    return ok_a / max(len(te), 1)


if __name__ == "__main__":
    print("="*60)
    print("CAPTURE SIMULTANÉE CROSS-MODALE (texte + phonétique + audio → même ID)")
    print("PRINCIPE : tout numérique, tout simultané, compréhension cross-modale")
    print("="*60)
    acc = train_simultaneous()
    print(f"\nAudio via cross-modal: {acc*100:.1f}%")
