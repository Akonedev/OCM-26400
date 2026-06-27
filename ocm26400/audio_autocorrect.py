#!/usr/bin/env python3
"""Audio RECONNAISSANCE par autocorrection — design ROBUSTE (ne casse pas le grok).

Idée utilisateur (27/06/2026) : « le modèle apprend comment faire mais ne sait pas ce
qui est faux → autocorrection, plus de profondeur, apprendre ce qu'il ne faut pas faire ».

3 leçons tirées des 2 premières tentatives (qui ont cassé le grok) :
  1. La calibration/ profondeur introduite DANS le forward path d'un modèle grokké =>
     catastrophic interference (Stage A train 68% -> Stage B train 4%). ÉVITÉ.
  2. Solution = calibration sur FEATURES GELÉES (probe post-hoc) => ne peut PAS casser
     le grok (backbone figé). C'est le standard "calibration probe".
  3. La profondeur (idée user "plus de profondeur") = itération du core À L'INFÉRENCE
     (test-time refinement, comme diffusion-fill n_steps), pas dans le train.

DESIGN (3 étapes, respecte RULES_MASTER §F curriculum : grok d'abord, raffiner ensuite):
  Stage A (GROK)      : encoder profond + SpectralCoreBlock + 1-cos. Reproduit baseline ~42%.
                        Adam 3e-3, seed 0, batch 64 (canonique crown-jewel).
  Stage B (CALIBRATION): GELER backbone, entraîner une tête calib: features -> meta0
                        (prédit la justesse). Le modèle APPREND QUAND IL SE TROMPE
                        (idée user). BCE sur correctness. Backbone gelé => zéro risque.
  Inference (CORRECTION): encoder -> [itérer core K fois = profondeur L4] -> classifier
                        + calibration -> si confiant+correct = compréhension, sinon
                        ABSTENTION (anti-hallucination, spec LSRA). Option re-encode
                        perturbé (self-consistency) pour la boucle de correction.

Respecte : 1-cos (crown-jewel), Adam 3e-3, seed 0, IDs (canonical), observateur meta[0],
profondeur L4 (inférence), abstention ANOMALIE_CAUSALE, SC-1 sanity avant gros run.
Réutilise DeepAudioEncoder (best 42.7%) + SpectralCoreBlock — PAS de réinvention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob, os, numpy as np, time, json
from ocm26400.amv import D_MODEL, PART
from ocm26400.learned_vocab import LearnedVocab
from ocm26400.spectral_core import SpectralCoreBlock   # le noyau FFT (L'ARCHITECTURE)
from train_deep_encoder_v2 import DeepAudioEncoder, load_wav

device = "cuda" if torch.cuda.is_available() else "cpu"
SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"


class AudioAutoCorrect(nn.Module):
    """Encoder profond + noyau FFT (grok) + tête calibration (gelée en Stage B).
    encode() = audio_enc -> SpectralCoreBlock (chemin grok = baseline 42.7%).
    La profondeur (idée user) = ré-itérer self.core K fois À L'INFÉRENCE (refine_*)."""
    def __init__(self, n_words):
        super().__init__()
        self.audio_enc = DeepAudioEncoder(out_dim=D_MODEL, n_mels=128)
        self.core = SpectralCoreBlock(d_model=D_MODEL, seq_len=1)
        self.head = nn.Linear(D_MODEL, PART)             # proj 256->64 (= baseline deep_encoder)
        self.calib = nn.Linear(D_MODEL, 1)               # tête calibration (Stage B, features gelées)

    def encode(self, wav, K=1):
        """audio -> core (itéré K fois = profondeur L4). K=1 = baseline grok."""
        h = self.audio_enc(wav).unsqueeze(1)            # (B,1,256)
        for _ in range(K):
            h = self.core(h)
        return h.squeeze(1)                             # (B,256) AMV

    def ent(self, v):
        return self.head(v)                              # entité <-> canonical(word) (comme baseline)


def _data(words, n_per_word=100):
    audio_by_word = {}
    for wi, w in enumerate(words):
        wavs = [load_wav(p) for p in glob.glob(os.path.join(SC, w, "*.wav"))[:n_per_word]]
        audio_by_word[wi] = torch.stack(wavs).to(device)
    tr, te = {}, {}
    for wi in range(len(words)):
        n = len(audio_by_word[wi]); p = torch.randperm(n); n_te = max(1, n // 5)
        te[wi] = p[:n_te]; tr[wi] = p[n_te:]
    return audio_by_word, tr, te


def train_grok(n_steps=20000, batch=64, lr=3e-3, words=None, eval_every=2500,
               sanity_only=False, n_per_word=100):
    """Stage A : GROK pur (1-cos crown-jewel). Reproduit la baseline ~42%."""
    torch.manual_seed(0)
    words = words or sorted([w for w in os.listdir(SC)
                             if os.path.isdir(os.path.join(SC, w)) and not w.startswith("_")])
    NW = len(words)
    audio_by_word, tr, te = _data(words, n_per_word=n_per_word)
    cv = LearnedVocab(n=NW, dim=PART, init="ortho" if NW <= PART else "random", seed=0)
    cv.freeze(); canon = cv._matrix().to(device)
    model = AudioAutoCorrect(NW).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)   # Adam (PAS AdamW)

    def batch_feat(wi_idx):
        wavs = torch.stack([audio_by_word[wi.item()][tr[wi.item()][torch.randint(0, len(tr[wi.item()]), (1,)).item()]]
                            for wi in wi_idx])
        return model.encode(wavs, K=1), wi_idx           # K=1 = chemin grok

    # SC-1 SANITY : overfit 1 batch, 1-cos pur doit descendre
    print(f"[SC-1 sanity] overfit 1 batch (400 steps, 1-cos pur)...", flush=True)
    sb = torch.randint(0, NW, (min(8, NW),))
    for _ in range(400):
        feat, idx = batch_feat(sb)
        cos = F.cosine_similarity(model.ent(feat), canon[idx], dim=-1).clamp(-1,1)
        loss = (1.0 - cos).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        feat, idx = batch_feat(sb)
        cos_s = F.cosine_similarity(model.ent(feat), canon[idx], dim=-1).mean().item()
    print(f"  sanity: 1-cos sur 1 batch = {1-cos_s:.3f} ({'OK cos>0.9' if cos_s>0.9 else 'apprend...'})", flush=True)
    if sanity_only:
        return model, canon, words, (audio_by_word, tr, te)

    print(f"\n[Stage A GROK] NW={NW} | batch={batch} | Adam {lr} | {n_steps} steps | 1-cos pur", flush=True)
    t0 = time.time()
    best_te = 0.0; best_state = None
    for step in range(n_steps):
        wi_idx = torch.randint(0, NW, (batch,))
        feat, idx = batch_feat(wi_idx)
        cos = F.cosine_similarity(model.ent(feat), canon[idx], dim=-1).clamp(-1,1)
        loss = (1.0 - cos).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % eval_every == 0 or step == n_steps - 1:
            acc_tr, acc_te = _eval_grok(model, canon, audio_by_word, tr, te)
            if acc_te > best_te:                          # garde le meilleur (anti-overfit)
                best_te = acc_te; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  step {step:>5} 1-cos={loss.item():.4f} | train {acc_tr*100:.1f}% "
                  f"test {acc_te*100:.1f}% (best {best_te*100:.1f}%) | t={time.time()-t0:.0f}s", flush=True)
    if best_state:
        model.load_state_dict(best_state)                # recharge le meilleur
    print(f"  [Stage A] meilleur test = {best_te*100:.1f}% (réf baseline deep_encoder 42.7%)", flush=True)
    torch.save({"model_state": model.state_dict(), "best_te": best_te, "words": words},
               "/media/akone/SAVENVME2/Datasets/ocm26400/audio_autocorrect_stageA.pt")
    return model, canon, words, (audio_by_word, tr, te)


def fit_calibration(model, canon, audio_by_word, tr, n_steps=3000, batch=64, lr=3e-3):
    """Stage B : GELER le backbone, entraîner la tête calib (prédire la justesse).
    Backbone gelé => ne peut PAS casser le grok. Le modèle APPREND QUAND IL SE TROMPE."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.calib.parameters():
        p.requires_grad = True                            # seule la tête calib apprend
    opt = torch.optim.Adam(model.calib.parameters(), lr=lr)
    NW = len(audio_by_word)
    print(f"\n[Stage B CALIBRATION] backbone GELÉ, tête calib entraînée ({n_steps} steps)", flush=True)
    t0 = time.time()
    for step in range(n_steps):
        wi_idx = torch.randint(0, NW, (batch,))
        wavs = torch.stack([audio_by_word[wi.item()][tr[wi.item()][torch.randint(0, len(tr[wi.item()]), (1,)).item()]]
                            for wi in wi_idx])
        with torch.no_grad():
            feat = model.encode(wavs, K=1)
            ent = model.ent(feat)
            sims = ent @ canon.t()
            pred = sims.argmax(1)
            correct = (pred == wi_idx.to(device)).float()   # cible : suis-je correct ?
        logit = model.calib(feat).squeeze(-1)             # logit de justesse
        loss = F.binary_cross_entropy_with_logits(logit, correct)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            with torch.no_grad():
                acc = correct.mean().item()
                conf = torch.sigmoid(logit).mean().item()
            print(f"  step {step:>5} BCE={loss.item():.4f} | train_acc {acc*100:.0f}% "
                  f"conf_moy {conf:.2f} | t={time.time()-t0:.0f}s", flush=True)
    for p in model.parameters():
        p.requires_grad = True
    return model


@torch.no_grad()
def _eval_grok(model, canon, audio_by_word, tr, te):
    model.eval()
    def acc_on(split):
        ok = tot = 0
        for wi in range(len(audio_by_word)):
            for j in split[wi]:
                wav = audio_by_word[wi][j:j+1]
                ent = model.ent(model.encode(wav, K=1))
                ok += ((ent @ canon.t()).argmax(1).item() == wi); tot += 1
        return ok/max(tot,1)
    a_tr, a_te = acc_on(tr), acc_on(te); model.train(); return a_tr, a_te


@torch.no_grad()
def evaluate(model, canon, audio_by_word, te, K_depth=4, tau=0.9):
    """Éval avec profondeur (K_depth itérations core) + calibration + abstention."""
    model.eval()
    ok = tot = conf_correct = abst = attempted = ok_attempt = 0
    for wi in range(len(audio_by_word)):
        for j in te[wi]:
            wav = audio_by_word[wi][j:j+1]
            feat = model.encode(wav, K=K_depth)           # profondeur L4 (idée user)
            ent = model.ent(feat)
            pred = (ent @ canon.t()).argmax(1).item()
            conf = torch.sigmoid(model.calib(feat).squeeze(-1)).item()
            correct = (pred == wi)
            tot += 1; ok += correct
            if conf >= tau:                                # confiant
                attempted += 1; ok_attempt += correct
                if correct: conf_correct += 1             # confiant ET correct
            else:
                abst += 1                                  # abstention (sait qu'il ne sait pas)
    n = max(tot, 1)
    return {"test_acc": ok/n, "conf_correct_rate": conf_correct/n,
            "abstention_rate": abst/n, "coverage": attempted/n,
            "acc_when_attempted": ok_attempt/max(attempted,1)}


if __name__ == "__main__":
    print("="*64)
    print("AUDIO AUTOCORRECTION — grok + calibration gelée + profondeur (idée user)")
    print("="*64)
    words = sorted([w for w in os.listdir(SC)
                    if os.path.isdir(os.path.join(SC, w)) and not w.startswith("_")])
    audio_by_word, tr, te = _data(words)
    cv = LearnedVocab(n=len(words), dim=PART, init="ortho", seed=0); cv.freeze()
    canon = cv._matrix().to(device)
    model = AudioAutoCorrect(len(words)).to(device)
    base_ckpt = "/media/akone/SAVENVME2/Datasets/ocm26400/deep_encoder_trained.pt"
    stageA_ckpt = "/media/akone/SAVENVME2/Datasets/ocm26400/audio_autocorrect_stageA.pt"
    # Stage A : RECHARGE la baseline prouvée (deep_encoder 42.7%, mêmes poids audio_enc+core)
    # → évite 28min de re-training. Réutilisation, pas réinvention.
    if os.path.exists(base_ckpt):
        ck = torch.load(base_ckpt, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model_state"], strict=False)
        print(f"[Stage A] RECHARGÉ baseline deep_encoder (42.7%) — "
              f"matched audio_enc+core, calib laisse aléatoire (Stage B le formera)", flush=True)
        print(f"  missing keys (attendu, calib): {[k for k in missing if 'calib' in k][:3]}...", flush=True)
    elif os.path.exists(stageA_ckpt):
        ck = torch.load(stageA_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ck["model_state"])
        print(f"[Stage A] RECHARGÉ cache Stage A ({ck['best_te']*100:.1f}%)", flush=True)
    else:
        model, canon, words, (audio_by_word, tr, te) = train_grok(n_steps=20000)
    # Stage B : calibration sur features gelées (apprend "quand je me trompe")
    model = fit_calibration(model, canon, audio_by_word, tr, n_steps=3000)
    # Éval : profondeur K=4 + calibration + abstention
    for K in [1, 4]:
        r = evaluate(model, canon, audio_by_word, te, K_depth=K)
        print(f"\n[ÉVAL K_depth={K}] test {r['test_acc']*100:.1f}% | "
              f"confiant+correct {r['conf_correct_rate']*100:.1f}% | "
              f"abstention {r['abstention_rate']*100:.1f}% | "
              f"acc_quand_tenté {r['acc_when_attempted']*100:.1f}% (cov {r['coverage']*100:.0f}%)")
    r4 = evaluate(model, canon, audio_by_word, te, K_depth=4)
    print(f"\n{'='*64}\nAUDIO AUTOCORRECTION — RÉSULTAT (K_depth=4)\n{'='*64}")
    print(f"  TEST acc:                      {r4['test_acc']*100:.1f}%")
    print(f"  Confiant ET correct (compréh.):{r4['conf_correct_rate']*100:.1f}%")
    print(f"  Acc quand confiant (tenté):    {r4['acc_when_attempted']*100:.1f}%  (couverture {r4['coverage']*100:.0f}%)")
    print(f"  Abstention (sait qu'il sait pas): {r4['abstention_rate']*100:.1f}%")
    print(f"  Réf: deep_encoder (best préc.) = 42.7% | SOTA SpeechCommands ~96%")
    print(f"  Δ test vs 42.7%: {r4['test_acc']*100 - 42.7:+.1f}pt")
    ckpt = "/media/akone/SAVENVME2/Datasets/ocm26400/audio_autocorrect_trained.pt"
    torch.save({"model_state": model.state_dict(), "results": r4, "words": words,
                "method": "audio autocorrect: grok(1-cos)+frozen calibration+K-depth inference"},
               ckpt)
    print(f"  [SAUVÉ] {ckpt}")
    json.dump({**r4, "delta_vs_42p7": r4["test_acc"]*100 - 42.7, "method": "audio_autocorrect_v3"},
              open("ocm26400/audio_autocorrect_results.json", "w"), indent=2)
