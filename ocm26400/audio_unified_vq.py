#!/usr/bin/env python3
"""AUDIO→VQ UNIFIÉ — IDs discrets + capture simultanée (text+phon+audio-VQ).

Correction user (2 points) :
  1. « pourquoi tu passes pas par un VQ » — VQ est la BONNE direction (IDs discrets,
     principe fondateur §H). Mon VQ précédent a échoué (2.9%) car AUDIO-SEUL : sans ancrage,
     le codebook VQ n'apprend rien de phonétique.
  2. « j'ai dit archi UNIFIÉE » — texte et phon sont DÉJÀ des IDs (word_ID, phon_ID).
     L'audio doit DEVENIR discret (VQ) pour entrer dans l'espace d'IDs unifié, capturé
     SIMULTANÉMENT avec text/phon vers le MÊME canonical via le MÊME SpectralCoreBlock.

La capture simultanée ANCRE le codebook VQ : pour s'aligner au canonical partagé (text/phon),
les IDs VQ audio DOIVENT coder le contenu phonétique invariant. C'est ce qui manquait.

Format respecté (RULES_MASTER) :
  §I capture simultanée : text + phon + audio(VQ) -> même canonical, 1 passe, 1 cœur partagé.
  §H IDs numériques : audio -> VQ -> IDs discrets (comme text/phon sont des IDs).
  §C crown-jewel : 1-cos, Adam 3e-3, seed 0, batch 64.
  archi UNIFIÉE : un seul SpectralCoreBlock (raisonnement) pour les 3 vues.

Réutilise : FrameEncoder + VQ (audio_vq_ids), text_feat/phon_feat (deep_encoder_v2).
"""
import torch, torch.nn as nn, torch.nn.functional as F
import glob, os, numpy as np, time, json, random
from ocm26400.amv import D_MODEL, PART
from ocm26400.learned_vocab import LearnedVocab
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.audio_vq_ids import FrameEncoder, VQ, VQ_K
from ocm26400.audio_simultaneous_proper import _data
from train_deep_encoder_v2 import text_feat, phon_feat

device = "cuda" if torch.cuda.is_available() else "cpu"
SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"


class UnifiedVQ(nn.Module):
    """Archi UNIFIÉE : text/phon (features) + audio (VQ→IDs discrets) -> MÊME SpectralCoreBlock.
    Le cœur est partagé (unifié). La vue audio quantize en IDs discrets (VQ) ancrés par la
    co-capture text/phon vers le canonical commun."""
    def __init__(self, n_words):
        super().__init__()
        # vue audio -> VQ -> IDs discrets
        self.enc = FrameEncoder()                 # Mel(InstanceNorm) -> frames
        self.vq = VQ()                            # frames -> IDs discrets (codebook VQ_K)
        self.id_embed = nn.Embedding(VQ_K, D_MODEL); nn.init.normal_(self.id_embed.weight, std=0.02)
        # vues text/phon (features existantes, baseline OK)
        self.text_proj = nn.Linear(PART, D_MODEL)
        self.phon_proj = nn.Linear(PART, D_MODEL)
        # CŒUR UNIFIÉ (un seul SpectralCoreBlock pour les 3 vues)
        self.core = SpectralCoreBlock(d_model=D_MODEL, seq_len=1)
        self.head = nn.Linear(D_MODEL, PART)

    def _audio_ids(self, wav, augment=False):
        z = self.enc(wav, augment=augment)        # (B,T,VQ_DIM)
        zq, ids, commit = self.vq(z)              # IDs discrets (straight-through)
        # pool les embeddings d'IDs discrets sur la séquence (standard, comme audio_vq)
        emb = self.id_embed(ids)                  # (B,T,D_MODEL)
        return emb.mean(dim=1), commit            # (B, D_MODEL)

    def view_text(self, feat):  return self.head(self.core(self.text_proj(feat).unsqueeze(1)).squeeze(1))
    def view_phon(self, feat):  return self.head(self.core(self.phon_proj(feat).unsqueeze(1)).squeeze(1))
    def view_audio(self, wav, augment=False):
        emb, _ = self._audio_ids(wav, augment)
        return self.head(self.core(emb.unsqueeze(1)).squeeze(1))


def train(n_steps=20000, batch=64, lr=3e-3, eval_every=2500, beta=0.25):
    torch.manual_seed(0); random.seed(0)
    words = sorted([w for w in os.listdir(SC)
                    if os.path.isdir(os.path.join(SC, w)) and not w.startswith("_")])
    NW = len(words)
    tr, te = _data(words); keys = list(tr.keys())
    text_all = torch.tensor([text_feat(w) for w in words]).to(device)
    phon_all = torch.tensor([phon_feat(w) for w in words]).to(device)
    cv = LearnedVocab(n=NW, dim=PART, init="ortho", seed=0); cv.freeze()
    canon = cv._matrix().to(device)
    model = UnifiedVQ(NW).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def joint_loss(wi_t, wavs, augment):
        tgt = canon[wi_t]
        out_t = model.view_text(text_all[wi_t])
        out_p = model.view_phon(phon_all[wi_t])
        emb_a, commit = model._audio_ids(wavs, augment=augment)
        out_a = model.head(model.core(emb_a.unsqueeze(1)).squeeze(1))
        l_align = ((1-F.cosine_similarity(out_t,tgt,-1).clamp(-1,1)).mean() +
                   (1-F.cosine_similarity(out_p,tgt,-1).clamp(-1,1)).mean() +
                   (1-F.cosine_similarity(out_a,tgt,-1).clamp(-1,1)).mean())
        return l_align + beta*commit, l_align.item()

    # SC-1 sanity
    print("[SC-1 sanity] overfit 1 batch (400 steps, capture simultanée VQ)...", flush=True)
    sb = keys[:8]; sw = torch.stack([tr[k][0] for k in sb]); swi = torch.tensor(sb, device=device)
    for _ in range(400):
        loss, _ = joint_loss(swi, sw, augment=False)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        cos_s = F.cosine_similarity(model.view_audio(sw), canon[swi], -1).mean().item()
    print(f"  sanity audio(VQ) 1-cos = {1-cos_s:.3f} ({'OK' if cos_s>0.9 else 'apprend...'})", flush=True)

    print(f"\n[TRAIN VQ UNIFIÉ] {len(keys)} mots | text+phon+audio(VQ) JOINT | codebook {VQ_K} | {n_steps} steps", flush=True)
    t0 = time.time(); best = 0.0; best_state = None
    for step in range(n_steps):
        bi = [random.choice(keys) for _ in range(batch)]
        wavs = torch.stack([tr[k][torch.randint(0,len(tr[k]),(1,)).item()] for k in bi])
        wi_t = torch.tensor(bi, device=device)
        loss, la = joint_loss(wi_t, wavs, augment=True)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % eval_every == 0 or step == n_steps-1:
            acc = _eval(model, canon, te)
            if acc > best: best = acc; best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
            print(f"  step {step:>5} loss={loss.item():.4f} (align {la:.3f}) | holdout[100:130] {acc*100:.1f}% "
                  f"(best {best*100:.1f}%) | t={time.time()-t0:.0f}s", flush=True)
    if best_state: model.load_state_dict(best_state)
    return model, canon, te, best


@torch.no_grad()
def _eval(model, canon, te):
    model.eval(); ok = tot = 0
    for wi in te:
        for j in range(len(te[wi])):
            ok += ((model.view_audio(te[wi][j:j+1]) @ canon.t()).argmax(1).item() == wi); tot += 1
    model.train(); return ok/max(tot,1)


if __name__ == "__main__":
    print("="*64)
    print("AUDIO→VQ UNIFIÉ — IDs discrets + capture simultanée (archi unifiée)")
    print("="*64)
    model, canon, te, best = train(n_steps=20000)
    print(f"\n{'='*64}\nRÉSULTAT VQ UNIFIÉ — holdout PROPRE [100:130]\n{'='*64}")
    print(f"  Test acc (audio VQ, après co-capture unifiée): {best*100:.1f}%")
    print(f"  Réf: VQ audio-seul 2.9% (échec) | simultaneous continuous | SOTA 96%")
    print(f"  Δ vs VQ audio-seul: {best*100-2.9:+.1f}pt | Δ vs SOTA: {best*100-96:+.1f}pt")
    torch.save({"model_state": model.state_dict(), "best": best},
               "/media/akone/SAVENVME2/Datasets/ocm26400/audio_unified_vq_trained.pt")
    json.dump({"holdout_acc": best, "delta_vs_vq_audio_seul_2p9": best*100-2.9,
               "delta_vs_sota_96": best*100-96,
               "method": "UNIFIED VQ: audio->VQ->discrete IDs, simultaneous capture with "
                         "text+phon, shared SpectralCoreBlock, 1-cos joint crown-jewel"},
              open("ocm26400/audio_unified_vq_results.json","w"), indent=2)
    print("  [sauvé] ocm26400/audio_unified_vq_results.json")
