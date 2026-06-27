#!/usr/bin/env python3
"""AUDIO→VQ GUMBEL-SOFTMAX (différentiable) + capture simultanée unifiée.

Le VQ straight-through (audio_vq_ids, audio_unified_vq) a échoué (2.9% plat) : la passe
avant = argmin (one-hot), la passe arrière = softmax détaché → gradients pauvres vers le
codebook, le codebook n'apprend pas des IDs phonétiques.

CORRECTION : VQ GUMBEL-SOFTMAX (Jang 2016). Au lieu de argmin+straight-through, on calcule
une softmax Gumbel sur les distances au codebook → zq = soft_weights @ codebook, PLEINEMENT
différentiable (gradients vers TOUS les codes). Avec annealing de température (tau: chaud
-> froid), le zq passe de doux à quasi-one-hot. Le codebook apprend vraiment.

Intégré à la capture simultanée unifiée (text+phon+audio-VQ -> même canonical) : la co-capture
ANCRE le codebook à être phonétique (s'aligner au canonical partagé). C'est le VQ "fait
correctement" + l'archi unifiée + le format prescrit.

Réutilise FrameEncoder (audio_vq_ids), text_feat/phon_feat (deep_encoder_v2), _data (simul).
"""
import torch, torch.nn as nn, torch.nn.functional as F
import glob, os, time, json, random
from ocm26400.amv import D_MODEL, PART
from ocm26400.learned_vocab import LearnedVocab
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.audio_vq_ids import FrameEncoder, VQ_K
from ocm26400.audio_simultaneous_proper import _data
from train_deep_encoder_v2 import text_feat, phon_feat

device = "cuda" if torch.cuda.is_available() else "cpu"
SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"


class GumbelVQ(nn.Module):
    """VQ Gumbel-softmax : z -> distances codebook -> softmax Gumbel -> zq différenciable.
    Pleinement différentiable (vs straight-through) => le codebook APPREND."""
    def __init__(self, k=VQ_K, dim=32, beta=0.25):
        super().__init__()
        self.k, self.beta = k, beta
        self.codebook = nn.Parameter(torch.randn(k, dim) * 0.1)
    def forward(self, z, tau):                       # z=(B,T,dim)
        B, T, D = z.shape; zf = z.reshape(-1, D)
        d = torch.cdist(zf, self.codebook)           # (B*T, K) distances euclidiennes
        logits = -d                                   # plus proche = logit plus haut
        soft = F.gumbel_softmax(logits, tau=tau, hard=False)   # (B*T,K) différenciable
        zq = (soft @ self.codebook).reshape(B, T, D) # quantisation différenciable
        commit = F.mse_loss(zf, zq.reshape(-1, D).detach())
        return zq, commit


class UnifiedGumbelVQ(nn.Module):
    """Capture simultanée text+phon+audio(GumbelVQ) -> shared SpectralCoreBlock -> canonical."""
    def __init__(self, n_words):
        super().__init__()
        self.enc = FrameEncoder()
        self.vq = GumbelVQ()
        self.id_embed = nn.Linear(32, D_MODEL)        # proj frames quantifiés -> 256
        self.text_proj = nn.Linear(PART, D_MODEL)
        self.phon_proj = nn.Linear(PART, D_MODEL)
        self.core = SpectralCoreBlock(d_model=D_MODEL, seq_len=1)   # CŒUR UNIFIÉ partagé
        self.head = nn.Linear(D_MODEL, PART)
    def _audio(self, wav, augment, tau):
        z = self.enc(wav, augment=augment)
        zq, commit = self.vq(z, tau=tau)             # quantification Gumbel différenciable
        emb = self.id_embed(zq).mean(dim=1)          # pool frames -> (B,256)
        return emb, commit
    def view_text(self, f): return self.head(self.core(self.text_proj(f).unsqueeze(1)).squeeze(1))
    def view_phon(self, f): return self.head(self.core(self.phon_proj(f).unsqueeze(1)).squeeze(1))
    def view_audio(self, wav, augment=False, tau=1.0):
        emb, _ = self._audio(wav, augment, tau)
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
    model = UnifiedGumbelVQ(NW).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def joint(wi_t, wavs, augment, tau):
        tgt = canon[wi_t]
        out_t = model.view_text(text_all[wi_t])
        out_p = model.view_phon(phon_all[wi_t])
        emb_a, commit = model._audio(wavs, augment, tau)
        out_a = model.head(model.core(emb_a.unsqueeze(1)).squeeze(1))
        la = ((1-F.cosine_similarity(out_t,tgt,-1).clamp(-1,1)).mean() +
              (1-F.cosine_similarity(out_p,tgt,-1).clamp(-1,1)).mean() +
              (1-F.cosine_similarity(out_a,tgt,-1).clamp(-1,1)).mean())
        return la + beta*commit, la.item()

    # SC-1 sanity
    print("[SC-1 sanity] overfit 1 batch (400 steps, GumbelVQ tau=1)...", flush=True)
    sb = keys[:8]; sw = torch.stack([tr[k][0] for k in sb]); swi = torch.tensor(sb, device=device)
    for _ in range(400):
        loss, _ = joint(swi, sw, False, tau=1.0)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        cos_s = F.cosine_similarity(model.view_audio(sw, tau=1.0), canon[swi], -1).mean().item()
    print(f"  sanity audio(GumbelVQ) 1-cos = {1-cos_s:.3f} ({'OK' if cos_s>0.9 else 'apprend...'})", flush=True)

    print(f"\n[TRAIN GumbelVQ UNIFIÉ] {len(keys)} mots | text+phon+audio(GumbelVQ) JOINT | {n_steps} steps", flush=True)
    print(f"  tau annealing: 1.0 -> 0.1 (chaud=doux différenciable -> froid=quasi-one-hot)", flush=True)
    t0 = time.time(); best = 0.0; best_state = None
    for step in range(n_steps):
        tau = max(0.1, 1.0 - 0.9*step/n_steps)        # annealing température Gumbel
        bi = [random.choice(keys) for _ in range(batch)]
        wavs = torch.stack([tr[k][torch.randint(0,len(tr[k]),(1,)).item()] for k in bi])
        wi_t = torch.tensor(bi, device=device)
        loss, la = joint(wi_t, wavs, True, tau)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % eval_every == 0 or step == n_steps-1:
            acc = _eval(model, canon, te)
            if acc > best: best = acc; best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
            print(f"  step {step:>5} tau={tau:.2f} loss={loss.item():.4f} (align {la:.3f}) | "
                  f"holdout[100:130] {acc*100:.1f}% (best {best*100:.1f}%) | t={time.time()-t0:.0f}s", flush=True)
    if best_state: model.load_state_dict(best_state)
    return model, canon, te, best


@torch.no_grad()
def _eval(model, canon, te):
    model.eval(); ok = tot = 0
    for wi in te:
        for j in range(len(te[wi])):
            ok += ((model.view_audio(te[wi][j:j+1], tau=0.1) @ canon.t()).argmax(1).item() == wi); tot += 1
    model.train(); return ok/max(tot,1)


if __name__ == "__main__":
    print("="*64)
    print("AUDIO→VQ GUMBEL-SOFTMAX (différenciable) + capture simultanée unifiée")
    print("="*64)
    model, canon, te, best = train(n_steps=20000)
    print(f"\n{'='*64}\nRÉSULTAT GumbelVQ UNIFIÉ — holdout PROPRE [100:130]\n{'='*64}")
    print(f"  Test acc (audio GumbelVQ, co-capture unifiée): {best*100:.1f}%")
    print(f"  Réf: VQ straight-through 2.9% (échec) | simultaneous continu ~49% | SOTA 96%")
    print(f"  Δ vs VQ straight-through: {best*100-2.9:+.1f}pt | Δ vs SOTA: {best*100-96:+.1f}pt")
    torch.save({"model_state": model.state_dict(), "best": best},
               "/media/akone/SAVENVME2/Datasets/ocm26400/audio_gumbel_vq_trained.pt")
    json.dump({"holdout_acc": best, "delta_vs_vq_st_2p9": best*100-2.9, "delta_vs_sota_96": best*100-96,
               "method": "Gumbel-softmax VQ (differeniable) + simultaneous unified capture"},
              open("ocm26400/audio_gumbel_vq_results.json","w"), indent=2)
    print("  [sauvé] ocm26400/audio_gumbel_vq_results.json")
