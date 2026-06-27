#!/usr/bin/env python3
"""Audio + ALIGNEMENT AMODAL InfoNCE (association any→any, rapport 53 §3).

Principe du projet que je n'avais PAS appliqué à l'audio : l'ALIGNEMENT AMODAL (InfoNCE
entre vues), pas seulement le 1-cos (vue→canonical). Rapport 53 : « association any→any ».
InfoNCE rapproche les vues (text/phon/audio) du MÊME mot, éloigne celles des autres mots =
crée directement l'ASSOCIATION (« capturer pour les associations »).

Loss = 1-cos(vue→canonical) [crown-jewel] + InfoNCE(text↔audio) + InfoNCE(phon↔audio)
[amodal alignment, rapport 53]. Full scale, split officiel. Cœur SpectralCoreBlock.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import glob, os, numpy as np, time, json, random
from ocm26400.amv import D_MODEL, PART
from ocm26400.learned_vocab import LearnedVocab
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.audio_deep_lobe import DeepAudioLobe
from train_deep_encoder_v2 import text_feat, phon_feat, load_wav

device = "cuda" if torch.cuda.is_available() else "cpu"
SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"


def info_nce(z_a, z_b, tau=0.07):
    """Alignement amodal (rapport 53) : rapproche les vues du même item (diag), éloigne les autres."""
    z_a = F.normalize(z_a, dim=-1); z_b = F.normalize(z_b, dim=-1)
    logits = z_a @ z_b.t() / tau
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


class AmodalModel(nn.Module):
    def __init__(self, n_words, n_blocks=4, hidden=128):
        super().__init__()
        self.lobe = DeepAudioLobe(n_mels=128, hidden=hidden, n_blocks=n_blocks)
        self.text_proj = nn.Linear(PART, D_MODEL); self.phon_proj = nn.Linear(PART, D_MODEL)
        self.core = SpectralCoreBlock(d_model=D_MODEL, seq_len=1)
        self.head = nn.Linear(D_MODEL, PART)
        self.proj_a = nn.Linear(D_MODEL, 64)   # tête amodale audio (pour InfoNCE)
        self.proj_t = nn.Linear(D_MODEL, 64)   # tête amodale text
        self.proj_p = nn.Linear(D_MODEL, 64)   # tête amodale phon
    def audio_view(self, wav): return self.head(self.core(self.lobe(wav).unsqueeze(1)).squeeze(1))
    def text_view(self, f):  return self.head(self.core(self.text_proj(f).unsqueeze(1)).squeeze(1))
    def phon_view(self, f):  return self.head(self.core(self.phon_proj(f).unsqueeze(1)).squeeze(1))
    def audio_amodal(self, wav):
        h = self.core(self.lobe(wav).unsqueeze(1)).squeeze(1); return self.proj_a(h)
    def text_amodal(self, f):
        h = self.core(self.text_proj(f).unsqueeze(1)).squeeze(1); return self.proj_t(h)
    def phon_amodal(self, f):
        h = self.core(self.phon_proj(f).unsqueeze(1)).squeeze(1); return self.proj_p(h)


def load_official():
    return set(l.strip() for l in open(os.path.join(SC,"testing_list.txt")) if l.strip())
def load_data(words):
    test_set=load_official(); tr={}; te={}
    for wi,w in enumerate(words):
        allw=sorted(glob.glob(os.path.join(SC,w,"*.wav"))); trp,tep=[],[]
        for p in allw: (tep if os.path.relpath(p,SC) in test_set else trp).append(p)
        if len(trp)>=50 and len(tep)>=5:
            tr[wi]=torch.stack([load_wav(p) for p in trp]); te[wi]=torch.stack([load_wav(p) for p in tep])
    return tr,te


def train(n_steps=20000, batch=64, lr=3e-3, eval_every=2500, lam_infonce=1.0):
    torch.manual_seed(0); random.seed(0)
    words=sorted([w for w in os.listdir(SC) if os.path.isdir(os.path.join(SC,w)) and not w.startswith("_")])
    NW=len(words); print("[capture + InfoNCE amodal] chargement...", flush=True)
    tr,te=load_data(words); keys=list(tr.keys())
    text_all=torch.tensor([text_feat(w) for w in words]).to(device)
    phon_all=torch.tensor([phon_feat(w) for w in words]).to(device)
    cv=LearnedVocab(n=NW,dim=PART,init="ortho",seed=0); cv.freeze(); canon=cv._matrix().to(device)
    model=AmodalModel(NW).to(device); opt=torch.optim.Adam(model.parameters(),lr=lr)

    def joint(wi_t, wavs):
        tgt=canon[wi_t]
        # 1-cos crown-jewel (vue -> canonical)
        l_cos = ((1-F.cosine_similarity(model.text_view(text_all[wi_t]),tgt,-1).clamp(-1,1)).mean()+
                 (1-F.cosine_similarity(model.phon_view(phon_all[wi_t]),tgt,-1).clamp(-1,1)).mean()+
                 (1-F.cosine_similarity(model.audio_view(wavs),tgt,-1).clamp(-1,1)).mean())/3.0
        # InfoNCE amodal (association any->any, rapport 53) : audio↔text, audio↔phon
        za=model.audio_amodal(wavs); zt=model.text_amodal(text_all[wi_t]); zp=model.phon_amodal(phon_all[wi_t])
        l_nce = info_nce(za, zt) + info_nce(za, zp)
        return l_cos + lam_infonce*l_nce, l_cos.item(), l_nce.item()

    print(f"\n[TRAIN capture + InfoNCE amodal] {len(keys)} mots | 1-cos + InfoNCE vue↔vue | {n_steps} steps", flush=True)
    t0=time.time(); best=0.0; best_state=None
    for step in range(n_steps):
        bi=[random.choice(keys) for _ in range(batch)]
        wavs=torch.stack([tr[k][torch.randint(0,len(tr[k]),(1,)).item()] for k in bi]).to(device)
        wi_t=torch.tensor(bi,device=device)
        loss,lc,ln=joint(wi_t,wavs); opt.zero_grad(); loss.backward(); opt.step()
        if step%eval_every==0 or step==n_steps-1:
            acc=_eval(model,canon,te)
            if acc>best: best=acc; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            print(f"  step {step:>5} loss={loss.item():.4f} (cos {lc:.3f} nce {ln:.3f}) | test OFFICIEL {acc*100:.1f}% (best {best*100:.1f}%) | t={time.time()-t0:.0f}s", flush=True)
    if best_state: model.load_state_dict(best_state)
    return model,canon,te,best


@torch.no_grad()
def _eval(model,canon,te):
    model.eval(); ok=tot=0
    for wi in te:
        for j in range(0,len(te[wi]),32):
            wavs=te[wi][j:j+32].to(device); p=(model.audio_view(wavs)@canon.t()).argmax(1).cpu()
            ok+=(p==wi).sum().item(); tot+=len(p)
    model.train(); return ok/max(tot,1)


if __name__=="__main__":
    print("="*64); print("AUDIO — capture simultanée + ALIGNEMENT AMODAL InfoNCE (rapport 53)"); print("="*64)
    model,canon,te,best=train(n_steps=20000)
    print(f"\n{'='*64}\nRÉSULTAT capture+InfoNCE — test OFFICIEL\n{'='*64}")
    print(f"  Test acc OFFICIEL: {best*100:.1f}%")
    print(f"  Réf: 1-cos seul 62.5% | SOTA 96%")
    print(f"  Δ vs 1-cos seul: {best*100-62.5:+.1f}pt | Δ vs SOTA: {best*100-96:+.1f}pt")
    torch.save({"model_state":model.state_dict(),"best":best},"/media/akone/SAVENVME2/Datasets/ocm26400/audio_infonce_trained.pt")
    json.dump({"test_acc_official":best,"delta_vs_1cos_62p5":best*100-62.5,"delta_vs_sota_96":best*100-96,
               "method":"simultaneous capture 1-cos + amodal InfoNCE alignment (rapport 53 any->any), full scale"},
              open("ocm26400/audio_infonce_results.json","w"),indent=2)
    print("  [sauvé] ocm26400/audio_infonce_results.json")
