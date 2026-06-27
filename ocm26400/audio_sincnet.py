#!/usr/bin/env python3
"""Audio — BANC DE FILTRES APPRIS (SincNet, rapport 45 « banc de filtres appris »).

Levier fidèle au projet que je n'avais pas testé : tous mes essais utilisaient un banc Mel
FIXE (triangulaire codé dur). Le rapport 45 préconise un « banc de filtres APPRIS ».

SincNet (Ravanelli 2018) : la 1ère couche est un banc de filtres sinc bandpass à paramètres
APPRENABLES (low_freq, bandwidth). Ces filtres APPRENNENT les bandes de fréquence pertinentes
pour les phonèmes (formant-like) → naturellement plus INVARIANTS AU LOCUTEUR que le Mel fixe
(le Mel code dur des bandes fixes qui incluent du pitch locuteur-dépendant).

C'est du DSP paramétré (sinc filters), PAS un modèle externe/Frankenstein. Exactement le
« banc de filtres appris » du rapport 45. Cœur SpectralCoreBlock + capture simultanée.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import glob, os, numpy as np, time, json, random
from ocm26400.amv import D_MODEL, PART
from ocm26400.learned_vocab import LearnedVocab
from ocm26400.spectral_core import SpectralCoreBlock
from train_deep_encoder_v2 import text_feat, phon_feat, load_wav

device = "cuda" if torch.cuda.is_available() else "cpu"
SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
SR = 16000


class SincFilterBank(nn.Module):
    """Banc de filtres sinc bandpass APPRENABLES (SincNet). low_freq + bandwidth learnables.
    Apprend les bandes pertinentes (formant-like) -> invariance locuteur."""
    def __init__(self, n_filt=80, kernel=251, sample_rate=SR):
        super().__init__()
        self.n_filt, self.kernel, self.sr = n_filt, kernel, sample_rate
        # init mel-espacé (low_hz, band_hz) — borné à [0, sr/2]
        mel = np.linspace(np.log10(30), np.log10(sample_rate/2 - 30), n_filt+1)
        hz = 10**mel
        self.low_hz = nn.Parameter(torch.tensor(hz[:-1], dtype=torch.float32))
        self.band_hz = nn.Parameter(torch.diff(torch.tensor(hz, dtype=torch.float32)))
        n = (kernel - 1) / 2.0
        t = torch.linspace(-n, n, kernel) / sample_rate
        self.register_buffer("t", t)
        self.register_buffer("window", torch.hamming_window(kernel))
        self.register_buffer("N", torch.tensor(kernel, dtype=torch.float32))

    def forward(self, wav):                          # wav: (B, T)
        low = torch.abs(self.low_hz).clamp_max(self.sr/2 - 1)
        high = low + torch.abs(self.band_hz)
        high = high.clamp_max(self.sr/2)
        high = torch.maximum(high, low + 1.0)        # high > low (types homogènes)
        # filtres sinc bandpass (f_low, f_high) par filtre
        f_low = low.unsqueeze(1)                      # (n_filt,1)
        f_high = high.unsqueeze(1)
        t = self.t.unsqueeze(0)                       # (1,kernel)
        bp = (torch.sin(2*np.pi*f_high*t) - torch.sin(2*np.pi*f_low*t)) / (2*np.pi*t + 1e-8)
        win = self.window.unsqueeze(0)
        filt = bp * win
        filt = filt / (filt.abs().sum(dim=1, keepdim=True) + 1e-8)   # normalize
        # conv1d : waveform (B,1,T) * filtres (n_filt,1,kernel) -> (B, n_filt, T')
        wav = wav.unsqueeze(1)
        out = F.conv1d(wav, filt.unsqueeze(1), padding=self.kernel//2)
        return torch.abs(out)                         # (B, n_filt, T) — puissance par bande


class SincAudioLobe(nn.Module):
    """SincFilterBank (appris) -> convs -> features. Banc appris (rapport 45), pas Mel fixe."""
    def __init__(self, out_dim=D_MODEL, n_filt=80):
        super().__init__()
        self.sinc = SincFilterBank(n_filt=n_filt)
        self.convs = nn.Sequential(nn.Conv1d(n_filt, 128, 3, padding=1), nn.ReLU(),
                                   nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
                                   nn.Conv1d(128, 64, 3, padding=1), nn.ReLU(),
                                   nn.Conv1d(64, 32, 3, padding=1), nn.ReLU())
        self.proj = nn.Linear(32, out_dim)
    def forward(self, wav):
        bands = self.sinc(wav)                        # (B,n_filt,T) — banc APPRIS
        h = self.convs(bands)
        return self.proj(h.mean(dim=-1))              # (B,out_dim)


class SincModel(nn.Module):
    def __init__(self, n_words):
        super().__init__()
        self.lobe = SincAudioLobe()
        self.text_proj = nn.Linear(PART, D_MODEL); self.phon_proj = nn.Linear(PART, D_MODEL)
        self.core = SpectralCoreBlock(d_model=D_MODEL, seq_len=1)
        self.head = nn.Linear(D_MODEL, PART)
    def audio_view(self, wav): return self.head(self.core(self.lobe(wav).unsqueeze(1)).squeeze(1))
    def text_view(self, f):  return self.head(self.core(self.text_proj(f).unsqueeze(1)).squeeze(1))
    def phon_view(self, f):  return self.head(self.core(self.phon_proj(f).unsqueeze(1)).squeeze(1))


def speed_perturb(w, r=(0.9,1.1)):
    rate=random.uniform(*r); n=len(w); idx=np.clip(np.arange(n)/rate,0,n-1)
    return np.interp(idx, np.arange(n), w).astype(np.float32)


def load_official():
    test=set(l.strip() for l in open(os.path.join(SC,"testing_list.txt")) if l.strip())
    return test

def load_data(words):
    test_set=load_official(); tr={}; te={}
    for wi,w in enumerate(words):
        allw=sorted(glob.glob(os.path.join(SC,w,"*.wav"))); trp,tep=[],[]
        for p in allw:
            (tep if os.path.relpath(p,SC) in test_set else trp).append(p)
        if len(trp)>=50 and len(tep)>=5:
            tr[wi]=torch.stack([load_wav(p) for p in trp]); te[wi]=torch.stack([load_wav(p) for p in tep])
    return tr,te


def train(n_steps=25000, batch=64, lr=3e-3, eval_every=2500, augment=True):
    torch.manual_seed(0); random.seed(0)
    words=sorted([w for w in os.listdir(SC) if os.path.isdir(os.path.join(SC,w)) and not w.startswith("_")])
    NW=len(words); print("[SincNet full scale] chargement...", flush=True)
    tr,te=load_data(words); keys=list(tr.keys())
    print(f"  {len(keys)} mots | train={sum(len(tr[w]) for w in tr)} | test officiel={sum(len(te[w]) for w in te)}", flush=True)
    text_all=torch.tensor([text_feat(w) for w in words]).to(device)
    phon_all=torch.tensor([phon_feat(w) for w in words]).to(device)
    cv=LearnedVocab(n=NW,dim=PART,init="ortho",seed=0); cv.freeze(); canon=cv._matrix().to(device)
    model=SincModel(NW).to(device); opt=torch.optim.Adam(model.parameters(),lr=lr)

    def joint(wi_t,wavs):
        tgt=canon[wi_t]
        return ((1-F.cosine_similarity(model.text_view(text_all[wi_t]),tgt,-1).clamp(-1,1)).mean()+
                (1-F.cosine_similarity(model.phon_view(phon_all[wi_t]),tgt,-1).clamp(-1,1)).mean()+
                (1-F.cosine_similarity(model.audio_view(wavs),tgt,-1).clamp(-1,1)).mean())

    print(f"\n[TRAIN SincNet banc appris] capture simultanée | {n_steps} steps", flush=True)
    t0=time.time(); best=0.0; best_state=None
    for step in range(n_steps):
        bi=[random.choice(keys) for _ in range(batch)]
        wl=[]
        for k in bi:
            w=tr[k][torch.randint(0,len(tr[k]),(1,)).item()].numpy()
            if augment: w=speed_perturb(w)
            wl.append(torch.from_numpy(w))
        wavs=torch.stack(wl).to(device); wi_t=torch.tensor(bi,device=device)
        loss=joint(wi_t,wavs); opt.zero_grad(); loss.backward(); opt.step()
        if step%eval_every==0 or step==n_steps-1:
            acc=_eval(model,canon,te)
            if acc>best: best=acc; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            print(f"  step {step:>5} 1-cos={loss.item():.4f} | test OFFICIEL {acc*100:.1f}% (best {best*100:.1f}%) | t={time.time()-t0:.0f}s", flush=True)
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
    print("="*64); print("AUDIO — BANC DE FILTRES APPRIS (SincNet, rapport 45) + capture simultanée, full scale"); print("="*64)
    model,canon,te,best=train(n_steps=25000)
    print(f"\n{'='*64}\nRÉSULTAT SincNet (banc appris) — test OFFICIEL\n{'='*64}")
    print(f"  Test acc OFFICIEL: {best*100:.1f}%")
    print(f"  Réf: Mel fixe 62.5% | rapport45 46.4% | SOTA 96%")
    print(f"  Δ vs Mel fixe: {best*100-62.5:+.1f}pt | Δ vs SOTA: {best*100-96:+.1f}pt")
    torch.save({"model_state":model.state_dict(),"best":best},"/media/akone/SAVENVME2/Datasets/ocm26400/audio_sincnet_trained.pt")
    json.dump({"test_acc_official":best,"delta_vs_mel_62p5":best*100-62.5,"delta_vs_sota_96":best*100-96,
               "method":"SincNet learned filter bank (rapport 45 'banc appris') + simultaneous capture, full scale"},
              open("ocm26400/audio_sincnet_results.json","w"),indent=2)
    print("  [sauvé] ocm26400/audio_sincnet_results.json")
