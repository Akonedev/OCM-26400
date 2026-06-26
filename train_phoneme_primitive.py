#!/usr/bin/env python3
"""Phonème-primitives grokkées → composition → génération depuis compréhension."""
import torch, torch.nn as nn, torch.nn.functional as F, glob, os, numpy as np, time
import soundfile as sf
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.amv import D_MODEL, PART
from ocm26400.learned_vocab import LearnedVocab
device="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0)
SC="/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
T=8000; N_MELS=32; N_FRAMES=16; n_fft=256; hop=T//(N_FRAMES+1)
win=np.hanning(n_fft)
mel_fb=np.zeros((N_MELS,n_fft//2+1))
for m in range(N_MELS):
    c=(m+1)*(n_fft//2)/(N_MELS+1)
    for f in range(n_fft//2+1): mel_fb[m,f]=max(0,1-abs(f-c)/max(1,n_fft//2/(N_MELS+1)))
mel_fb=mel_fb/(mel_fb.sum(axis=1,keepdims=True)+1e-8)
def extract_mel(y):
    if len(y)<T: y=np.pad(y,(0,T-len(y)))
    else: y=y[:T]
    frames=[]
    for s in range(0,T-n_fft,hop):
        fft=np.fft.rfft(y[s:s+n_fft]*win); mel_fb_local=mel_fb
        frames.append(np.log1p(mel_fb_local@(np.abs(fft)**2)))
    while len(frames)<N_FRAMES: frames.append(frames[-1] if frames else np.zeros(N_MELS))
    return np.array(frames[:N_FRAMES],dtype=np.float32)
PHON_MAP={chr(i):(i-97)%20 for i in range(97,123)}
def word_to_phon(w):
    ids=[PHON_MAP.get(c,0) for c in w.lower()][:N_FRAMES]
    while len(ids)<N_FRAMES: ids.append(20)
    return ids
words=sorted([w for w in os.listdir(SC) if os.path.isdir(os.path.join(SC,w)) and not w.startswith("_")])
NW=len(words); NPW=5
print(f"[phonème-primitive grok] {NW} mots, {NPW}/mot={NW*NPW} total (comprehension>memory)",flush=True)
mel_s=[];phon_s=[];wid_s=[]
for wi,w in enumerate(words):
    for p in glob.glob(os.path.join(SC,w,"*.wav"))[:NPW]:
        y,sr=sf.read(p);y=y.astype(np.float32)
        if y.ndim>1:y=y.mean(1)
        mel_s.append(extract_mel(y));phon_s.append(word_to_phon(w));wid_s.append(wi)
mel_t=torch.tensor(np.array(mel_s)).to(device)
phon_t=torch.tensor(np.array(phon_s),dtype=torch.long).to(device)
wid_t=torch.tensor(wid_s).to(device);N=len(wid_s)
cv=LearnedVocab(n=NW,dim=PART,init="ortho" if NW<=PART else"random",seed=0);cv.freeze()
canon=cv._matrix().to(device)
perm=torch.randperm(N);n_tr=int(N*0.8);tr_i,te_i=perm[:n_tr],perm[n_tr:]

class Model(nn.Module):
    def __init__(s):
        super().__init__()
        s.pe=nn.Embedding(21,D_MODEL)  # 20 phon + PAD
        s.pc=SpectralCoreBlock(d_model=D_MODEL,seq_len=N_FRAMES,bidirectional=True)
        s.ph=nn.Linear(D_MODEL,PART)
        s.mp=nn.Linear(N_MELS,D_MODEL)
        s.ac=SpectralCoreBlock(d_model=D_MODEL,seq_len=N_FRAMES,bidirectional=True)
        s.ah=nn.Linear(D_MODEL,PART)
        # génération: concept(PART) → proj(D_MODEL) → core → Mel
        s.gp=nn.Linear(PART,D_MODEL)
        s.gc=SpectralCoreBlock(d_model=D_MODEL,seq_len=N_FRAMES,bidirectional=True)
        s.gh=nn.Linear(D_MODEL,N_MELS)
    def fwd_phon(s,ids):
        x=s.pe(ids);return s.ph(s.pc(x).mean(1))
    def fwd_audio(s,mel):
        x=s.mp(mel);return s.ah(s.ac(x).mean(1))
    def generate(s,concept):
        x=s.gp(concept).unsqueeze(1).expand(-1,N_FRAMES,-1)
        return s.gh(s.gc(x))  # (B,N_FRAMES,N_MELS) Mel généré

m=Model().to(device);opt=torch.optim.Adam(m.parameters(),lr=3e-3)
print(f"\n[GROK phonème→concept + audio→concept + concept→GÉNÉRER Mel — simultané]",flush=True)
t0=time.time()
for step in range(20000):
    bi=tr_i[torch.randint(0,len(tr_i),(32,))];tgt=canon[wid_t[bi]]
    out_p=m.fwd_phon(phon_t[bi]);out_a=m.fwd_audio(mel_t[bi])
    gen=m.generate(out_p.detach())
    loss_p=(1-F.cosine_similarity(out_p,tgt).clamp(-1,1)).mean()
    loss_a=(1-F.cosine_similarity(out_a,tgt).clamp(-1,1)).mean()
    loss_g=F.mse_loss(gen,mel_t[bi])
    loss=loss_p+loss_a+0.5*loss_g
    opt.zero_grad();loss.backward();opt.step()
    if step%4000==0:
        m.eval()
        with torch.no_grad():
            ok_a=sum(1 for j in te_i.tolist() if(m.fwd_audio(mel_t[j:j+1])@canon.t()).argmax(1).item()==wid_t[j].item())
            ok_p=sum(1 for j in te_i.tolist() if(m.fwd_phon(phon_t[j:j+1])@canon.t()).argmax(1).item()==wid_t[j].item())
            gen_ok=0
            for j in te_i.tolist()[:20]:
                g=m.generate(m.fwd_phon(phon_t[j:j+1]))
                gen_ok+=((m.fwd_audio(g)@canon.t()).argmax(1).item()==wid_t[j].item())
        print(f"  step {step:>5} loss={loss.item():.3f}|p={loss_p.item():.3f} a={loss_a.item():.3f} g={loss_g.item():.3f}|reco_a={ok_a}/{len(te_i)} reco_p={ok_p}/{len(te_i)} gen={gen_ok}/20 t={time.time()-t0:.0f}s",flush=True)
        m.train()
m.eval()
with torch.no_grad():
    ok_a=sum(1 for j in te_i.tolist() if(m.fwd_audio(mel_t[j:j+1])@canon.t()).argmax(1).item()==wid_t[j].item())
    ok_p=sum(1 for j in te_i.tolist() if(m.fwd_phon(phon_t[j:j+1])@canon.t()).argmax(1).item()==wid_t[j].item())
    gen_ok=0
    for j in te_i.tolist()[:50]:
        g=m.generate(m.fwd_phon(phon_t[j:j+1]))  # (1,N_FRAMES,N_MELS)
        pred=(m.fwd_audio(g)@canon.t()).argmax(1).item()
        gen_ok+=(pred==wid_t[j].item())
print(f"\n{'='*60}")
print(f"PHONÈME-PRIMITIVES → COMPOSITION → GÉNÉRATION (comprehension)")
print(f"{'='*60}")
print(f"  données: {NPW}/mot ({N} total — comprehension > mémoire)")
print(f"  RECO audio: {ok_a}/{len(te_i)}={ok_a/max(len(te_i),1)*100:.1f}%")
print(f"  RECO phonème: {ok_p}/{len(te_i)}={ok_p/max(len(te_i),1)*100:.1f}%")
print(f"  GÉNÉRATION (concept→Mel→reconnu): {gen_ok}/50={gen_ok/50*100:.0f}%")
print(f"  temps: {time.time()-t0:.0f}s")
torch.save({"model_state":m.state_dict(),"canon":canon,"reco_audio":ok_a/max(len(te_i),1),
            "reco_phon":ok_p/max(len(te_i),1),"gen_acc":gen_ok/50,"words":words,"n_per_word":NPW},
    "/media/akone/SAVENVME2/Datasets/ocm26400/phoneme_primitive_grok.pt")
print(f"  [SAUVÉ]")
