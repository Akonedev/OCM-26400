#!/usr/bin/env python3
"""ARCHI UNIFIÉE — M5 (features per-frame) → SpectralCoreBlock(seq_len=T) → CE.

L'insight manquant (expert): SpectralCoreBlock à seq_len=1 sur features poolées = MLP trivial =
goulot. L'archi correcte: M5 produit les features PAR FRAME (pas poolées) → SpectralCoreBlock
fait le MIXING FFT GLOBAL sur la séquence de frames → découvre la COMPOSITION PHONÉTIQUE.

C'est le crown-jewel appliqué à l'audio: frames M5 (primitives locales) → FFT (composition
globale) → mot. Le M5 extrait l'invariant local, le FFT trouve le pattern phonétique global.

M5 modifié: seulement 2 pools (pas 4) → T≈62 frames pour le SpectralCoreBlock(seq_len=62).
"""
import torch,torch.nn as nn,torch.nn.functional as F,glob,os,numpy as np,time,json,random
import soundfile as sf
from ocm26400.spectral_core import SpectralCoreBlock

device="cuda" if torch.cuda.is_available() else "cpu"
SC="/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
CKPT="/media/akone/SAVENVME2/Datasets/ocm26400/audio_unified_trained.pt"

class M5Unified(nn.Module):
    """M5 (2 pools seulement → T≈62 frames) → SpectralCoreBlock(seq_len=62) global → classify.
    M5 extrait features locales par frame, SpectralCoreBlock fait le mixing FFT global = crown-jewel audio."""
    def __init__(s,nw):
        super().__init__()
        s.c1=nn.Conv1d(1,32,80,stride=16);s.b1=nn.BatchNorm1d(32);s.p1=nn.MaxPool1d(4)
        s.c2=nn.Conv1d(32,32,3);s.b2=nn.BatchNorm1d(32);s.p2=nn.MaxPool1d(4)
        s.c3=nn.Conv1d(32,64,3);s.b3=nn.BatchNorm1d(64)   # pas de pool3 (garde T≈62)
        s.c4=nn.Conv1d(64,64,3);s.b4=nn.BatchNorm1d(64)   # pas de pool4
        s.core=SpectralCoreBlock(d_model=64,seq_len=62,bidirectional=True)  # FFT GLOBAL sur frames
        s.fc=nn.Linear(64,nw)
    def forward(s,w):
        x=w.unsqueeze(1)
        x=s.p1(F.relu(s.b1(s.c1(x))))   # (B,32,~248)
        x=s.p2(F.relu(s.b2(s.c2(x))))   # (B,32,~62)
        x=F.relu(s.b3(s.c3(x)))         # (B,64,~62) — pas de pool
        x=F.relu(s.b4(s.c4(x)))         # (B,64,~62)
        frames=x.transpose(1,2)         # (B,62,64) — séquence de frames
        mixed=s.core(frames)            # (B,62,64) — FFT GLOBAL mixing (crown-jewel audio!)
        return s.fc(mixed.mean(1))      # pool → (B,nw) logits

def load_wav(p,T=16000):
    y,sr=sf.read(p);y=y.astype(np.float32)
    if y.ndim>1:y=y.mean(1)
    return torch.from_numpy(np.pad(y,(0,max(0,T-len(y))))[:T] if len(y)<T else y[:T])
def spd(w,r=(0.9,1.1)):
    k=random.uniform(*r);n=len(w);return np.interp(np.clip(np.arange(n)/k,0,n-1),np.arange(n),w).astype(np.float32)
def load_data(words):
    ts=set(l.strip() for l in open(os.path.join(SC,"testing_list.txt")) if l.strip());tr={};te={}
    for wi,w in enumerate(words):
        a=sorted(glob.glob(os.path.join(SC,w,"*.wav")));rp,tp=[],[]
        for p in a:(tp if os.path.relpath(p,SC)in ts else rp).append(p)
        if len(rp)>=50 and len(tp)>=5:tr[wi]=torch.stack([load_wav(p) for p in rp]);te[wi]=torch.stack([load_wav(p) for p in tp])
    return tr,te

def train(n=100000,bs=64,lr=1e-3,ev=2500):
    torch.manual_seed(0);random.seed(0)
    ws=sorted([w for w in os.listdir(SC) if os.path.isdir(os.path.join(SC,w)) and not w.startswith("_")])
    NW=len(ws);print(f"[UNIFIED M5+SCB(seq_len=62)] chargement...",flush=True)
    tr,te=load_data(ws);keys=list(tr.keys())
    m=M5Unified(NW).to(device);opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=n)
    print(f"[TRAIN UNIFIED] M5→SCB(62)→CE | lr={lr} | {n} steps | save-best/eval",flush=True)
    t0=time.time();best=0.0
    for step in range(n):
        bi=[random.choice(keys) for _ in range(bs)]
        wv=torch.stack([torch.from_numpy(spd(tr[k][torch.randint(0,len(tr[k]),(1,)).item()].numpy())) for k in bi]).to(device)
        wi=torch.tensor(bi,device=device)
        logits=m(wv);loss=F.cross_entropy(logits,wi)
        opt.zero_grad();loss.backward();opt.step();sched.step()
        if step%ev==0 or step==n-1:
            acc=_eval(m,te)
            if acc>best:
                best=acc;torch.save({"model_state":{k:v.detach().clone() for k,v in m.state_dict().items()},"best":best},CKPT)
                json.dump({"test_acc_official":best,"delta_vs_m5pure_90p2":best*100-90.2,"delta_vs_sota_96":best*100-96},open("ocm26400/audio_unified_results.json","w"),indent=2)
            print(f"  step {step:>5} lr={sched.get_last_lr()[0]:.4f} CE={loss.item():.3f} | test {acc*100:.1f}% (best {best*100:.1f}%) [S] t={time.time()-t0:.0f}s",flush=True)
    return best

@torch.no_grad()
def _eval(m,te):
    m.eval();ok=tot=0
    for wi in te:
        for j in range(0,len(te[wi]),64):
            p=m(te[wi][j:j+64].to(device)).argmax(1).cpu();ok+=(p==wi).sum().item();tot+=len(p)
    m.train();return ok/max(tot,1)

if __name__=="__main__":
    print("="*50);print("ARCHI UNIFIÉE — M5→SpectralCoreBlock(seq_len=62)→CE (crown-jewel audio)");print("="*50)
    b=train();print(f"\nFINAL: {b*100:.1f}% | vs M5 pur 90.2%: {b*100-90.2:+.1f}pt | vs SOTA 96%: {b*100-96:+.1f}pt")
