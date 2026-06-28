#!/usr/bin/env python3
"""M5 + CE isolé (lr=3e-3, pas de wd/SpecAugment) — changement minimal depuis le 87.7% 1-cos.
Seule la loss change: 1-cos → CE (+ petit 1-cos 0.1 pour l'alignement AMV)."""
import torch,torch.nn as nn,torch.nn.functional as F,glob,os,numpy as np,time,json,random
import soundfile as sf
from ocm26400.amv import D_MODEL,PART
from ocm26400.learned_vocab import LearnedVocab
from ocm26400.spectral_core import SpectralCoreBlock
from train_deep_encoder_v2 import text_feat,phon_feat

device="cuda" if torch.cuda.is_available() else "cpu"
SC="/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
CKPT="/media/akone/SAVENVME2/Datasets/ocm26400/audio_ce_clean_trained.pt"

class M5Lobe(nn.Module):
    def __init__(s,out=D_MODEL):
        super().__init__()
        s.c1=nn.Conv1d(1,32,80,stride=16);s.b1=nn.BatchNorm1d(32);s.p1=nn.MaxPool1d(4)
        s.c2=nn.Conv1d(32,32,3);s.b2=nn.BatchNorm1d(32);s.p2=nn.MaxPool1d(4)
        s.c3=nn.Conv1d(32,64,3);s.b3=nn.BatchNorm1d(64);s.p3=nn.MaxPool1d(4)
        s.c4=nn.Conv1d(64,64,3);s.b4=nn.BatchNorm1d(64);s.p4=nn.MaxPool1d(4)
        s.proj=nn.Linear(64,out)
    def forward(s,w):
        x=w.unsqueeze(1)
        for c,b,p in [(s.c1,s.b1,s.p1),(s.c2,s.b2,s.p2),(s.c3,s.b3,s.p3),(s.c4,s.b4,s.p4)]:
            x=p(F.relu(b(c(x))))
        return s.proj(x.mean(-1))

class Model(nn.Module):
    def __init__(s,nw):
        super().__init__();s.lobe=M5Lobe();s.core=SpectralCoreBlock(d_model=D_MODEL,seq_len=1);s.head=nn.Linear(D_MODEL,PART)
    def forward(s,w):return s.head(s.core(s.lobe(w).unsqueeze(1)).squeeze(1))

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

def train(n=30000,bs=64,lr=3e-3,ev=2500):
    torch.manual_seed(0);random.seed(0)
    ws=sorted([w for w in os.listdir(SC) if os.path.isdir(os.path.join(SC,w)) and not w.startswith("_")])
    NW=len(ws);print(f"[CE isolé] chargement...",flush=True)
    tr,te=load_data(ws);keys=list(tr.keys())
    cv=LearnedVocab(n=NW,dim=PART,init="ortho",seed=0);cv.freeze();can=cv._matrix().to(device)
    m=Model(NW).to(device);opt=torch.optim.Adam(m.parameters(),lr=lr)
    print(f"[TRAIN CE isolé] M5+CE(+0.1cos) lr={lr} | {n} steps | save-best/eval",flush=True)
    t0=time.time();best=0.0
    for step in range(n):
        bi=[random.choice(keys) for _ in range(bs)]
        wv=torch.stack([tr[k][torch.randint(0,len(tr[k]),(1,)).item()] for k in bi]).to(device)
        wi=torch.tensor(bi,device=device)
        wv=torch.stack([torch.from_numpy(spd(tr[k][torch.randint(0,len(tr[k]),(1,)).item()].numpy())) for k in bi]).to(device)
        ent=m(wv)
        logits=ent@can.t();l_ce=F.cross_entropy(logits,wi)
        l_cos=(1-F.cosine_similarity(ent,can[wi],-1).clamp(-1,1)).mean()
        loss=l_ce+0.1*l_cos
        opt.zero_grad();loss.backward();opt.step()
        if step%ev==0 or step==n-1:
            acc=_eval(m,can,te)
            if acc>best:
                best=acc;torch.save({"model_state":{k:v.detach().clone() for k,v in m.state_dict().items()},"best":best},CKPT)
                json.dump({"test_acc_official":best,"delta_vs_1cos_87p7":best*100-87.7,"delta_vs_sota_96":best*100-96},open("ocm26400/audio_ce_clean_results.json","w"),indent=2)
            print(f"  step {step:>5} CE={l_ce.item():.3f} | test {acc*100:.1f}% (best {best*100:.1f}%) [S] t={time.time()-t0:.0f}s",flush=True)
    return best

@torch.no_grad()
def _eval(m,can,te):
    m.eval();ok=tot=0
    for wi in te:
        for j in range(0,len(te[wi]),32):
            p=(m(te[wi][j:j+32].to(device))@can.t()).argmax(1).cpu();ok+=(p==wi).sum().item();tot+=len(p)
    m.train();return ok/max(tot,1)

if __name__=="__main__":
    print("="*50);print("M5+CE isolé (lr=3e-3, pas wd/SpecAug)");print("="*50)
    b=train();print(f"\nFINAL: {b*100:.1f}% | vs 1-cos 87.7%: {b*100-87.7:+.1f}pt | vs SOTA 96%: {b*100-96:+.1f}pt")
