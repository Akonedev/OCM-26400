#!/usr/bin/env python3
"""MODÈLE PARADIGME COMPLET — UN seul modèle, tous les principes respectés.

C'est l'expérience qui n'a JAMAIS été faite correctement:
  1. M5 lobe → features per-frame (T≈62)     [Lobe Licensing]
  2. SpectralCoreBlock(seq_len=62) FFT global   [crown-jewel sur audio]
  3. Capture SIMULTANÉE text+phon+audio→canon  [associations, §I]
  4. Loss 1-COS (PAS CE) — grokking             [crown-jewel, train_binary_block]
  5. Gate meta[0] entraîné (CONF_TARGET=4.0)   [observateur, §J]
  6. Adam 3e-3, seed 0                          [canonique]
  7. Save-best/eval + watch for GROK phase      [grokking detection]

Si le GROK se déclenche (phase transition test acc), l'accuracy SAUTE (pas graduel).
C'est ce que le paradigme prédit: Apprendre→Comprendre→Raisonner→Générer.
"""
import torch,torch.nn as nn,torch.nn.functional as F,glob,os,numpy as np,time,json,random
import soundfile as sf
from ocm26400.amv import D_MODEL, PART
from ocm26400.learned_vocab import LearnedVocab
from ocm26400.spectral_core import SpectralCoreBlock
from train_deep_encoder_v2 import text_feat, phon_feat, load_wav

device="cuda"
SC="/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
CONF_TARGET=4.0  # sigmoid(4)≈0.98 > TAU_GROK=0.9 (reasoner.py)
CKPT="/media/akone/SAVENVME2/Datasets/ocm26400/audio_paradigm_grok_trained.pt"

class M5Lobe(nn.Module):
    """M5 lobe (2 pools → T≈62 frames) — features per-frame pour le SCB."""
    def __init__(s):
        super().__init__()
        s.c1=nn.Conv1d(1,32,80,stride=16);s.b1=nn.BatchNorm1d(32);s.p1=nn.MaxPool1d(4)
        s.c2=nn.Conv1d(32,32,3);s.b2=nn.BatchNorm1d(32);s.p2=nn.MaxPool1d(4)
        s.c3=nn.Conv1d(32,64,3);s.b3=nn.BatchNorm1d(64)
        s.c4=nn.Conv1d(64,64,3);s.b4=nn.BatchNorm1d(64)
    def forward(s,w):
        x=w.unsqueeze(1)
        x=s.p1(F.relu(s.b1(s.c1(x))));x=s.p2(F.relu(s.b2(s.c2(x))))
        x=F.relu(s.b3(s.c3(x)));x=F.relu(s.b4(s.c4(x)))
        return x.transpose(1,2)  # (B,~62,64) frames per-frame

class ParadigmModel(nn.Module):
    """UN seul modèle. Capture simultanée + 1-cos grok + gate meta[0] + SCB(62)."""
    def __init__(s,nw):
        super().__init__()
        s.lobe=M5Lobe()
        s.text_proj=nn.Linear(PART,64);s.phon_proj=nn.Linear(PART,64)
        s.core=SpectralCoreBlock(d_model=64,seq_len=62,bidirectional=True)
        s.head=nn.Linear(64,PART)        # → ent (PART=64) pour 1-cos vs canonical
        s.gate_head=nn.Linear(64,1)      # → meta[0] confidence (gate/observer)
    def audio_frames(s,w):return s.lobe(w)  # (B,62,64)
    def text_frames(s,f):return s.text_proj(f).unsqueeze(1).expand(-1,62,-1)  # broadcast
    def phon_frames(s,f):return s.phon_proj(f).unsqueeze(1).expand(-1,62,-1)
    def process(s,frames):
        mixed=s.core(frames)            # FFT global (crown-jewel audio!)
        pooled=mixed.mean(1)            # (B,64)
        ent=s.head(pooled)              # (B,PART) pour 1-cos
        conf=s.gate_head(pooled).squeeze(-1)  # (B,) meta[0] gate
        return ent,conf

def speed_perturb(w,r=(0.9,1.1)):
    k=random.uniform(*r);n=len(w);return np.interp(np.clip(np.arange(n)/k,0,n-1),np.arange(n),w).astype(np.float32)
def load_data(words):
    ts=set(l.strip() for l in open(os.path.join(SC,"testing_list.txt")) if l.strip());tr={};te={}
    for wi,w in enumerate(words):
        a=sorted(glob.glob(os.path.join(SC,w,"*.wav")));rp,tp=[],[]
        for p in a:(tp if os.path.relpath(p,SC)in ts else rp).append(p)
        if len(rp)>=50 and len(tp)>=5:tr[wi]=torch.stack([load_wav(p) for p in rp]);te[wi]=torch.stack([load_wav(p) for p in tp])
    return tr,te

def train(n=50000,bs=64,lr=3e-3,ev=1000):
    """Entraînement PARADIGME: 1-cos + capture simultanée + gate. Watch for GROK."""
    torch.manual_seed(0);random.seed(0)
    ws=sorted([w for w in os.listdir(SC) if os.path.isdir(os.path.join(SC,w)) and not w.startswith("_")])
    NW=len(ws);print(f"[PARADIGME GROK] chargement...",flush=True)
    tr,te=load_data(ws);keys=list(tr.keys())
    text_all=torch.tensor([text_feat(w) for w in ws]).to(device)
    phon_all=torch.tensor([phon_feat(w) for w in ws]).to(device)
    cv=LearnedVocab(n=NW,dim=PART,init="ortho",seed=0);cv.freeze()
    canon=cv._matrix().to(device)
    m=ParadigmModel(NW).to(device)
    opt=torch.optim.Adam(m.parameters(),lr=lr)  # Adam 3e-3 (canonique, PAS AdamW)

    print(f"\n[TRAIN PARADIGME] M5→SCB(62)+1-cos+capture simultanée+gate | seed0 | {n} steps",flush=True)
    print(f"  Watch for GROK phase transition (test acc jump)!",flush=True)
    t0=time.time();best=0;prev_acc=0;grok_detected=False
    for step in range(n):
        bi=[random.choice(keys) for _ in range(bs)]
        wv=torch.stack([torch.from_numpy(speed_perturb(tr[k][torch.randint(0,len(tr[k]),(1,)).item()].numpy())) for k in bi]).to(device)
        wi=torch.tensor(bi,device=device);tgt=canon[wi]
        # CAPTURE SIMULTANÉE: 3 vues (text+phon+audio) → même canonical
        ent_a,conf_a=m.process(m.audio_frames(wv))
        ent_t,conf_t=m.process(m.text_frames(text_all[wi]))
        ent_p,conf_p=m.process(m.phon_frames(phon_all[wi]))
        # LOSS 1-COS (crown-jewel, PAS CE) sur les 3 vues
        l_cos=((1-F.cosine_similarity(ent_a,tgt,-1).clamp(-1,1)).mean()+
               (1-F.cosine_similarity(ent_t,tgt,-1).clamp(-1,1)).mean()+
               (1-F.cosine_similarity(ent_p,tgt,-1).clamp(-1,1)).mean())/3
        # GATE: meta[0]→CONF_TARGET quand correct (observateur)
        cos_a=F.cosine_similarity(ent_a,tgt,-1).clamp(-1,1).detach()
        correct_mask=(cos_a>0.5).float()
        l_gate=((conf_a-CONF_TARGET*correct_mask)**2).mean()
        loss=l_cos+0.5*l_gate
        opt.zero_grad();loss.backward();opt.step()
        if step%ev==0 or step==n-1:
            m.eval();ok=tot=0;conf_ok=0;conf_tot=0
            with torch.no_grad():
                for w in te:
                    for j in range(0,len(te[w]),64):
                        wavs=te[w][j:j+64].to(device)
                        ent,c=m.process(m.audio_frames(wavs))
                        pred=(ent@canon.t()).argmax(1).cpu()
                        ok+=(pred==w).sum().item();tot+=len(pred)
                        # selective: confident AND correct?
                        conf_mask=(torch.sigmoid(c)>0.9).cpu()
                        conf_tot+=conf_mask.sum().item()
                        conf_ok+=((pred==w)&conf_mask).sum().item()
            m.train();acc=ok/max(tot,1)
            # GROK detection: phase transition (sudden jump > 5pt)
            if acc-prev_acc>0.05 and step>1000:
                grok_detected=True
                print(f"  *** GROK DÉTECTÉ! acc {prev_acc*100:.1f}%→{acc*100:.1f}% (+{(acc-prev_acc)*100:.1f}pt) step {step} ***",flush=True)
            prev_acc=acc
            if acc>best:
                best=acc
                torch.save({"model_state":{k:v.detach().clone() for k,v in m.state_dict().items()},"best":best},
                           CKPT)
            sel=conf_ok/max(conf_tot,1) if conf_tot>0 else 0
            print(f"  step {step:>5} 1-cos={l_cos.item():.4f} gate={l_gate.item():.4f} | test {acc*100:.1f}% (best {best*100:.1f}%) sel {sel*100:.0f}%@{conf_tot} t={time.time()-t0:.0f}s",flush=True)
    print(f"\nFINAL: {best*100:.1f}% | GROK détecté: {grok_detected}")
    return best,grok_detected

if __name__=="__main__":
    print("="*60)
    print("MODÈLE PARADIGME COMPLET — UN modèle, 1-cos grok, capture simultanée, gate")
    print("Apprendre→Comprendre→Raisonner→Générer")
    print("="*60)
    best,grok=train()
    json.dump({"best":best,"grok_detected":grok,"method":"paradigm: 1cos+capture+gate+SCB(62)+M5"},
              open("ocm26400/audio_paradigm_grok_results.json","w"),indent=2)
