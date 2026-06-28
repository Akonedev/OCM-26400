#!/usr/bin/env python3
"""BOUCLE LSRA D'AUTO-CORRECTION audio — observer → +itérations SCB → correction.

Le système cognitif (user vision):
  1. OBSERVER (confidence gate): détecte l'incertain (softmax max < τ)
  2. AGENT CORRECTION: quand incertain, itérer le SpectralCoreBlock PLUS de fois (LSRA depth)
     → raisonnement phonétique plus profond → prédiction corrigée
  3. AUTO-CORRECTION: la prédiction corrigée (post-LSRA) est restituée

C'est le crown-jewel sur l'incertain: more reasoning steps = correct.
Applique au modèle unifié entraîné (M5→SCB(62)→CE).
"""
import torch,torch.nn as nn,torch.nn.functional as F,glob,os,numpy as np,json
import soundfile as sf
from ocm26400.spectral_core import SpectralCoreBlock

device="cuda" if torch.cuda.is_available() else "cpu"
SC="/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"

class M5Unified(nn.Module):
    def __init__(s,nw):
        super().__init__()
        s.c1=nn.Conv1d(1,32,80,stride=16);s.b1=nn.BatchNorm1d(32);s.p1=nn.MaxPool1d(4)
        s.c2=nn.Conv1d(32,32,3);s.b2=nn.BatchNorm1d(32);s.p2=nn.MaxPool1d(4)
        s.c3=nn.Conv1d(32,64,3);s.b3=nn.BatchNorm1d(64);s.c4=nn.Conv1d(64,64,3);s.b4=nn.BatchNorm1d(64)
        s.core=SpectralCoreBlock(d_model=64,seq_len=62,bidirectional=True)
        s.fc=nn.Linear(64,nw)
    def extract_frames(s,w):
        x=w.unsqueeze(1)
        x=s.p1(F.relu(s.b1(s.c1(x))));x=s.p2(F.relu(s.b2(s.c2(x))))
        x=F.relu(s.b3(s.c3(x)));x=F.relu(s.b4(s.c4(x)))
        return x.transpose(1,2)  # (B,62,64) frames
    def classify(s,frames):
        mixed=s.core(frames);return s.fc(mixed.mean(1))  # (B,nw)
    def forward(s,w):
        return s.classify(s.extract_frames(w))


@torch.no_grad()
def predict_with_autocorrection(model, wav, tau=0.9, max_lsra=4):
    """Observer → si incertain, LSRA (+itérations SCB) → correction.
    Retourne (pred, conf, n_steps, was_corrected)."""
    frames = model.extract_frames(wav)
    # Step 1: prédiction initiale (1 passage SCB)
    logits = model.classify(frames)
    probs = F.softmax(logits, -1)
    conf, pred = probs.max(1)
    n_steps = 1; was_corrected = False

    # Step 2: OBSERVER — si incertain, raisonner PLUS (LSRA depth)
    for k in range(max_lsra):
        uncertain = conf < tau
        if not uncertain.any():
            break  # toutes confiantes → fini
        # Agent correction: itérer le SCB sur les frames incertaines (+1 raisonnement)
        frames = model.core(frames)   # LSRA: v(t+1) = SCB(v(t)) — deeper reasoning
        logits_new = model.classify(frames)
        probs_new = F.softmax(logits_new, -1)
        conf_new, pred_new = probs_new.max(1)
        # Mettre à jour seulement les incertaines
        updated = uncertain.squeeze(-1) if uncertain.dim()>1 else uncertain
        pred[updated] = pred_new[updated]
        conf[updated] = conf_new[updated]
        n_steps += 1
        was_corrected = True

    return pred, conf, n_steps, was_corrected


def load_wav(p,T=16000):
    y,sr=sf.read(p);y=y.astype(np.float32)
    if y.ndim>1:y=y.mean(1)
    return torch.from_numpy(np.pad(y,(0,max(0,T-len(y))))[:T] if len(y)<T else y[:T])

def load_data(words):
    ts=set(l.strip() for l in open(os.path.join(SC,"testing_list.txt")) if l.strip());te={}
    for wi,w in enumerate(words):
        a=sorted(glob.glob(os.path.join(SC,w,"*.wav")))
        tp=[load_wav(p) for p in a if os.path.relpath(p,SC)in ts][:50]
        if len(tp)>=5:te[wi]=torch.stack(tp)
    return te


def evaluate_autocorrection(ckpt_path, tau=0.9, max_lsra=4):
    """Évalue le modèle AVEC auto-correction LSRA vs SANS (baseline)."""
    ws=sorted([w for w in os.listdir(SC) if os.path.isdir(os.path.join(SC,w)) and not w.startswith("_")])
    NW=len(ws); te=load_data(ws)
    model=M5Unified(NW).to(device)
    ck=torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ck["model_state"]); model.eval()
    base_best=ck.get("best",0)

    # Baseline (sans auto-correction)
    ok_base=tot=0
    # Avec auto-correction LSRA
    ok_ac=0; n_corrected=0; n_corrected_right=0; n_steps_avg=[]

    for wi in te:
        for j in range(0,len(te[wi]),32):
            wavs=te[wi][j:j+32].to(device)
            # baseline
            logits=model(wavs); pred_base=logits.argmax(1)
            ok_base+=(pred_base.cpu()==wi).sum().item(); tot+=len(pred_base)
            # auto-correction
            pred_ac,conf,steps,corrected=predict_with_autocorrection(model,wavs,tau,max_lsra)
            ok_ac+=(pred_ac.cpu()==wi).sum().item()
            n_steps_avg.append(steps)
            if corrected:
                # compter les corrections (changement de prédiction)
                changed=(pred_ac.cpu()!=pred_base.cpu())
                n_corrected+=changed.sum().item()
                n_corrected_right+=((pred_ac.cpu()==wi)&changed).sum().item()

    acc_base=ok_base/max(tot,1); acc_ac=ok_ac/max(tot,1)
    avg_steps=np.mean(n_steps_avg) if n_steps_avg else 0
    corr_rate=n_corrected/max(tot,1)
    corr_precision=n_corrected_right/max(n_corrected,1)

    print(f"\n{'='*60}")
    print(f"AUTO-CORRECTION LSRA — observer → +SCB itérations → correction")
    print(f"{'='*60}")
    print(f"  Modèle de base (sans LSRA):     {acc_base*100:.1f}%")
    print(f"  AVEC auto-correction LSRA:       {acc_ac*100:.1f}%")
    print(f"  Δ auto-correction:               {(acc_ac-acc_base)*100:+.1f}pt")
    print(f"  Steps moyen LSRA:                {avg_steps:.1f}")
    print(f"  Échantillons corrigés:           {n_corrected}/{tot} ({corr_rate*100:.1f}%)")
    print(f"  Précision des corrections:       {corr_precision*100:.1f}% (corrigés qui étaient justes)")
    print(f"  τ (seuil observer):              {tau}")
    return acc_base, acc_ac


if __name__=="__main__":
    ckpt="/media/akone/SAVENVME2/Datasets/ocm26400/audio_unified_trained.pt"
    if not os.path.exists(ckpt):
        ckpt="/media/akone/SAVENVME2/Datasets/ocm26400/audio_m5_trained.pt"
    print(f"[AUTO-CORRECTION LSRA] modèle: {ckpt}")
    base, ac = evaluate_autocorrection(ckpt, tau=0.9, max_lsra=4)
    json.dump({"baseline_acc":base,"autocorrect_acc":ac,"delta":ac-base},
              open("ocm26400/audio_lsra_autocorrect_results.json","w"),indent=2)
