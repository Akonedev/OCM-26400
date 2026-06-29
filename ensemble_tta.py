#!/usr/bin/env python3
"""Ensemble + TTA (Test-Time Augmentation) → pousser vers/dépasser le SOTA 96%.

TTA : pour chaque sample de test, créer K versions speed-perturbées, faire prédire
l'ensemble (3 modèles × K versions), moyenner les softmax → diversité d'inférence.
Typiquement +0.5-1pt sur l'ensemble seul.
"""
import torch,torch.nn.functional as F,glob,os,numpy as np,json
import soundfile as sf
from ocm26400.audio_unified_m5scb import M5Unified, load_wav, load_data, spd

device="cuda"
SC="/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
CKPTS={
    0:"/media/akone/SAVENVME2/Datasets/ocm26400/audio_unified_trained.pt",
    1:"/media/akone/SAVENVME2/Datasets/ocm26400/audio_unified_seed1.pt",
    2:"/media/akone/SAVENVME2/Datasets/ocm26400/audio_unified_seed2.pt",
}
TTA_RATES = [1.0, 0.95, 1.05]  # original + 2 speed-perturbed

@torch.no_grad()
def main():
    ws=sorted([w for w in os.listdir(SC) if os.path.isdir(os.path.join(SC,w)) and not w.startswith("_")])
    NW=len(ws); tr,te=load_data(ws)
    models={}
    for seed,ckpt in CKPTS.items():
        if os.path.exists(ckpt):
            m=M5Unified(NW).to(device)
            ck=torch.load(ckpt,map_location=device,weights_only=True)
            m.load_state_dict(ck["model_state"]);m.eval();models[seed]=m
    print(f"[ENSEMBLE+TTA] {len(models)} modèles × {len(TTA_RATES)} TTA = {len(models)*len(TTA_RATES)} prédictions/sample",flush=True)
    ok_ens=tot=0; ok_tta=0
    for wi in te:
        for j in range(0,len(te[wi]),32):
            batch=te[wi][j:j+32]; B=len(batch)
            # ensemble simple (1 TTA rate = 1.0)
            probs_ens=torch.zeros(B,NW,device=device)
            # ensemble + TTA (tous les rates)
            probs_tta=torch.zeros(B,NW,device=device)
            for rate in TTA_RATES:
                if rate==1.0:
                    wavs=batch.to(device)
                else:
                    wavs=torch.stack([torch.from_numpy(spd(w.numpy(),(rate-0.05,rate+0.05))) for w in batch]).to(device)
                for s,m in models.items():
                    probs=F.softmax(m(wavs),-1)
                    probs_tta+=probs
                    if rate==1.0: probs_ens+=probs
            ok_ens+=(probs_ens.argmax(1).cpu()==wi).sum().item()
            ok_tta+=(probs_tta.argmax(1).cpu()==wi).sum().item()
            tot+=B
    acc_ens=ok_ens/max(tot,1); acc_tta=ok_tta/max(tot,1)
    print(f"\n{'='*60}")
    print(f"ENSEMBLE vs ENSEMBLE+TTA — test officiel")
    print(f"{'='*60}")
    print(f"  Ensemble simple (3 modèles):       {acc_ens*100:.1f}%")
    print(f"  Ensemble + TTA ({len(TTA_RATES)} rates):       {acc_tta*100:.1f}%")
    print(f"  Δ TTA:                              {(acc_tta-acc_ens)*100:+.1f}pt")
    print(f"  SOTA SpeechCommands:                ~96%")
    print(f"  Δ vs SOTA:                          {acc_tta*100-96:+.1f}pt")
    json.dump({"ensemble_acc":acc_ens,"ensemble_tta_acc":acc_tta,"delta_tta":acc_tta-acc_ens,
               "delta_vs_sota_96":acc_tta*100-96,"tta_rates":TTA_RATES,"n_models":len(models)},
              open("ocm26400/audio_ensemble_tta_results.json","w"),indent=2)
    print("  [sauvé] ocm26400/audio_ensemble_tta_results.json")

if __name__=="__main__":
    main()
