#!/usr/bin/env python3
"""Évaluation ENSEMBLE 3-modèles (agents systems) → vote → SOTA push.

Charge les 3 modèles (seeds 0/1/2, M5→SCB(62)→CE), average leurs softmax sur le test set,
argmax → prédiction ensemble. L'ensemble donne typiquement +1-2pt sur le meilleur single model.
"""
import torch,torch.nn.functional as F,glob,os,numpy as np,json
import soundfile as sf
from ocm26400.audio_unified_m5scb import M5Unified, load_wav, load_data

device="cuda"
SC="/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
CKPTS={
    0:"/media/akone/SAVENVME2/Datasets/ocm26400/audio_unified_trained.pt",
    1:"/media/akone/SAVENVME2/Datasets/ocm26400/audio_unified_seed1.pt",
    2:"/media/akone/SAVENVME2/Datasets/ocm26400/audio_unified_seed2.pt",
}

@torch.no_grad()
def main():
    ws=sorted([w for w in os.listdir(SC) if os.path.isdir(os.path.join(SC,w)) and not w.startswith("_")])
    NW=len(ws); tr,te=load_data(ws)
    # charge les modèles disponibles
    models={}
    for seed,ckpt in CKPTS.items():
        if os.path.exists(ckpt):
            m=M5Unified(NW).to(device)
            ck=torch.load(ckpt,map_location=device,weights_only=True)
            m.load_state_dict(ck["model_state"]);m.eval()
            models[seed]=m
            print(f"  seed{seed}: loaded (best was {ck.get('best',0)*100:.1f}%)",flush=True)
    if len(models)<2:
        print(f"Pas assez de modèles ({len(models)}). Attendre seed 2.");return
    print(f"\n[ENSEMBLE] {len(models)} modèles -> average softmax -> vote",flush=True)
    ok_ens=tot=0; ok_indiv={s:0 for s in models}
    for wi in te:
        for j in range(0,len(te[wi]),64):
            wavs=te[wi][j:j+64].to(device)
            probs_sum=torch.zeros(min(64,len(te[wi])-j),NW,device=device)
            for s,m in models.items():
                logits=m(wavs);probs=F.softmax(logits,-1)
                probs_sum+=probs
                ok_indiv[s]+=(logits.argmax(1).cpu()==wi).sum().item()
            pred_ens=probs_sum.argmax(1).cpu()
            ok_ens+=(pred_ens==wi).sum().item();tot+=len(pred_ens)
    acc_ens=ok_ens/max(tot,1)
    print(f"\n{'='*60}")
    print(f"ENSEMBLE {len(models)}-modèles — test officiel SpeechCommands")
    print(f"{'='*60}")
    for s in models:
        print(f"  seed{s} individuel: {ok_indiv[s]/max(tot,1)*100:.1f}%")
    print(f"  ENSEMBLE (avg softmax): {acc_ens*100:.1f}%")
    best_single=max(ok_indiv.values())/max(tot,1)
    print(f"  Δ ensemble vs best single: {(acc_ens-best_single)*100:+.1f}pt")
    print(f"  SOTA SpeechCommands: ~96%")
    print(f"  Δ vs SOTA: {acc_ens*100-96:+.1f}pt")
    json.dump({"ensemble_acc":acc_ens,"n_models":len(models),"delta_vs_best_single":acc_ens-best_single,
               "delta_vs_sota_96":acc_ens*100-96,"individual":{str(s):ok_indiv[s]/max(tot,1) for s in models}},
              open("ocm26400/audio_ensemble_results.json","w"),indent=2)
    print("  [sauvé] ocm26400/audio_ensemble_results.json")

if __name__=="__main__":
    main()
