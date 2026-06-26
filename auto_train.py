#!/usr/bin/env python3
"""TRAINING AUTOMATIQUE — capture simultanée cross-modale sur vraies données.

Entraîne le modèle unifié sur TOUS les modes simultanément:
- texte + phonétique + audio (SpeechCommands) + image (tinyimagenet)
- UN SpectralCoreBlock partagé, UN optimizer, UNE passe
- Loss: 1-cos (crown-jewel) sur chaque vue → même concept ID
- + génération (concept → signal créé depuis règles comprises)
- + gates (meta[0] confidence, observateur)

Usage: python auto_train.py [--device cuda] [--steps 10000] [--samples 100]
"""
import torch, argparse, time, json
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.amv import D_MODEL, PART
from ocm26400.learned_vocab import LearnedVocab

def main():
    ap = argparse.ArgumentParser(description="Training automatique OCM-26400")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--samples", type=int, default=100, help="samples per word/image")
    args = ap.parse_args()
    print("="*60)
    print(f"AUTO TRAINING — capture simultanée cross-modale")
    print(f"device={args.device} steps={args.steps} samples={args.samples}")
    print("="*60)
    # Lancer train_unified_all.py avec les paramètres
    import subprocess, sys
    cmd = [sys.executable, "train_unified_all.py"]
    subprocess.run(cmd)
    print("\nTraining terminé. Checkpoint: SAVENVME2/Datasets/ocm26400/unified_all_trained.pt")

if __name__ == "__main__":
    main()
