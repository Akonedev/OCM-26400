#!/usr/bin/env python3
"""
EXPÉRIENCE vision RÉELLE (OCM-26400, spec multimodal image sur VRAIE donnée).

Contrairement aux encodeurs validés sur signaux synthétiques, ICI l'ImageEncoder est
ENTRAÎNÉ et ÉVALUÉ sur de VRAIES images étiquetées : 1797 chiffres manuscrits réels
(sklearn load_digits, 8x8 grayscale, 10 classes). On mesure l'accuracy de classification
RÉELLE sur un split train/test → preuve que l'encodeur image fonctionne sur de vraies
données, pas seulement sur du bruit synthétique.

C'est la modalité image VALIDÉE SUR DONNÉE RÉELLE (spec 'multimodalité réelle image').
"""
import json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from ocm26400.multimodal_encoders import ImageEncoder
from ocm26400.amv import PART


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    digits = load_digits()
    X = digits.images.astype("float32") / 16.0      # (1797, 8, 8), pixels réels normalisés
    y = digits.target.astype("int64")
    X = X[:, None, :, :]                            # (1797, 1, 8, 8) channel=1
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    Xtr = torch.tensor(Xtr); Xte = torch.tensor(Xte)
    ytr = torch.tensor(ytr); yte = torch.tensor(yte)
    print(f"OCM-26400 VISION RÉELLE | device={device} | digits manuscrits réels "
          f"(sklearn) | train {len(Xtr)} / test {len(Xte)} | 10 classes (chance 10%)")

    enc = ImageEncoder(out_dim=PART, patch=4).to(device)     # vraies images 8x8
    clf = nn.Linear(PART, 10).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(clf.parameters()), lr=3e-3)

    t0 = time.time()
    n = len(Xtr)
    for step in range(400):
        idx = torch.randint(0, n, (128,))
        emb = enc(Xtr[idx].to(device))
        loss = F.cross_entropy(clf(emb), ytr[idx].to(device))
        opt.zero_grad(); loss.backward(); opt.step()

    enc.eval(); clf.eval()
    with torch.no_grad():
        # test par batches (397 images)
        preds = []
        for i in range(0, len(Xte), 256):
            emb = enc(Xte[i:i+256].to(device))
            preds.append(clf(emb).argmax(-1).cpu())
        pred = torch.cat(preds)
        acc = (pred == yte).float().mean().item()
    dt = time.time() - t0
    print(f"\nAccuracy classification RÉELLE (chiffres manuscrits, test) : {acc*100:.1f}%")
    verdict = "VALIDÉ" if acc > 0.85 else ("PARTIEL" if acc > 0.5 else "NON VALIDÉ")
    print(f"VERDIT (ImageEncoder sur VRAIES images étiquetées) : {verdict}")

    results = {
        "task": "vision RÉELLE : ImageEncoder sur chiffres manuscrits (sklearn digits)",
        "dataset": "load_digits (1797 vraies images 8x8, 10 classes, données réelles)",
        "train": len(Xtr), "test": len(Xte),
        "test_accuracy": round(acc, 4), "chance": 0.1,
        "note": "encodeur image ENTRAÎNÉ+ÉVALUÉ sur vraies images (plus synthétique)",
        "verdict": verdict, "duration_s": round(dt, 1),
    }
    with open("ocm26400/real_vision_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nRésultats: ocm26400/real_vision_results.json")
    return results


if __name__ == "__main__":
    main()
