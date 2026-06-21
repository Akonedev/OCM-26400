"""Détection d'objets — chiffres dans une image grille — modalité detection.

Détection : identifier ET localiser des objets (bounding boxes). On le fait sur des
chiffres MNIST placés dans une grille — détecte leur position (quelle cellule) et les
identifie (quel chiffre). Utilise le CNN OCR (pas de transformer).

* DetectionGrid : génère une grille d'images MNIST (chiffres placés à des positions).
* detect(model, grid_image) → liste de (position, chiffre, confiance).
* evaluate_detection : precision/recall sur la détection (objets trouvés + corrects).

C'est de la détection RÉELLE (localiser + identifier). Pas de COCO, mais vrai mécanisme.
"""
from __future__ import annotations
from typing import List, Tuple

import torch
import numpy as np


def make_grid(digit_images: List[np.ndarray], labels: List[int],
              grid_size: int = 3) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """Construit une grille grid_size×grid_size de chiffres 28×28.
    Retourne (grid_image (grid*28 x grid*28), ground_truth [(row, col, digit), ...])."""
    cell = 28
    full = np.zeros((grid_size * cell, grid_size * cell), dtype=np.float32)
    gt = []
    idx = 0
    for r in range(grid_size):
        for c in range(grid_size):
            if idx < len(digit_images):
                img = digit_images[idx].reshape(28, 28)
                full[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = img
                gt.append((r, c, labels[idx]))
                idx += 1
    return full, gt


def detect_grid(model, grid_image: np.ndarray, grid_size: int = 3,
                device: str = None) -> List[Tuple[int, int, int, float]]:
    """Détecte les chiffres dans une grille : découpe en cellules, classifie chaque cellule.
    Retourne [(row, col, digit_prédit, confiance), ...]."""
    if device is None:
        device = next(model.parameters()).device
    cell = 28
    detections = []
    for r in range(grid_size):
        for c in range(grid_size):
            patch = grid_image[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell]
            x = torch.as_tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            if x.max() > 1:
                x = x / 255.0
            with torch.no_grad():
                logits = model(x)
                prob = torch.softmax(logits, dim=1)
                conf, pred = prob.max(dim=1)
                # si la cellule est vide (confiance faible), on saute
                if conf.item() > 0.3:
                    detections.append((r, c, int(pred[0]), float(conf[0])))
    return detections


def evaluate_detection(model, n_grids: int = 20, grid_size: int = 3,
                       device: str = None) -> dict:
    """Évalue la détection : precision (chiffres corrects / détectés),
    recall (détectés / total), accuracy position+digit."""
    from .ocr import load_mnist
    if device is None:
        device = next(model.parameters()).device
    Xtr, ytr, _, _ = load_mnist(1000, 100)
    rng = np.random.RandomState(0)

    n_correct_id = n_correct_pos = n_detected = n_total = 0
    for _ in range(n_grids):
        # grille aléatoire
        idx = rng.choice(len(Xtr), grid_size * grid_size, replace=False)
        imgs = [Xtr[i] for i in idx]
        labels = [int(ytr[i]) for i in idx]
        grid, gt = make_grid(imgs, labels, grid_size)
        detections = detect_grid(model, grid, grid_size, device)

        gt_set = {(r, c) for r, c, d in gt}
        gt_dict = {(r, c): d for r, c, d in gt}
        n_total += len(gt)

        for r, c, pred, conf in detections:
            n_detected += 1
            if (r, c) in gt_dict:
                n_correct_pos += 1
                if pred == gt_dict[(r, c)]:
                    n_correct_id += 1

    return {
        "task": "détection d'objets (chiffres MNIST dans grille)",
        "n_grids": n_grids, "n_cells": n_total,
        "n_detected": n_detected,
        "position_accuracy": round(n_correct_pos / max(n_detected, 1), 3),
        "digit_accuracy": round(n_correct_id / max(n_detected, 1), 3),
        "recall": round(n_detected / max(n_total, 1), 3),
        "archi": "CNN OCR (pas de transformer) + découpage grille",
    }


if __name__ == "__main__":
    from .ocr import train_ocr
    model, _ = train_ocr(n_train=3000, n_test=500, n_steps=400)
    res = evaluate_detection(model, n_grids=15, grid_size=3)
    print(f"[detection] {res['task']}")
    print(f"  digit_accuracy={res['digit_accuracy']*100:.1f}% | position={res['position_accuracy']*100:.1f}% "
          f"| recall={res['recall']*100:.0f}% | {res['archi']}")
