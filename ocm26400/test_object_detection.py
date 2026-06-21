"""Tests détection d'objets (OCM-26400)."""
import numpy as np
from ocm26400.object_detection import make_grid, detect_grid, evaluate_detection


def test_make_grid_shape():
    imgs = [np.zeros(784) for _ in range(9)]
    labels = [i for i in range(9)]
    grid, gt = make_grid(imgs, labels, grid_size=3)
    assert grid.shape == (84, 84)     # 3×28
    assert len(gt) == 9


def test_make_grid_gt():
    imgs = [np.ones(784) * i * 10 for i in range(4)]
    labels = [0, 1, 2, 3]
    _, gt = make_grid(imgs, labels, grid_size=2)
    assert (0, 0, 0) in gt and (0, 1, 1) in gt


def test_detect_grid_returns_detections():
    from ocm26400.ocr import train_ocr
    model, _ = train_ocr(n_train=500, n_steps=50)
    imgs = [np.random.randn(784) for _ in range(4)]
    grid, _ = make_grid(imgs, [0, 1, 2, 3], grid_size=2)
    dets = detect_grid(model, grid, grid_size=2)
    assert len(dets) > 0     # détecte quelque chose


def test_evaluate_detection_runs():
    from ocm26400.ocr import train_ocr
    model, _ = train_ocr(n_train=500, n_steps=50)
    res = evaluate_detection(model, n_grids=3, grid_size=2)
    assert res["digit_accuracy"] >= 0
    assert res["recall"] > 0
