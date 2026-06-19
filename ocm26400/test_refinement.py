"""Tests TDD — P3 gate calibrée + abstention (OCM-26400).

Honnête : on ne teste PAS un crown-jewel « TTC achète accuracy » (tautologie,
enterré par le DA). On teste la MÉCANIQUE honnête : OOD détecté, gate stop/anomalie
de lsra_loop, et calibration valid-vs-OOD après entraînement.
"""
import torch
import torch.nn as nn

from ocm26400.verifier import SymbolicDict, Verifier, P_MOD
from ocm26400.reasoner import encode_input, lsra_loop, TAU_GROK, DEVICE
from ocm26400.experiment_refinement import make_ood_input, train_calibrated_block, confidence


# --- OOD : l'entrée garbage n'est pas un symbole décodable ---

def test_ood_input_decodes_invalid():
    """Une entrée OOD (ent = bruit) n'est PAS un one-hot valide → decode valid=False."""
    d = SymbolicDict(n=P_MOD)
    x = make_ood_input(d, P_MOD, device="cpu")
    _, valid = d.decode(x[0:64])
    assert valid is False, "le bruit gaussien ne doit pas décoder comme un symbole valide"


# --- mécanique de la gate lsra_loop : stop vs ANOMALIE (mocks déterministes) ---

class _MockBlock(nn.Module):
    """Block factice qui force meta[0] à une valeur constante (teste la gate isolément)."""
    def __init__(self, meta_val):
        super().__init__()
        self._dummy = nn.Linear(1, 1).to(DEVICE)   # pour next(parameters).device
        self.meta_val = meta_val

    def forward(self, x):
        out = x.clone()
        out[..., 192] = self.meta_val
        return out


def test_lsra_loop_confident_when_meta_high():
    """meta[0] haut (sigmoid>tau) → lsra_loop stop au pas 1, confident=True."""
    d = SymbolicDict(n=P_MOD)
    blk = _MockBlock(meta_val=5.0)                  # sigmoid(5)=0.993 > 0.9
    x0 = encode_input(2, 3, d)
    idx, steps, conf = lsra_loop(blk, d, x0, max_iter=8, tau=TAU_GROK)
    assert conf is True
    assert steps == 1, f"devrait stopper au pas 1, pas {steps}"


def test_lsra_loop_anomaly_when_meta_low():
    """meta[0] bas (sigmoid<tau) → jamais confiant → ANOMALIE (confident=False)."""
    d = SymbolicDict(n=P_MOD)
    blk = _MockBlock(meta_val=-5.0)                 # sigmoid(-5)=0.007 < 0.9
    x0 = encode_input(2, 3, d)
    idx, steps, conf = lsra_loop(blk, d, x0, max_iter=8, tau=TAU_GROK)
    assert conf is False, "meta basse doit déclencher l'ANOMALIE (jamais confiant)"
    assert steps == 8


# --- calibration : le block apprend valid=haut / OOD=bas (intégration, mini-training) ---

def test_calibrated_block_separates_valid_from_ood():
    """Après entraînement calibré, confidence(valid) >> confidence(OOD)."""
    d = SymbolicDict(n=P_MOD)
    ver = Verifier(d)
    blk = train_calibrated_block(d, ver, n_steps=400)   # mini-training (~2s)
    import random as _r
    _r.seed(1)
    valid_x = [encode_input(_r.randrange(P_MOD), _r.randrange(P_MOD), d) for _ in range(60)]
    ood_x = [make_ood_input(d, P_MOD) for _ in range(60)]
    cv = sum(confidence(blk, x) for x in valid_x) / len(valid_x)
    co = sum(confidence(blk, x) for x in ood_x) / len(ood_x)
    assert cv > co + 0.3, f"calibration insuffisante: valid={cv:.3f} ood={co:.3f}"
    assert cv > 0.6, f"confidence valide trop basse: {cv:.3f}"
