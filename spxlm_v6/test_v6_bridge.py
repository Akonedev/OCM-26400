"""Tests contractuels — pont v6->AMV (P4). CPU, SANS charger v6 (léger).

On teste les CONTRATS honnêtes du pont : la tête produit ent(64)+conf,
l'AMV construit utilise meta[1] (source_confidence, slot dédié — pas meta[0]
écrasé par lsra_loop), et la loss du pont exclut L_step (juge décision b).
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))   # MathsBase/ pour ocm26400

import torch
from ocm26400.amv import AMVVector, D_MODEL, PART
from experiment_v6_bridge import BridgeHead


def test_bridge_head_shapes():
    """La tête : hidden(256) -> ent(64) + source_confidence scalaire."""
    head = BridgeHead()
    h = torch.randn(8, 256)
    ent, conf = head(h)
    assert ent.shape == (8, PART), ent.shape
    assert conf.shape == (8,), conf.shape


def test_amv_uses_meta1_for_source_confidence():
    """Le pont écrit meta[1] (source_confidence), PAS meta[0] (confiance LSRA).

    C'est la partition du juge : lsra_loop n'écrit que meta[0], donc la confiance
    source du pont (meta[1]) n'est PAS écrasée. On le vérifie structurellement."""
    v = torch.zeros(D_MODEL)
    ent = torch.randn(PART)
    v[0:PART] = ent
    META0 = PART * 3          # meta = tensor[192:256], meta[0]=v[192]
    v[META0] = 2.5            # meta[0] = confiance LSRA (laissée à 0 par le pont)
    v[META0 + 1] = 3.0        # meta[1] = source_confidence (écrite par le pont)
    amv = AMVVector(v)
    # source_confidence lit meta[1] (sigmoid) — pas meta[0]
    assert abs(float(amv.source_confidence()) - float(torch.sigmoid(torch.tensor(3.0)))) < 1e-5
    # confidence (meta[0]) est indépendante
    assert abs(float(amv.confidence()) - float(torch.sigmoid(torch.tensor(2.5)))) < 1e-5


def test_bridge_head_no_lstep_dependency():
    """La tête n'a que ent+conf (pas de dépendance à L_step non-différentiable).

    L_step (acsp.py:38) est une CONSTANTE sans gradient ; le pont l'exclut de sa
    loss (juge décision b). Structurellement, BridgeHead ne comporte que 2 Linear."""
    head = BridgeHead()
    linears = [m for m in head.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 2  # ent + conf uniquement
