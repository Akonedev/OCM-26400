"""OmniModel MULTI-RÈGLES conjoint + génération INTER-RÈGLES (OCM-26400, paradigme complet).

Réfute la critique DA-2 du panel : le capstone ne grokkait qu'UNE opération (« génère
n'importe quoi » = survente). ICI, UN SEUL noyau (ReasonerBlock = OmniModel.core) est
entraîné CONJOINTEMENT sur PLUSIEURS RÈGLES hétérogènes (add linéaire, mul BILINÉAIRE,
linop) via dispatch op_id, puis GÉNÈRE des compositions INTER-RÈGLES sur chaînes mixtes
jamais vues (add puis mul puis linop...).

C'est le paradigme spec COMPLET : comprendre TOUTES les règles (1 noyau, N règles) ->
généraliser -> GÉNÉRER n'importe quelle composition inter-règles.

* RULES         : add(a,b)=(a+b)%n, mul(a,b)=(a*b)%n (bilinéaire, vraiment différent),
                   linop(a,b)=(3a+5b)%n.
* train_omni_rules : entraîne le noyau op-aware (slot op = règle) sur les 3 règles.
* comprehend   : verify — le noyau a-t-il compris chaque règle (accuracy sur la règle) ?
* generate_chain : GÉNÉRATION inter-règles — applique une chaîne mixte de règles sur des
                   entrées neuves, compare à la vérité (composition des fonctions règles).
"""
from __future__ import annotations
from typing import List, Tuple, Dict
import random
import torch

from .amv import D_MODEL
from .verifier import SymbolicDict, P_MOD
from .reasoner import ReasonerBlock, encode_input, DEVICE

N = P_MOD
RULES: Dict[str, callable] = {
    "add":  lambda a, b: (a + b) % N,
    "mul":  lambda a, b: (a * b) % N,          # bilinéaire (vraiment différent d'add)
    "linop": lambda a, b: (3 * a + 5 * b) % N,
}
RULE_NAMES = list(RULES.keys())


def encode_rule(a: int, b: int, op_id: int, d: SymbolicDict) -> torch.Tensor:
    """AMV : ent=a, prop=b, op=règle (one-hot op_id dans le slot op)."""
    v = torch.zeros(D_MODEL)
    v[0:64] = d.canonical(a)
    v[64:128] = d.canonical(b)
    v[128:192] = d.canonical(op_id)            # slot op = identifiant de règle
    return v


def train_omni_rules(d: SymbolicDict, n_steps: int = 2000, lr: float = 3e-3,
                     batch: int = 128) -> ReasonerBlock:
    """Entraîne UN noyau op-aware sur TOUTES les règles (perte jointe)."""
    torch.manual_seed(0)
    blk = ReasonerBlock().to(DEVICE)
    opt = torch.optim.Adam(blk.parameters(), lr=lr)
    for _ in range(n_steps):
        k = torch.randint(0, len(RULE_NAMES), (batch,))     # règle tirée
        a = torch.randint(0, N, (batch,)); b = torch.randint(0, N, (batch,))
        loss = torch.tensor(0.0, device=DEVICE)
        for i in range(batch):
            name = RULE_NAMES[int(k[i])]; ai, bi = int(a[i]), int(b[i])
            x = encode_rule(ai, bi, int(k[i]), d).unsqueeze(0).to(DEVICE)
            out = blk(x)[0]
            ent = out[0:64]
            tgt = d.canonical(RULES[name](ai, bi)).to(DEVICE)
            cos = (ent @ tgt) / (ent.norm() * tgt.norm() + 1e-8)
            loss = loss + (1.0 - cos)
        loss = loss / batch
        opt.zero_grad(); loss.backward(); opt.step()
    return blk


@torch.no_grad()
def comprehend(blk: ReasonerBlock, d: SymbolicDict, n_test: int = 60) -> Dict[str, float]:
    """COMPRÉHENSION : accuracy par règle (le noyau a-t-il compris chaque règle ?)."""
    blk.eval()
    out = {}
    for op_id, name in enumerate(RULE_NAMES):
        ok = 0
        for _ in range(n_test):
            a, b = random.randrange(N), random.randrange(N)
            x = encode_rule(a, b, op_id, d).unsqueeze(0).to(DEVICE)
            r_pred, _ = d.decode(blk(x)[0][0:64])
            ok += (r_pred == RULES[name](a, b))
        out[name] = ok / n_test
    return out


def inter_rule_gt(chain: List[Tuple[str, int]], init: int) -> int:
    """Vérité : compose les fonctions règles. chain = [(rule_name, operand)], init=a0.
    r = rule(a0, op0) puis rule(r, op1) ..."""
    cur = init
    for name, op in chain:
        cur = RULES[name](cur, op)
    return cur


@torch.no_grad()
def generate_chain(blk: ReasonerBlock, d: SymbolicDict,
                   chain: List[Tuple[str, int]], init: int) -> int:
    """GÉNÉRATION inter-règles : le noyau applique la chaîne mixte de règles (op_id par étape)."""
    blk.eval()
    cur = init
    for name, op in chain:
        op_id = RULE_NAMES.index(name)
        x = encode_rule(cur, op, op_id, d).unsqueeze(0).to(DEVICE)
        cur, _ = d.decode(blk(x)[0][0:64])
    return cur
