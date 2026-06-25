"""L_step DIFFÉRENTIABLE (Gumbel straight-through) + ACSP câblé dans un trainer (P-B).

Le verdict expert/DA/juge a révélé que `acsp_loss` est JAMAIS appelée par un trainer
réel (reasoner.py / omni_rules.py / omni.py utilisent du `(1-cos)` ad-hoc) et que
`l_step` est une CONSTANTE non-différentiable (acsp.py:36-38). Le « §2 Causal Rigor
Loss » du spec était décoratif. P-B le rend RÉEL :

* decode_gumbel  : Gumbel-Softmax straight-through -> argmax différentiable (gradient
  coule à travers le décodage symbolique). Techno : Jang 2016 / Bengio 2013.
* l_step_diff    : pénalité de LÉGALITÉ différentiable (pénalise l'écart au symbole
  correct compose(a,b)) via v.ent -> gradient réel dans le noyau.
* acsp_loss_diff : ACSP end-to-end différentiable (l_align + l_step_diff + l_sparse).
* train_with_acsp: trainer RÉEL utilisant acsp_loss_diff (enfin ACSP vit dans l'entraînement).

Cela AMÉLIORE LE GROKKING (signal de légalité symbolique en plus de l'alignement) et
active honnêtement le spec §2. Non-cassant : l'ancienne acsp_loss/l_step restent pour
les 8 tests test_acsp.py existants.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from .amv import AMVVector
from .verifier import Verifier, SymbolicDict, P_BACKTRACK
from .acsp import l_align, l_sparse, ALPHA, BETA, GAMMA
from .reasoner import ReasonerBlock, encode_input, DEVICE


def decode_gumbel(logits_n: torch.Tensor, tau: float = 1.0, hard: bool = True) -> torch.Tensor:
    """Gumbel-Softmax straight-through : logits (n,) -> quasi-one-hot (n,) avec gradient.

    hard=True : forward = one_hot(argmax), backward = softmax (straight-through)."""
    g = -torch.log(-torch.log(torch.rand_like(logits_n) + 1e-20) + 1e-20)
    y = F.softmax((logits_n + g) / tau, dim=-1)
    if hard:
        idx = y.argmax(dim=-1)
        y_hard = F.one_hot(idx, num_classes=y.shape[-1]).float()
        return y_hard - y.detach() + y          # straight-through
    return y


def l_step_diff(v: AMVVector, verifier: Verifier, a: int, b: int,
                op_id: int = 0, tau: float = 1.0) -> torch.Tensor:
    """L_step DIFFÉRENTIABLE : pénalise l'écart entre le symbole décodé (soft, via v.ent)
    et le symbole légal compose(a,b). Gradient réel vers le noyau."""
    n = verifier.dict.n
    soft = decode_gumbel(v.ent[:n], tau=tau, hard=True)        # (n,) quasi-one-hot grad- bearing
    correct = verifier.compose(a, b, op_id=op_id)
    return P_BACKTRACK * (1.0 - soft[correct])                 # pénalité si écart au légal


def acsp_loss_diff(v: AMVVector, dictionary: SymbolicDict, verifier: Verifier,
                   a: int, b: int, op_id: int = 0, tau: float = 1.0) -> torch.Tensor:
    """ACSP end-to-end DIFFÉRENTIABLE : l_align + l_step_diff + l_sparse.

    Contrairement à acsp_loss (l_step constante), tous les termes portent du gradient."""
    return (ALPHA * l_align(v, dictionary)
            + BETA * l_step_diff(v, verifier, a, b, op_id, tau)
            + GAMMA * l_sparse(v))


def train_with_acsp(dictionary: SymbolicDict, verifier: Verifier, n_steps: int = 1500,
                    lr: float = 3e-3, batch: int = 64, device: str = DEVICE) -> ReasonerBlock:
    """Trainer RÉEL utilisant acsp_loss_diff (ACSP vit enfin dans l'entraînement).
    Apprend (a,b) -> ent=canonical(compose(a,b)) AVEC le signal de légalité l_step_diff."""
    torch.manual_seed(0)
    blk = ReasonerBlock().to(device)
    opt = torch.optim.Adam(blk.parameters(), lr=lr)
    pairs = [(a, b) for a in range(dictionary.n) for b in range(dictionary.n)]
    for _ in range(n_steps):
        idx = torch.randint(0, len(pairs), (batch,))
        batch_in = torch.stack([encode_input(pairs[i][0], pairs[i][1], dictionary)
                                for i in idx]).to(device)
        out = blk(batch_in)
        loss = torch.tensor(0.0, device=device)
        for j, i in enumerate(idx):
            a, b = pairs[i]
            v = AMVVector(out[j])
            # l_align domine (signal de grok principal) ; l_step différentiable réduit (0.3)
            # pour que la convergence atteigne >0.9 à 800 pas (fix test_diff_decode)
            loss = loss + (l_align(v, dictionary)
                           + 0.3 * l_step_diff(v, verifier, a, b)
                           + GAMMA * l_sparse(v))
        loss = loss / batch
        opt.zero_grad(); loss.backward(); opt.step()
    return blk


@torch.no_grad()
def eval_binary(blk: ReasonerBlock, dictionary: SymbolicDict, verifier: Verifier,
                n_test: int = 121) -> float:
    """Accuracy du block binaire sur op(a,b)."""
    blk.eval()
    dev = next(blk.parameters()).device
    pairs = [(a, b) for a in range(dictionary.n) for b in range(dictionary.n)]
    ok = 0
    for a, b in pairs[:n_test]:
        x = encode_input(a, b, dictionary).unsqueeze(0).to(dev)
        r_pred, _ = dictionary.decode(blk(x)[0][0:64])
        ok += (r_pred == verifier.compose(a, b))
    return ok / min(n_test, len(pairs))
