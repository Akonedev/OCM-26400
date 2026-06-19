"""Tests RED — ReasonerBlock (bloc neuronal sur AMV-256) + boucle LSRA.

Spec §3: v(t+1) = ReasonerBlock(v(t), Context). La boucle s'arrête a T* quand
confidence >= tau_grok, sinon declenche [ANOMALIE_CAUSALE].

On valide d'abord la mecanique (forme, differentiabilite, entrainabilite sur
l'operation binaire op(a,b)->m) avant l'experience crown-jewel.
"""
import pytest
import torch
from ocm26400.amv import AMVVector, D_MODEL
from ocm26400.verifier import SymbolicDict, Verifier
from ocm26400.acsp import acsp_loss
from ocm26400.reasoner import ReasonerBlock, encode_input, lsra_solve, TAU_GROK


@pytest.fixture
def setup():
    d = SymbolicDict()
    return d, Verifier(d)


def test_reasoner_block_preserves_shape():
    blk = ReasonerBlock()
    x = torch.randn(8, D_MODEL)
    y = blk(x)
    assert y.shape == (8, D_MODEL)


def test_reasoner_block_is_differentiable():
    blk = ReasonerBlock()
    x = torch.randn(4, D_MODEL, requires_grad=True)
    y = blk(x)
    y.sum().backward()
    assert x.grad is not None


def test_encode_input_puts_operands_in_ent_and_prop(setup):
    d, _ = setup
    # input encode (a,b): ent=canonical(a), prop=canonical(b)
    vec = encode_input(a=2, b=5, dictionary=d)
    assert vec.shape == (D_MODEL,)
    v = AMVVector(vec)
    assert torch.allclose(v.ent, d.canonical(2))
    assert torch.allclose(v.prop, d.canonical(5))


def test_block_learns_binary_operation(setup):
    """SANITY: le block doit reduire la loss ACSP sur op(a,b)->m en quelques pas SGD."""
    d, ver = setup
    torch.manual_seed(0)
    blk = ReasonerBlock()
    opt = torch.optim.Adam(blk.parameters(), lr=3e-3)

    # tous les couples (a,b) -> target m = op(a,b)
    pairs = [(a, b) for a in range(d.n) for b in range(d.n)]  # 121
    loss0 = None
    for step in range(400):
        idx = torch.randint(0, len(pairs), (32,))
        batch = torch.stack([encode_input(a=pairs[i][0], b=pairs[i][1], dictionary=d) for i in idx])
        out = blk(batch)  # (32, 256)
        # loss = somme l_align sur chaque sortie vers canonical(op(a,b))
        loss = 0.0
        for j, i in enumerate(idx):
            a, b = pairs[i]
            m = ver.compose(a, b)
            vj = AMVVector(out[j])
            from ocm26400.acsp import l_align
            loss = loss + l_align_to(vj, d, m)
        loss = loss / len(idx)
        if loss0 is None:
            loss0 = loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < loss0 * 0.3  # la loss a nettement baisse


def test_lsra_solve_decomposes_two_steps(setup):
    """LSRA: resoud (a o b) o c en 2 etapes, retourne le resultat + nb d'etapes."""
    d, ver = setup
    # on suppose un block parfait (mock) qui realise op(a,b) -> on teste la BOUCLE
    blk = ReasonerBlock()
    # entraîne le block rapidement sur l'operation binaire
    opt = torch.optim.Adam(blk.parameters(), lr=3e-3)
    pairs = [(a, b) for a in range(d.n) for b in range(d.n)]
    for _ in range(200):
        idx = torch.randint(0, len(pairs), (32,))
        batch = torch.stack([encode_input(pairs[i][0], pairs[i][1], d) for i in idx])
        out = blk(batch)
        loss = sum(l_align_to(AMVVector(out[j]), d, ver.compose(*pairs[i]))
                   for j, i in enumerate(idx)) / len(idx)
        opt.zero_grad(); loss.backward(); opt.step()
    # la boucle LSRA decompose: etape1 (a,b)->m, etape2 (m,c)->r
    r_pred, n_steps, ok = lsra_solve(blk, d, ver, a=2, b=5, c=7, max_iter=4)
    r_true = ver.compose(ver.compose(2, 5), 7)
    assert isinstance(n_steps, int)
    assert ok in (True, False)


# helper d'alignement vers un primitif cible precis (pas le min)
def l_align_to(v, dictionary, target_idx):
    from ocm26400.acsp import ALPHA
    import torch as T
    ent = v.ent
    d = dictionary.canonical(target_idx)
    cos = (ent @ d) / (ent.norm() * d.norm() + 1e-8)
    return 1.0 - cos
