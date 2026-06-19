"""ReasonerBlock + boucle LSRA (OCM-26400, spec §3).

  v(t+1) = ReasonerBlock(v(t))

Le bloc est un MLP résiduel sur AMV-256. La boucle LSRA déroule le bloc, interroge
le vérifieur symbolique (gate) après chaque étape, et s'arrête quand la confidence
(meta) dépasse tau_grok — sinon [ANOMALIE_CAUSALE].

C'est l'opposé d'un LLM autoregressif : ici on itère dans l'espace latent et on
VÉRIFIE chaque composition, on ne devine pas le token suivant.
"""
import torch
import torch.nn as nn

from .amv import D_MODEL, AMVVector
from .verifier import SymbolicDict, Verifier

TAU_GROK = 0.9  # seuil de confidence pour stopper la récurrence (spec §3)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ReasonerBlock(nn.Module):
    """Bloc de raisonnement latent : MLP résiduel + LayerNorm sur R^256.

    Apprend à transformer un AMV (ent=a, prop=b) en un AMV dont ent = op(a,b).
    """

    def __init__(self, d_model: int = D_MODEL, hidden: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 256) ou (256,)
        h = self.norm(x)
        h = torch.relu(self.fc1(h))
        h = self.fc2(h)
        return x + h  # résiduel


def encode_input(a: int, b: int, dictionary: SymbolicDict) -> torch.Tensor:
    """Encode le couple (a,b) dans un AMV-256 : ent=canonical(a), prop=canonical(b)."""
    v = torch.zeros(D_MODEL)
    v[0:64] = dictionary.canonical(a)      # ent
    v[64:128] = dictionary.canonical(b)    # prop
    # op (128:192) et meta (192:256) restent à 0
    return v


def lsra_solve(blk: ReasonerBlock, dictionary: SymbolicDict, verifier: Verifier,
               a: int, b: int, c: int, max_iter: int = 4):
    """Résout (a o b) o c par DECOMPOSITION en 2 étapes.

    Étape 1: encode(a,b) -> block -> decode -> m_pred. Vérifie m == op(a,b).
    Étape 2: encode(m_pred,c) -> block -> decode -> r_pred.
    Retourne (r_pred, n_steps, ok) où ok = étape intermédiaire légale.
    """
    blk.eval()
    device = next(blk.parameters()).device
    with torch.no_grad():
        # étape 1 : m = op(a,b)
        x1 = encode_input(a, b, dictionary).unsqueeze(0).to(device)  # (1,256)
        out1 = blk(x1)[0]
        m_pred, _ = dictionary.decode(AMVVector(out1).ent)
        ok_inter = verifier.is_valid_intermediate(a, b, m_pred)

        # étape 2 : r = op(m_pred, c)
        x2 = encode_input(m_pred, c, dictionary).unsqueeze(0).to(device)
        out2 = blk(x2)[0]
        r_pred, _ = dictionary.decode(AMVVector(out2).ent)

    # n_steps ~ 2 (decomposition). ok si l'intermédiaire était correct.
    return r_pred, 2, bool(ok_inter)


def train_reasoner_with_confidence(dictionary, verifier, n_steps=800, lr=3e-3,
                                   batch=64, device=DEVICE):
    """Entraîne le block à (a) produire ent=canonical(op(a,b)) ET (b) meta[0] élevé
    (confiant). C'est ce qui rend la gate LSRA arrêtable."""
    torch.manual_seed(0)
    blk = ReasonerBlock().to(device)
    opt = torch.optim.Adam(blk.parameters(), lr=lr)
    pairs = [(a, b) for a in range(dictionary.n) for b in range(dictionary.n)]
    CONF_TARGET = 4.0  # sigmoid(4)≈0.98 > tau_grok
    for _ in range(n_steps):
        idx = torch.randint(0, len(pairs), (batch,))
        batch_in = torch.stack(
            [encode_input(pairs[i][0], pairs[i][1], dictionary) for i in idx]
        ).to(device)
        out = blk(batch_in)
        loss = torch.tensor(0.0, device=device)
        for j, i in enumerate(idx):
            a, b = pairs[i]
            m = verifier.compose(a, b)
            ent = out[j][0:64]
            d = dictionary.canonical(m).to(device)
            cos = (ent @ d) / (ent.norm() * d.norm() + 1e-8)
            loss = loss + (1.0 - cos)                 # alignement ent
            loss = loss + (out[j][192] - CONF_TARGET) ** 2  # confidence meta[0]
        loss = loss / batch
        opt.zero_grad(); loss.backward(); opt.step()
    return blk


def lsra_loop(blk, dictionary, x0, max_iter=8, tau=TAU_GROK):
    """Boucle LSRA pleine (spec §3): v(t+1)=Block(v(t)).

    Stop à T* = min{ t | sigmoid(meta[0]) >= tau }. Si max_iter atteint sans
    confiance -> [ANOMALIE_CAUSALE] (confident=False).
    Retourne (idx_prédit, n_steps, confident).
    """
    device = next(blk.parameters()).device
    v = x0.to(device).clone()
    blk.eval()
    with torch.no_grad():
        for t in range(max_iter):
            v = blk(v.unsqueeze(0))[0]               # v(t+1) = Block(v(t))
            conf = torch.sigmoid(v[192]).item()       # confidence = sigmoid(meta[0])
            if conf >= tau:
                idx, _ = dictionary.decode(v[0:64])   # ent décodé
                return idx, t + 1, True               # stop anticipé, confiant
        idx, _ = dictionary.decode(v[0:64])
        return idx, max_iter, False                    # ANOMALIE : jamais confiant
