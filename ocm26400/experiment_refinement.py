#!/usr/bin/env python3
"""
EXPÉRIENCE P3 — gate de confiance calibrée + abstention (OCM-26400, P3 reframeé).

HONNÊTETÉ (verdict panel d'experts 19/06) : le design ORIGINAL de P3 (supervision
d'une trajectoire géométrique ent_t = (1-λ)^t·canonical(a) + (1-(1-λ)^t)·canonical(op(a,b)),
λ<0.5) était TAUTOLOGIQUE — le 1-pas faux et la convergence à T* pas sont vrais PAR
CONSTRUCTION. Le DA l'a enterré. Cette version est REFRAMEÉE en claim d'ingénierie
honnête, non circulaire.

Pourquoi on ne démontre PAS du « raffinement par itération améliore l'accuracy » :
le ReasonerBlock est résiduel (v+f(v)) entraîné à mapper (a,b)->op(a,b). Itérer
v=blk(v) ne raffine pas vers la cible — il RECOMPOSE (prop reste b, donc l'étape
suivante calcule op(op(a,b),b)...). « Itérer améliore la précision sur la même
cible » est incohérent pour ce block. (Confirmation du DA : le block n'est pas une
contraction de Banach, et l'itération = composition, pas raffinement.)

La valeur HONNÊTE et NON-TAUTOLOGIQUE de la boucle LSRA (spec §3) est la GATE :
un forward 1-step retourne TOUJOURS une réponse (il hallucine même sur du garbage) ;
la boucle LSRA + gate de confiance calibrée permet au système de SAVOIR qu'il est
incertain et de REFUSER de répondre ([ANOMALIE_CAUSALE], confident=False).

On entraîne un block CALIBRÉ :
  * entrées valides (a,b)          -> ent = canonical(op(a,b)), meta[0] HAUT (confiant)
  * entrées OOD (ent = bruit pur)  -> meta[0] BAS (peu confiant)

Puis on mesure, sur des entrées valides ET OOD jamais vues :
  1. CALIBRATION : la confidence sépare-t-elle valid (haut) de OOD (bas) ?
  2. ABSTENTION via lsra_loop : valid -> confident=True (accepte) ; OOD -> confident=False
     (ANOMALIE, refuse). C'est l'apport sur le 1-step qui n'abstient jamais.

On réutilise lsra_loop (reasoner.py:110) SANS modifier sa signature.
Claim : ENGINEERING (incertitude épistémique / abstention), PAS crown-jewel.
"""
import json, random, time
import torch

from ocm26400.amv import D_MODEL
from ocm26400.verifier import SymbolicDict, Verifier, P_MOD
from ocm26400.reasoner import ReasonerBlock, encode_input, lsra_loop, TAU_GROK, DEVICE

SEED = 0
CONF_HIGH = 4.0    # meta[0] cible pour entrée valide (sigmoid(4)≈0.98 > tau_grok)
CONF_LOW = -4.0    # meta[0] cible pour entrée OOD (sigmoid(-4)≈0.018, jamais confiant)


def make_ood_input(d, n, device=DEVICE):
    """Entrée OOD : ent = bruit gaussien (pas un symbole one-hot), prop = symbole aléatoire."""
    v = torch.zeros(D_MODEL, device=device)
    v[0:64] = torch.randn(64, device=device)
    v[64:128] = d.canonical(random.randrange(n)).to(device)
    return v


def train_calibrated_block(d, ver, n_steps=1000, lr=3e-3, batch=128, device=DEVICE):
    """Block calibré : confiant sur les mappings valides, peu confiant sur OOD."""
    torch.manual_seed(SEED)
    random.seed(SEED)
    blk = ReasonerBlock().to(device)
    opt = torch.optim.Adam(blk.parameters(), lr=lr)
    n = d.n
    half = batch // 2
    for _ in range(n_steps):
        vp = [(random.randrange(n), random.randrange(n)) for _ in range(half)]
        valid_in = torch.stack([encode_input(a, b, d) for a, b in vp]).to(device)
        ood_in = torch.zeros(half, D_MODEL, device=device)
        ood_in[:, 0:64] = torch.randn(half, 64, device=device)
        for j in range(half):
            ood_in[j, 64:128] = d.canonical(random.randrange(n))
        out = blk(torch.cat([valid_in, ood_in], dim=0))
        loss = torch.tensor(0.0, device=device)
        for j, (a, b) in enumerate(vp):
            ent = out[j, 0:64]
            tgt = d.canonical(ver.compose(a, b)).to(device)
            cos = (ent @ tgt) / (ent.norm() * tgt.norm() + 1e-8)
            loss = loss + (1.0 - cos) + (out[j, 192] - CONF_HIGH) ** 2
        for j in range(half, batch):
            loss = loss + (out[j, 192] - CONF_LOW) ** 2
        loss = loss / batch
        opt.zero_grad(); loss.backward(); opt.step()
    return blk


@torch.no_grad()
def confidence(blk, x):
    blk.eval()
    dev = next(blk.parameters()).device
    out = blk(x.unsqueeze(0).to(dev))[0]
    return float(torch.sigmoid(out[192]).item())


@torch.no_grad()
def one_step_decode(blk, d, x):
    """Forward 1-step + decode (retourne TOUJOURS une réponse, n'abstient jamais)."""
    blk.eval()
    dev = next(blk.parameters()).device
    out = blk(x.unsqueeze(0).to(dev))[0]
    idx, _ = d.decode(out[0:64])
    return idx


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    d = SymbolicDict(n=P_MOD)                       # Z_11 one-hot
    ver = Verifier(d)
    n = d.n
    print(f"OCM-26400 P3 (gate calibrée + abstention) | device={DEVICE} | Z_{n}")
    print(f"Frame HONNÊTE : engineering claim (incertitude/abstention), PAS crown-jewel.")
    print(f"Le 1-step hallucine toujours ; lsra_loop + gate calibrée REFUSE l'OOD.\n")

    t0 = time.time()
    blk = train_calibrated_block(d, ver, n_steps=1000)
    dt_train = time.time() - t0

    # jeux de test jamais vus : 200 valides, 200 OOD
    valid_pairs = [(random.randrange(n), random.randrange(n)) for _ in range(200)]
    valid_x = [encode_input(a, b, d) for a, b in valid_pairs]
    ood_x = [make_ood_input(d, n) for _ in range(200)]

    # --- Démo 1 : CALIBRATION ---
    conf_valid = [confidence(blk, x) for x in valid_x]
    conf_ood = [confidence(blk, x) for x in ood_x]
    mv = sum(conf_valid) / len(conf_valid)
    mo = sum(conf_ood) / len(conf_ood)
    # separation/AUROC simplifié : fraction de paires (valid,ood) où conf_valid > conf_ood
    sep = sum(1 for cv in conf_valid for co in conf_ood if cv > co) / (len(conf_valid) * len(conf_ood))

    # --- Démo 2 : ABSTENTION via lsra_loop ---
    accept_valid = 0          # valid -> confident=True (bon : accepte)
    correct_valid = 0
    steps_valid = []
    for x, (a, b) in zip(valid_x, valid_pairs):
        idx, steps, conf = lsra_loop(blk, d, x, max_iter=8, tau=TAU_GROK)
        accept_valid += conf
        steps_valid.append(steps)
        correct_valid += (conf and idx == ver.compose(a, b))
    reject_ood = 0            # OOD -> confident=False (bon : ANOMALIE, refuse)
    for x in ood_x:
        _, _, conf = lsra_loop(blk, d, x, max_iter=8, tau=TAU_GROK)
        reject_ood += (not conf)

    dt = time.time() - t0
    accept_rate = accept_valid / len(valid_x)
    reject_rate = reject_ood / len(ood_x)
    mean_steps = sum(steps_valid) / len(steps_valid)

    print(f"--- CALIBRATION (confidence sigmoid(meta[0])) ---")
    print(f"conf moyenne  valides : {mv:.3f}   (cible > {TAU_GROK})")
    print(f"conf moyenne  OOD     : {mo:.3f}   (cible << {TAU_GROK})")
    print(f"séparation (AUROC proxy valid>OOD) : {sep:.3f}")
    print(f"\n--- ABSTENTION via lsra_loop (tau={TAU_GROK}) ---")
    print(f"valides -> accepte (confident=True) : {accept_rate*100:5.1f}%   (correct + confiant: {correct_valid}/{len(valid_x)})")
    print(f"OOD     -> refuse  (ANOMALIE)       : {reject_rate*100:5.1f}%")
    print(f"nb moyen d'itérations sur valides   : {mean_steps:.2f}  (stop anticipé par la gate)")
    print(f"\n1-step n'abstient JAMAIS (retourne toujours un idx) ; la gate calibrée refuse {reject_rate*100:.0f}% de l'OOD.")
    verdict = "VALIDÉ" if (accept_rate > 0.8 and reject_rate > 0.8) else "NON VALIDÉ"
    print(f"VERDIT (engineering) : {verdict}  (accept>80% ET reject>80%)")

    results = {
        "task": "P3 calibrated confidence gate + abstention (honest engineering claim)",
        "frame": "epistemic uncertainty / abstention, NOT crown-jewel TTC-buys-accuracy",
        "n_primitives": n, "train_steps": 1000, "train_s": round(dt_train, 1),
        "calibration": {
            "mean_conf_valid": round(mv, 4), "mean_conf_ood": round(mo, 4),
            "separation_auroc_proxy": round(sep, 4),
        },
        "abstention": {
            "valid_accept_rate": round(accept_rate, 4),
            "valid_correct_and_confident": correct_valid,
            "ood_reject_rate": round(reject_rate, 4),
            "mean_steps_valid": round(mean_steps, 2),
        },
        "verdict": verdict, "duration_s": round(dt, 1),
    }
    with open("ocm26400/refinement_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nRésultats: ocm26400/refinement_results.json")
    return results


if __name__ == "__main__":
    main()
