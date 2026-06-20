#!/usr/bin/env python3
"""
EXPÉRIENCE alignement amodal (OCM-26400, spec §A1.3 + cahier des charges).

Démontre la brique « capturer en une passe + alignement amodal » : K modalités
d'un même concept convergent vers un vecteur unique ancré dans le dictionnaire
LearnedVocab. Implémente le terme d'ancrage déféré par le verdict (P1 InfoNCE +
P2 LearnedVocab).

Honnête : K=3 modalités SIMULÉES (encodeurs placeholder, concept_id -> R^64).
La math d'alignement + l'ancrage AMV sont réelles ; les modalités sont des
placeholders pour de futurs signaux réels (texte/audio/image). Pas de claim de
vrai multimodal (objection DA).
"""
import json
import torch

from ocm26400.learned_vocab import LearnedVocab
from ocm26400.concept_amodal import (
    ModalityEncoder, amodal_align_loss, train_amodal,
    cross_view_retrieval, anchor_decode_accuracy,
)

N = 128          # concepts (V > 64, impossible one-hot)
K = 3            # modalités simulées (placeholder texte/audio/image)
STEPS = 1000


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"OCM-26400 AMODAL | device={device} | {N} concepts | {K} modalités simulées")
    vocab = LearnedVocab(n=N, init="random", seed=0).freeze().to(device)
    print(f"LearnedVocab({N}) | cos_inter={vocab.mean_inter_pair_cos():.3f}")
    encoders = [ModalityEncoder(N, seed=s).to(device) for s in range(K)]

    r0 = cross_view_retrieval(encoders, N, device=device)
    a0 = anchor_decode_accuracy(encoders, vocab, N, device=device)
    print(f"\nAvant entraînement : retrieval@1={r0*100:5.1f}%  anchor_decode={a0*100:5.1f}%")

    # monitor la loss
    ids_full = torch.arange(N, device=device)
    for phase in range(4):
        train_amodal(vocab, encoders, N, n_steps=STEPS // 4, batch=64, device=device)
        with torch.no_grad():
            views = [enc(ids_full) for enc in encoders]
            _, parts = amodal_align_loss(views, vocab, ids_full)
        r = cross_view_retrieval(encoders, N, device=device)
        print(f"  step {(phase+1)*STEPS// 4:4d} | consist={parts['consist']:.3f} anchor={parts['anchor']:.3f} | retrieval@1={r*100:5.1f}%")

    r1 = cross_view_retrieval(encoders, N, device=device)
    a1 = anchor_decode_accuracy(encoders, vocab, N, device=device)
    print(f"\nAprès entraînement : retrieval@1={r1*100:5.1f}%  anchor_decode={a1*100:5.1f}%")
    verdict = "VALIDÉ" if (r1 > 0.9 and a1 > 0.7) else "NON VALIDÉ"
    print(f"VERDICT (alignement amodal) : {verdict}")
    print(f"  f_T(C)~f_A(C)~f_V(C)~v_C : {K} modalités alignées cross-vue à {r1*100:.0f}%, ancrées dans E à {a1*100:.0f}%.")

    results = {
        "task": "amodal concept alignment (spec §A1.3, capture-in-one-pass)",
        "n_concepts": N, "n_modalities": K, "modalities": "simulated (placeholder)",
        "anchor_term": "||f_view - E_C|| (verdict deferred term, now added via P2)",
        "retrieval_at_1_before": round(r0, 4), "retrieval_at_1_after": round(r1, 4),
        "anchor_decode_before": round(a0, 4), "anchor_decode_after": round(a1, 4),
        "verdict": verdict,
    }
    with open("ocm26400/amodal_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nRésultats: ocm26400/amodal_results.json")
    return results


if __name__ == "__main__":
    main()
