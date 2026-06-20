#!/usr/bin/env python3
"""
EXPÉRIENCE vocabulaire anglais RÉEL à grande échelle (OCM-26400, cahier des charges §1).

Démontre l'addressage d'un VRAI vocabulaire anglais à grande échelle (370K mots réels
téléchargés, data/words_en.txt) via le mécanisme COMPOSITIONNEL : chaque mot est
adressé comme composition de ses CARACTÈRES (primitives a-z). L'espace adressable
n'est PAS borné par le slot 64-dim — c'est la voie paradigmatic vers « TOUT le
vocabulaire anglais (1M mots) ».

On indexe un grand échantillon de vrais mots (ex. 10 000), on mesure le retrieval@1
(un mot retrouvé-t-il lui-même dans l'index depuis son vecteur compositionnel ?) et la
séparabilité (distincts = vecteurs distincts).

Honnête : 370K ≠ 1M, mais c'est du VRAI vocabulaire anglais à l'échelle ~10^5, adressé
par composition (le mécanisme qui scale à 1M+). La séparabilité retrieval reste bornée
par le packing 64-dim à très grand N — levier dim>64 / index hiérarchique.
"""
import os, json, random, time
import torch

from ocm26400.learned_vocab import LearnedVocab
from ocm26400.compositional_vocab import CompositionalVocabulary

WORDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "words_en.txt")
N_INDEX = 10000        # mots réels indexés
N_EVAL = 500           # mots testés pour retrieval@1
MAX_LEN = 12           # longueur max (couvre la plupart des mots ; au-delà = tronqué)


def load_real_words(limit=None, max_len=MAX_LEN):
    """Charge de vrais mots anglais (a-z, longueur<=max_len)."""
    words = []
    with open(WORDS_FILE) as f:
        for line in f:
            w = line.strip().lower()
            if w and w.isalpha() and len(w) <= max_len:
                words.append(w)
    random.shuffle(words)
    return words[:limit] if limit else words


def main():
    random.seed(0); torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_words = load_real_words()
    print(f"OCM-26400 VOCABULAIRE ANGLAIS RÉEL | device={device}")
    print(f"Source : data/words_en.txt = {sum(1 for _ in open(WORDS_FILE)):,} mots anglais réels")
    print(f"Filtrés (a-z, <= {MAX_LEN} chars) : {len(all_words):,} mots adressables")

    # primitives = 26 caractères ; mots = composition de leurs chars
    prim = LearnedVocab(n=26, init="random", seed=0).freeze().to(device)
    cv = CompositionalVocabulary(prim, max_len=MAX_LEN).to(device)

    def word_to_chars(w):
        return [ord(c) - ord("a") for c in w[:MAX_LEN]]

    index_words = all_words[:N_INDEX]
    t0 = time.time()
    M, _ = cv.build_index([word_to_chars(w) for w in index_words])   # (N_INDEX, 64)
    dt_index = time.time() - t0
    print(f"Indexés : {N_INDEX:,} vrais mots (composition de chars), {dt_index:.1f}s")

    # retrieval@1 sur N_EVAL mots tirés de l'index
    eval_words = random.sample(index_words, N_EVAL)
    correct = 0
    for w in eval_words:
        found, conf = cv.retrieve(cv.word_vector(word_to_chars(w)), M,
                                  [word_to_chars(x) for x in index_words], threshold=0.0)
        correct += (found == word_to_chars(w))
    p1 = correct / N_EVAL

    # retrieval sur mots JAMAIS INDEXÉS (généralisation de l'addressage)
    oov_words = [w for w in all_words[N_INDEX:N_INDEX + N_EVAL]]
    oov_correct = 0
    for w in oov_words:
        ch = word_to_chars(w)
        found, _ = cv.retrieve(cv.word_vector(ch), M,
                               [word_to_chars(x) for x in index_words], threshold=0.0)
        oov_correct += (found == ch)   # ne devrait pas (OOV), mesure de collision
    oov_hit = oov_correct / len(oov_words)

    dt = time.time() - t0
    print(f"\nRetrieval@1 (mots indexés, n={N_EVAL})       : {p1*100:5.1f}%")
    print(f"Collision OOV (mots jamais indexés, n={len(oov_words)}) : {oov_hit*100:5.1f}% "
          f"(bas = peu de collisions, addressage distinct)")
    print(f"\n{len(all_words):,} vrais mots anglais adressables par composition de chars "
          f"(primitives={prim.n}, longueur<={MAX_LEN}).")
    verdict = "VALIDÉ" if p1 > 0.85 else "NON VALIDÉ"
    print(f"VERDICT (vocabulaire anglais réel à grande échelle par composition) : {verdict}")

    results = {
        "task": "vocabulaire anglais RÉEL à grande échelle (spec §1 '1M mots')",
        "source": "data/words_en.txt (dwyl/english-words, ~370K mots réels)",
        "real_words_addressable": len(all_words),
        "indexed": N_INDEX, "retrieval_at_1": round(p1, 4),
        "oov_collision_rate": round(oov_hit, 4),
        "mechanism": "composition de caractères (primitives a-z), addressage non borné par slot",
        "verdict": verdict, "duration_s": round(dt, 1),
    }
    with open("ocm26400/real_vocab_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nRésultats: ocm26400/real_vocab_results.json")
    return results


if __name__ == "__main__":
    main()
