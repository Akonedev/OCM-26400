"""Tokenizer BPE (Byte Pair Encoding) — réfute audit M16 CRITIQUE.

EX-B256, M16. Le BPE tokenise en sous-mots par fusions successives des paires les plus
fréquentes. Débloque la génération > 8 chars (le text_decoder était char-level). BPE est
la primitive de tokenization standard (GPT, etc.).

* train(corpus, vocab_size) : apprend les fusions BPE (paires les + fréquentes).
* encode(text) : texte → séquence de tokens.
* decode(tokens) : tokens → texte.
Vérifiable : encode(decode(x))≈x, vocab borné, fusions croissantes. Branchable sur le
text_decoder pour la génération à grande échelle.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


class BPETokenizer:
    """BPE : fusions de paires de symboles les plus fréquentes."""

    def __init__(self):
        self.merges: List[Tuple[str, str]] = []   # ordre des fusions apprises
        self.vocab: Dict[str, int] = {}
        self._merges_rank: Dict[Tuple[str, str], int] = {}

    @staticmethod
    def _get_pair_counts(word_splits: Dict[Tuple[str, ...], int]) -> Counter:
        pairs = Counter()
        for word, freq in word_splits.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    @staticmethod
    def _merge(word: Tuple[str, ...], pair: Tuple[str, str], new_sym: str) -> Tuple[str, ...]:
        out, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                out.append(new_sym)
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    def train(self, corpus: List[str], vocab_size: int = 200) -> None:
        """Apprend les fusions BPE sur le corpus (mots splittés en caractères + █ fin)."""
        word_freq: Dict[Tuple[str, ...], int] = defaultdict(int)
        for text in corpus:
            for w in text.lower().split():
                w = tuple(list(w) + ["█"])
                word_freq[w] += 1
        # vocab initial = caractères
        vocab = set()
        for w in word_freq:
            vocab.update(w)
        self.merges = []
        while len(vocab) < vocab_size:
            pairs = self._get_pair_counts(word_freq)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_sym = best[0] + best[1]
            self.merges.append(best)
            vocab.add(new_sym)
            # re-split
            new_wf = {}
            for w, f in word_freq.items():
                new_wf[self._merge(w, best, new_sym)] = f
            word_freq = new_wf
        self.vocab = {sym: i for i, sym in enumerate(sorted(vocab))}
        self._merges_rank = {m: i for i, m in enumerate(self.merges)}

    def _encode_word(self, word: str) -> List[str]:
        symbols = list(word) + ["█"]
        while len(symbols) > 1:
            # trouver la paire de rang le plus bas (fusion la + précoce apprise)
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            ranked = [(self._merges_rank[p], i, p) for i, p in enumerate(pairs)
                      if p in self._merges_rank]
            if not ranked:
                break
            _, i, pair = min(ranked)
            new_sym = pair[0] + pair[1]
            symbols = symbols[:i] + [new_sym] + symbols[i + 2:]
        return symbols

    def encode(self, text: str) -> List[int]:
        ids = []
        for w in text.lower().split():
            for sym in self._encode_word(w):
                ids.append(self.vocab.get(sym, self.vocab.get("<unk>", 0)))
        return ids

    def decode(self, ids: List[int]) -> str:
        inv = {i: s for s, i in self.vocab.items()}
        text = "".join(inv.get(i, "") for i in ids)
        return text.replace("█", " ").strip()


def train_default(corpus: List[str] = None, vocab_size: int = 150) -> BPETokenizer:
    if corpus is None:
        corpus = ["le chat mange une souris", "le chien dort dans la maison",
                  "la souris mange du fromage", "un chat noir sur le toit",
                  "le grand chien court vite", "la petite souris fuit le chat",
                  "manger dormir courir parler chanter", "grand petit noir blanc rouge"] * 5
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=vocab_size)
    return tok


if __name__ == "__main__":
    tok = train_default()
    print(f"[bpe] vocab={len(tok.vocab)} fusions={len(tok.merges)}")
    for s in ["chat", "manger", "chat mange"]:
        ids = tok.encode(s)
        print(f"  '{s}' → {ids} → '{tok.decode(ids)}'")
