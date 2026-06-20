#!/usr/bin/env python3
"""
MESURE v6 SINGLE-FORWARD vs DIFFUSE_FILL (read-only, P4-unblock).

Question (verdict panel d'experts, pièce P4) : le checkpoint v6_full_vocab_v3
donne best_avg=0.695, MAIS ce chiffre vient de model.diffuse_fill(n_steps=4)
(model.py:375) — un RAFFINEMENT ITÉRATIF (4 forwards successifs qui complètent
le contexte). Le pont v6→AMV (P4) consommerait un SINGLE forward
(h = blocks(token_emb+pos) puis projection), SANS la boucle de raffinement.

Cette mesure établit la prémisse manquante : accuracy fill-in-blank en
SINGLE forward (n_steps=1) vs diffuse_fill (n_steps=4), sur le MÊME jeu de test
que verify_v6.py (seed=0, 120 mots). Honnête, sans triche.

DiffuseFiller.fill / diffuse_fill font :
    for step in range(n_steps):
        logits = model.forward(x)        # forward complet
        pred  = logits.argmax(-1)
        x     = where(mask, pred, x)     # remplit les masques
n_steps=1 = exactement UN forward sur l'entrée masquée = single-forward.
n_steps=4 = 4 forwards (raffinement itératif, contexte qui se complète).

READ-ONLY : ne modifie PAS model.py ni aucun checkpoint. Crée juste la mesure.
"""
import torch, json, random
from model import SpXLMv6
from protocol_full_vocab_v3 import (
    CharTokenizer, encode_entry, find_field, load_dataset, get_field_value,
    FIELD_DEFS, SEQ_LEN,
)

CKPT = "v6_full_vocab_v3_model.pt"
STEPS_SINGLE = 1      # single-forward
STEPS_DIFFUSE = 4     # raffinement itératif (référence 0.695)
device = "cuda" if torch.cuda.is_available() else "cpu"

tok = CharTokenizer()
data = load_dataset(n_words=800)
random.seed(0)
test = random.sample(data, min(120, len(data)))   # MÊME split que verify_v6.py
print(f"device={device} | vocab={tok.vocab_size} | test_words={len(test)}")

model = SpXLMv6(vocab_size=tok.vocab_size, d_model=256, n_blocks=4,
                seq_len=SEQ_LEN, mode="reasoning").to(device)
sd = torch.load(CKPT, map_location=device, weights_only=True)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"load: missing={len(missing)} unexpected={len(unexpected)} "
      f"params={sum(p.numel() for p in model.parameters()):,}")
model.eval()


def detok_range(ids, s, e):
    return "".join(tok.itos.get(int(i), "?") for i in ids[s:e])


def eval_steps(n_steps):
    """Accuracy fill-in-blank par champ pour un n_steps donné (forward(s))."""
    maskable = [f[0] for f in FIELD_DEFS if f[0] not in ("pos", "wid")]
    acc = {}
    for fname in maskable:
        ok = 0; n = 0
        for props in test:
            ids, fields, _ = encode_entry(props, tok)
            x = torch.tensor([ids], device=device)
            fs, fe = find_field(fields, fname)
            mask = torch.zeros((1, SEQ_LEN), dtype=torch.bool, device=device)
            mask[0, fs:fe] = True
            x_in = x.clone(); x_in[mask] = tok.mask
            out = model.diffuse_fill(x_in, mask, n_steps=n_steps)   # n_steps=1 => single forward
            pred = detok_range(out[0].tolist(), fs, fe).rstrip("-")
            exp = ""
            for fn, fw, ft in FIELD_DEFS:
                if fn == fname:
                    exp = get_field_value(props, fname).rstrip("-"); break
            n += 1
            if pred == exp:
                ok += 1
        acc[fname] = ok / n if n else 0.0
    return acc


print(f"\n=== SINGLE-FORWARD (n_steps={STEPS_SINGLE}) vs DIFFUSE_FILL (n_steps={STEPS_DIFFUSE}) ===")
acc1 = eval_steps(STEPS_SINGLE)
acc4 = eval_steps(STEPS_DIFFUSE)
maskable = list(acc1.keys())

print(f"\n  {'champ':11} {'single':>8} {'diffuse':>8} {'gap':>7}")
for f in maskable:
    gap = acc4[f] - acc1[f]
    print(f"  {f:11} {acc1[f]*100:7.1f}% {acc4[f]*100:7.1f}% {gap*100:+6.1f}pt")

avg1 = sum(acc1.values()) / len(acc1)
avg4 = sum(acc4.values()) / len(acc4)
# average alignée sur les 11 champs du JSON d'entraînement (exclut 'word' = 0% irrécupérable)
JSON_FIELDS = ["plural", "past", "gerund", "third", "phoneme", "length",
               "vowels", "consonants", "syllables", "cat_id", "syn_id"]
j1 = sum(acc1[f] for f in JSON_FIELDS) / len(JSON_FIELDS)
j4 = sum(acc4[f] for f in JSON_FIELDS) / len(JSON_FIELDS)
print(f"\n  {'AVG (tous maskable)':22} {avg1*100:7.1f}% {avg4*100:7.1f}% {(avg4-avg1)*100:+6.1f}pt")
print(f"  {'AVG (11 champs JSON)':22} {j1*100:7.1f}% {j4*100:7.1f}% {(j4-j1)*100:+6.1f}pt")
print(f"\nRéférence JSON best_avg (11 champs, eval ENTRAÎNEMENT plus dure) = 0.695")
print(f"Ici (seed=0, 120 mots, eval 1-champ-masqué PLUS FACILE) :")
print(f"  single={j1*100:.1f}%  diffuse={j4*100:.1f}%  (11 champs JSON-alignés)")
print(f"Coût du raffinement itératif (within-eval, propre) : {(j4-j1)*100:+.1f}pt  ← négatif = le raffinement HURTS")

# Le verdict propre se base sur la comparaison within-eval (single vs diffuse, même eval)
# ET sur les champs linguistiques discriminatifs (non gonflés par faible couverture).
LING = ["plural", "past", "gerund", "third", "phoneme"]   # morpho/phon, coverage pleine
l1 = sum(acc1[f] for f in LING) / len(LING)
verdict = (
    "PRÉMISSE P4 TIENT" if (j1 >= j4 and l1 > 0.6) else
    "PRÉMISSE P4 FRAGILE" if l1 > 0.45 else
    "PRÉMISSE P4 INVALIDÉE (single-forward trop bas)"
)
print(f"\nChamps linguistiques discriminatifs (single) : {l1*100:.1f}%")
print(f"VERDICT P4 : {verdict}")
print(f"  → single ≥ diffuse (within-eval) : {j1 >= j4}  |  linguistique single : {l1*100:.1f}%")
print(f"  → réfute l'estimation DA (single ~50-60% < 0.695) : single={j1*100:.1f}% ≥ diffuse={j4*100:.1f}%")

results = {
    "task": "v6 single-forward vs diffuse_fill (P4 unblock, read-only)",
    "ckpt": CKPT, "test_words": len(test), "seed": 0,
    "eval_note": "verify_v6 eval (1 champ masqué) est PLUS FACILE que l'eval entraînement (0.695); "
                 "cat_id/syn_id gonflés par faible couverture (59%/3%). Comparaison within-eval = propre.",
    "n_steps_single": STEPS_SINGLE, "n_steps_diffuse": STEPS_DIFFUSE,
    "per_field_single": {k: round(v, 4) for k, v in acc1.items()},
    "per_field_diffuse": {k: round(v, 4) for k, v in acc4.items()},
    "avg_all_single": round(avg1, 4), "avg_all_diffuse": round(avg4, 4),
    "avg_11json_single": round(j1, 4), "avg_11json_diffuse": round(j4, 4),
    "linguistic_single": round(l1, 4),
    "gap_11json_points": round(j4 - j1, 4),
    "reference_json_best_avg_training_eval": 0.695,
    "verdict_P4": verdict,
}
with open("v6_single_forward_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nRésultats: spxlm_v6/v6_single_forward_results.json")
