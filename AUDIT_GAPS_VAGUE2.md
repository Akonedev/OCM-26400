# AUDIT GAPS — VAGUE 2 (Densification)

**Projet** : OCM-26400 — Modèle cognitif unifié (SpectralCoreBlock)
**Date** : 2026-06-20
**Auteur** : EXPERT-DENSIFICATION
**Pré-requis** : Vague 1 terminée (top-15 de `AUDIT_GAPS_DENSIFICATION.md` comblé). 66 modules de capacité, ~790 tests verts, BENCH_LEVEL 94.9/100.

---

## 0. Méthodologie

1. Lecture exhaustive de `Besoins/Besoins_Tests.md` (~73 exigences EX-T), `Besoins/Besoins.md` (~305 exigences EX-B), `Besoins/Besoins_Maths.md` (~45 exigences EX-M + 3 claims brevet + loi depth_max).
2. Recoupement avec `CAPABILITIES.md`, `ocm26400/STATUS.md`, `ocm26400/__init__.py` pour écarter les doublons.
3. **Exclusion explicite** des 6 modalités à corpus externe (bloqueurs honnêtes) : vidéo (VideoMME), parole (LibriSpeech), OCR (IAM), object detection (COCO), radar (SENTINEL-1), code-à-l'échelle (GitHub). Aucune des 20 capacités ci-dessous n'exige de corpus étiqueté.
4. **Respect de l'architecture** : `SpectralCoreBlock` (noyau unifié FFT, `d_model=256`, O(L log L)) est FIGÉ. Pas de Mamba, pas de wrapper, pas de transformer autoregressif massif. Paradigme : **primitives → grok → composer → généraliser** (params constants ≈ 263K, raisonnement = étapes, pas params).
5. Chaque capacité est **implémentable sans corpus externe** (synthétique, règles, SymPy, torch pur).

---

## 1. Synthèse de l'état post-vague 1

| Catégorie | Comblé vague 1 | Manquant (cible vague 2) |
|---|---|---|
| Crown-jewel neural (C1/C2/C10) | `neural_multihop.py`, `cot_arithmetic.py` | — |
| Maths symboliques (C3/M17) | `symbolic_math.py`, `equation_solver.py` (SymPy) | Quaternions/octonions, Dijkstra/A*, PID, Jacobien, LNS/RNS |
| Physique units (C3) | `physics_units.py` | — |
| Linguistique FR (C4) | `morphology_fr.py` (conjugaison) | Syntaxe/SVO, phonologie/IPA, étymologie, sèmes, collocations |
| Décodeur texte (C6/C9) | `text_decoder.py` (char-level max_len=8) | BPE tokenizer pour scale phrases |
| RAG (H10) | `document_learner.py` (single-hop) | Multi-hop + re-ranking |
| Browser/Computer (H1/H2/D26) | `browser_tool.py`, `computer_use.py` | — |
| Code (H3) | `code_generator.py` (template+exec) | Compression/simplification entraînée |
| OWASP (H15) | `owasp_scanner.py` | OSINT |
| Sommeil (H19) | `sleep_phases.py` | — |
| Cognition (M1/M2/M4/M5/M10) | `cognition.py`, `commonsense.py` | Calibration Brier, MCTS latent, sarcasme, explication |
| Continual/RL | (rien) | EWC/SI, DPO/GRPO, curiosité |

**Découvertes lors de l'audit** :
- `logic_engine.py` (logique propositionnelle : tautologies, tables de vérité, modus ponens) existe et est exporté — ne pas dupliquer.
- `chemistry.py`, `finance.py`, `genetics.py` existent et sont exportés dans `__init__.py`, **mais n'ont aucun test dédié** (`test_chemistry.py` / `test_genetics.py` / `test_finance.py` absents) — action de robustesse prioritaire (cf. §5).
- `domain_trainer.py::evaluate_all_domains` et `reasoning_bench_aime` sont des **tautologies** (`apply == apply`, §2.3 de l'audit vague 1) — à réimplémenter via le core neural ou à retirer du BENCH_LEVEL.

---

## 2. Tableau priorisé — TOP 20 capacités de la vague 2

Légende priorité : **CRITIQUE** = blocante pour le paradigme ou explicitement exigée par le cahier des charges comme pré-requis archi ; **HAUTE** = exigée nommément, ROI fort ; **MOYENNE** = exigée, ROI modéré. "Sans corpus" = implémentable avec données synthétiques/règles/torch pur.

| # | ID gap | Capacité | Priorité | Sans corpus | Fichier cible | Lien cahier des charges |
|---|---|---|---|---|---|---|
| 1 | **M-QUAT** | Quaternions / octonions / sédénions / Cayley-Dickson / Hurwitz | **CRITIQUE** | Oui | `hypercomplex.py` | EX-B284 à EX-B288 |
| 2 | **M16** | Tokenizer BPE branchable (scale phrases > 8 chars) | **CRITIQUE** | Oui | `bpe_tokenizer.py` | EX-B256, M16 |
| 3 | **C5** | Analyseur syntaxique S-V-C + dépendances | **CRITIQUE** | Oui | `syntax_parser.py` | EX-T8, EX-B192, EX-B198 |
| 4 | **MCTS** | Planification MCTS latent + backtracking (test-time compute) | **CRITIQUE** | Oui | `mcts_planner.py` | EX-M (maths), EX-B205 |
| 5 | **M19** | Continual learning EWC/SI (anti catastrophic forgetting) | **CRITIQUE** | Oui | `continual_learning.py` | EX-B187, EX-M28-31, M19 |
| 6 | **L5** | Phonologie + transcription IPA | **HAUTE** | Oui | `phonology.py` | EX-T6, L5 |
| 7 | **L7** | Étymologie (radical / lexème / morphème / affixes) | **HAUTE** | Oui | `etymology.py` | EX-T7, EX-B197 |
| 8 | **L11** | Sémantique fine (traits / sèmes / polysémie vs homonymie) | **HAUTE** | Oui | `semantic_traits.py` | EX-T8, EX-B196, L11 |
| 9 | **L13** | Mots composés / collocations / locutions figées | **HAUTE** | Oui | `collocations.py` | EX-B197, L13 |
| 10 | **M20** | RL post-training DPO / GRPO | **HAUTE** | Oui | `rl_posttraining.py` | EX-B213, M20 |
| 11 | **M21** | Curiosité / exploration active (récompense intrinsèque) | **HAUTE** | Oui | `curiosity.py` | EX-B173, EX-B185, M21 |
| 12 | **H18** | Calibration Brier + conscience épistémique des lacunes | **HAUTE** | Oui | `calibration.py` | EX-B186, EX-B202, H18 |
| 13 | **M9** | Explication structurée ("pourquoi cette réponse") | **HAUTE** | Oui | `explainer.py` | EX-B191, M9 |
| 14 | **M11** | RAG multi-hop retrieval + re-ranking | **HAUTE** | Oui | `multihop_rag.py` | M11 |
| 15 | **G-ALGO** | Graph algorithms : Dijkstra + A* + BFS/DFS | **HAUTE** | Oui | `graph_algorithms.py` | EX-B297, EX-B298 |
| 16 | **SIG** | Traitement du signal + PID controller + spectre | **MOYENNE** | Oui | `signal_processing.py` | EX-B292 à EX-B296, EX-B289 |
| 17 | **JAC** | Jacobien + changement de variables + résidus | **MOYENNE** | Oui | `jacobian.py` | EX-B289 à EX-B301 |
| 18 | **M3** | Sarcasme / ironie (classifieur entraîné) | **MOYENNE** | Oui | `sarcasm.py` | M3 |
| 19 | **M12** | World model neuronal JEPA-prédictif (prédire état t+1) | **MOYENNE** | Oui | `world_model.py` | M12, EX-B205 |
| 20 | **OSINT** | OSINT scanner (domaines, emails, fuites, DNS) | **MOYENNE** | Oui | `osint_scanner.py` | EX-T12, EX-B199 |

**Bonus (hors top 20, si budget restant)** : M7 Abstraction/catégorisation (`abstraction.py`), EX-B304-305 LNS/RNS (`alt_number_systems.py`), EX-B290 Diagonalisation matrices (étendre `equation_solver.py`), M18 Compression code (`code_compressor.py`).

---

## 3. Plan d'implémentation concret (top 20)

### CRITIQUE

#### 1. `hypercomplex.py` — Algèbres hypercomplexes (M-QUAT)
**Exigence cahier** : EX-B284 (quaternions Hamilton, rotation 3D sans gimbal lock), EX-B285 (octonions dim 8), EX-B286 (sédénions dim 16), EX-B287 (construction Cayley-Dickson), EX-B288 (théorème de Hurwitz : seules dim 1,2,4,8 admettent algèbre à division).
**Justification archi** : le `SpectralCoreBlock` opère dans le domaine fréquentiel complexe ; les quaternions généralisent naturellement les complexes et justifient la rotation de phase des filtres `filter_real`/`filter_imag`. C'est la **justification mathématique du choix d'archi** (EX-B281, "maths des types de réseaux de la solution").
**API** :
```python
class Quaternion:
    def __init__(self, w, x, y, z): ...
    def __mul__(self, other) -> Quaternion: ...   # hamilton product
    def conjugate(self) -> Quaternion: ...
    def norm(self) -> float: ...
    def rotate(self, v: tuple) -> tuple: ...      # rotation 3D sans gimbal lock
    def to_rotation_matrix(self) -> list: ...

def cayley_dickson(a: tuple, b: tuple) -> tuple:  # construit C depuis R, H depuis C, O depuis H, S depuis O
    ...

def is_division_algebra(dim: int) -> bool:        # théorème de Hurwitz
    ...  # retourne dim in (1, 2, 4, 8)
```
**Test** : `test_hypercomplex.py` — (i) `Quaternion(0,1,0,0)**2 == Quaternion(-1,0,0,0)` ; (ii) rotation d'un vecteur par q puis q⁻¹ = vecteur initial ; (iii) `is_division_algebra(16) == False` ; (iv) `cayley_dickson((1,1),(1,1))` dimension double ; (v) non-associativité des octonions `(e1*e2)*e3 != e1*(e2*e3)`.
**Conformité** : primitives mathématiques pures, aucun entraînement requis, zéro impact sur `SpectralCoreBlock`.

#### 2. `bpe_tokenizer.py` — Tokenizer BPE branchable (M16)
**Exigence** : EX-B256 (génération contrainte par dictionnaire), M16 (tokenizer BPE/SentencePiece branchable). Le `text_decoder.py` actuel est char-level max_len=8 → **plafonne la génération de phrases cohérentes** (EX-B198, AT-B8 "générations de textes courts et créatifs").
**API** :
```python
class BPETokenizer:
    def __init__(self, vocab_size: int = 4096): ...
    def train(self, corpus: list[str], verbose: bool = False) -> None:  # merges glouton
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def vocab_size(self) -> int: ...
    def save(self, path: str) / load(cls, path): ...

def default_corpus() -> list[str]:  # phrases synthétiques + mots appris du LearnedVocab
    ...
```
**Test** : `test_bpe_tokenizer.py` — (i) `encode("hello")` puis `decode` == "hello" (round-trip) ; (ii) `vocab_size` croît avec `train` ; (iii) token fréquent "ing" < token rare ; (iv) entraînement sur 1000 phrases synthétiques → OOV géré (subword) ; (v) intégration : brancher sur `CharGenerator` pour générer > 8 tokens.
**Conformité** : ne touche pas au core ; devient la **tête d'entrée texte** optionnelle de l'`OmniModel`. Données synthétiques (générateur de phrases à partir du `LearnedVocab` + règles `morphology_fr`).

#### 3. `syntax_parser.py` — Analyseur syntaxique S-V-C (C5)
**Exigence** : EX-T8 (syntaxe), EX-B192 (Sujet/Verbe/Compléments/adverbe/adjectifs, liens entre mots = sens), EX-B198 (règles pour créer des phrases correctes). `phrase.py` actuel est composition d'IDs, **pas un parser grammatical**.
**API** :
```python
@dataclass
class Token:
    word: str; pos: str  # NOUN, VERB, ADJ, DET, PREP, ...

@dataclass
class Dependency:
    head: int; dep: int; relation: str  # nsubj, dobj, amod, ...

class SyntaxParser:
    def __init__(self, lexicon: dict | None = None): ...  # lexique FR+EN minimal
    def tokenize(self, sentence: str) -> list[Token]: ...
    def tag(self, tokens: list[Token]) -> list[Token]: ...   # POS tagging règle-based
    def parse(self, sentence: str) -> tuple[list[Token], list[Dependency]]: ...
    def is_grammatical(self, sentence: str) -> bool: ...     # S-V-C valide
    def extract_subject_verb_object(self, sentence: str) -> tuple: ...
```
**Test** : `test_syntax_parser.py` — (i) "Le chat mange une souris" → (sujet=chat, verbe=mange, objet=souris) ; (ii) "mange souris" → `is_grammatical == False` (sujet absent) ; (iii) dépendance nsubj correcte ; (iv) phrase négative "ne ... pas" détectée ; (v) généralisation sur phrases jamais vues via lexique.
**Conformité** : règles + lexique, pas de corpus. Débloque EX-B181 (comment encoder les règles de grammaire).

#### 4. `mcts_planner.py` — Planification MCTS latent (MCTS)
**Exigence** : EX-M "Latent MCTS" (Besoins_Maths), EX-B205 (orchestration, planification, exécution, test, validation), test-time compute avec backtracking (EX-M20 : backtracking immédiat sur violation).
**API** :
```python
@dataclass
class MCTSNode:
    state: torch.Tensor        # AMV-256 latent
    parent: MCTSNode | None
    children: list
    visits: int; value: float

class MCTSPlanner:
    def __init__(self, core: SpectralCoreBlock, verifier, n_simulations: int = 100,
                 c_puct: float = 1.4): ...
    def select(self, node: MCTSNode) -> MCTSNode: ...    # UCB
    def expand(self, node: MCTSNode) -> None: ...        # via core.forward
    def simulate(self, node: MCTSNode) -> float: ...     # rollout + verifier gate
    def backprop(self, node: MCTSNode, value: float) -> None: ...
    def search(self, root_state: torch.Tensor, depth: int = 5) -> MCTSNode: ...
    def best_plan(self, root_state) -> list[torch.Tensor]: ...
```
**Test** : `test_mcts_planner.py` — (i) sur tâche 2+2=4, le plan trouvé passe le verifier à chaque étape ; (ii) backtracking déclenché sur étape illégale (vérifier `[ANOMALIE_CAUSALE]`) ; (iii) `n_simulations=100` > `n_simulations=10` en qualité ; (iv) profondeur 3 atteignable ; (v) UCB équilibre exploration/exploitation (visites distribuées).
**Conformité** : réutilise `SpectralCoreBlock` comme modèle de transition et `verifier.py` comme Verification Gate (EX-M17, EX-M19). C'est l'instanciation du **test-time compute** du paradigme.

#### 5. `continual_learning.py` — EWC / Synaptic Intelligence (M19)
**Exigence** : EX-B187 (apprentissage continu sans catastrophic forgetting), EX-M28-31 (anti-pattern bottleneck collapse, solution LoRA), M19. `sleep.py` consolide mais ne protège pas les poids.
**API** :
```python
class EWC:
    """Elastic Weight Consolidation (Kirkpatrick 2017)."""
    def __init__(self, model: nn.Module, lam: float = 400.0): ...
    def compute_fisher(self, dataloader, n_samples: int = 100) -> None: ...  # diagonale Fisher
    def penalty(self, model) -> torch.Tensor: ...   # sum_i F_i * (theta_i - theta*_i)^2

class SynapticIntelligence:
    """Zenke 2017 — importance cumulative en ligne."""
    def __init__(self, model, lam: float = 0.01): ...
    def update(self, model) -> None: ...   # après chaque batch
    def penalty(self, model) -> torch.Tensor: ...

def continual_train(model, tasks: list, method: str = "ewc") -> dict:
    """Entraîne séquentiellement sur tasks, retourne acc par tâche avant/après."""
```
**Test** : `test_continual_learning.py` — (i) sans EWC, tâche A oubliée après tâche B (acc A chute) ; (ii) **avec EWC, acc A préservée** (gap < 5%) ; (iii) Fisher diagonal positif ; (iv) trade-off lam (trop grand = fige, trop petit = oublie) ; (v) SI en ligne ≤ overhead 10%.
**Conformité** : wrapper de loss externe, ne modifie pas `SpectralCoreBlock`. Débloque le scénario "capital de France" (EX-T62) et la consolidation honnête.

---

### HAUTE

#### 6. `phonology.py` — Phonologie + IPA (L5)
**Exigence** : EX-T6 (phonologie, morphologie, étymologie), L5. Actuellement `morphology.py`/`morphology_fr.py` font la morphologie, **pas la phonétique**.
**API** :
```python
class Phonology:
    def __init__(self): ...
    def to_ipa(self, word: str, lang: str = "fr") -> str: ...     # règles FR/EN
    def syllables(self, word: str) -> list[str]: ...
    def rhyme(self, word_a: str, word_b: str) -> bool: ...        # rime syllabique
    def stress_pattern(self, sentence: str) -> list[str]: ...      # accent tonique
    def phonemes(self, word: str) -> list[str]: ...
```
**Test** : `test_phonology.py` — (i) `to_ipa("chat")` contient "ʃ" ; (ii) `syllables("bonjour") == ["bon", "jour"]` ; (iii) `rhyme("matin", "destin") == True` ; (iv) `phonemes` count > 0 ; (v) règle générale sur mot inconnu (generalization).

#### 7. `etymology.py` — Étymologie / morphèmes (L7)
**Exigence** : EX-T7 (affixes, morphèmes, lexèmes, radicaux), EX-B197, L7.
**API** :
```python
class Etymology:
    def __init__(self): ...
    def decompose(self, word: str) -> dict:   # {prefix, root, suffix, infix}
        ...
    def root(self, word: str) -> str: ...     # lexème
    def family(self, word: str) -> list[str]: ...  # mots même racine
    def cognates(self, word: str, lang_a: str, lang_b: str) -> list[str]: ...
```
**Test** : `test_etymology.py` — (i) `decompose("unhappiness") == {prefix:"un", root:"happi", suffix:"ness"}` ; (ii) `root("répétition")` ≈ "pet"/"peat" ; (iii) `family("bonheur")` inclut "bon", "bonnement" ; (iv) cognates FR/EN "nuit"/"night".

#### 8. `semantic_traits.py` — Sémantique fine / sèmes (L11)
**Exigence** : EX-T8 (sémantique), EX-B196 (règles sémantiques), L11 (polysémie vs homonymie). `phrase.py` fait similarité cosinus, **pas de traits sémantiques**.
**API** :
```python
class SemanticTraits:
    def __init__(self): ...
    def traits(self, word: str) -> dict: ...      # {animé:+, concret:+, ...}
    def polysemy(self, word: str) -> list[str]: ...   # sens multiples
    def is_homonym(self, w1: str, w2: str) -> bool: ...
    def synonym(self, word: str, context: str) -> str: ...  # dépend du contexte
    def antonym(self, word: str) -> str: ...
```
**Test** : `test_semantic_traits.py` — (i) `traits("chien")["animé"] == True` ; (ii) `polyseme("avocat")` ≥ 2 sens (fruit / métier) ; (iii) `is_homonym("port", "port")` (de mer / de manteau) ; (iv) synonyme contextuel différencié.

#### 9. `collocations.py` — Mots composés / locutions (L13)
**Exigence** : EX-B197 (dictionnaires avec cas d'utilisation), L13.
**API** :
```python
class Collocations:
    def is_compound(self, phrase: str) -> bool: ...     # "pomme de terre"
    def split_compound(self, phrase: str) -> list[str]: ...
    def fixed_expression(self, phrase: str) -> bool: ... # "au fur et à mesure"
    def collocate(self, word: str) -> list[str]: ...     # "fort" ↔ "café fort"
```
**Test** : `test_collocations.py` — (i) `is_compound("pomme de terre") == True` ; (ii) `split_compound` correct ; (iii) locution figée détectée ; (iv) collocation cohérente vs aléatoire.

#### 10. `rl_posttraining.py` — DPO / GRPO (M20)
**Exigence** : EX-B213 (auto-amélioration), M20. DPO (Rafailov 2023), GRPO (DeepSeek 2024).
**API** :
```python
def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
             ref_chosen: torch.Tensor, ref_rejected: torch.Tensor,
             beta: float = 0.1) -> torch.Tensor: ...

def grpo_loss(policy_rewards: list[float], group_size: int = 8,
              beta: float = 0.04) -> torch.Tensor: ...

class DPOTrainer:
    def __init__(self, model, ref_model, beta: float = 0.1): ...
    def train_step(self, batch) -> dict: ...

def preference_dataset_synthetic(n: int = 200) -> list:  # paires (chosen, rejected)
    ...
```
**Test** : `test_rl_posttraining.py` — (i) DPO loss décroît sur batch synthétique ; (ii) policy diverge de ref après entraînement ; (iii) GRPO Advantage centré ; (iv) beta contrôle écart au ref.

#### 11. `curiosity.py` — Exploration active (M21)
**Exigence** : EX-B173 (conscience/artificial curiosity), EX-B185 (meta-learning), M21.
**API** :
```python
class CuriosityDrive:
    """Récompense intrinsèque = erreur de prédiction (Pathak ICM 2017)."""
    def __init__(self, forward_model: nn.Module, inverse_model: nn.Module | None = None): ...
    def intrinsic_reward(self, state, action, next_state) -> float: ...  # ||f(s,a) - s'||
    def novelty(self, state) -> float: ...   # distance au plus proche voisin en mémoire
    def should_explore(self, state, threshold: float = 0.5) -> bool: ...

class RandomNetworkDistillation:  # Burda 2018
    def __init__(self, target: nn.Module, predictor: nn.Module): ...
    def reward(self, state) -> float: ...   # ||predictor(s) - target(s)||
```
**Test** : `test_curiosity.py` — (i) état nouveau → reward élevé ; (ii) état vu → reward bas (habituation) ; (iii) RND predictor converge → reward décroît ; (iv) `should_explore` True sur OOD.

#### 12. `calibration.py` — Calibration Brier + conscience épistémique (H18)
**Exigence** : EX-B186 (auto-évaluation et conscience des lacunes), EX-B202 (conscience de ce qu'il sait/ne sait pas), H18. `self_correction.py` fait re-raisonnement, **pas de calibration proper**.
**API** :
```python
def brier_score(probs: list[float], outcomes: list[int]) -> float: ...  # lower=better
def expected_calibration_error(probs, outcomes, n_bins: int = 10) -> float: ...
def reliability_diagram(probs, outcomes) -> dict: ...

class EpistemicMonitor:
    """Surveille la confiance vs l'accuracy réelle du modèle."""
    def __init__(self): ...
    def update(self, prediction, confidence, ground_truth): ...
    def knows_that_it_knows(self) -> float: ...   # P(correct | confiant)
    def knows_that_it_doesnt_know(self) -> float: ...  # P(incorrect | peu confiant)
    def should_abstain(self, confidence: float) -> bool: ...  # seuil optimal
```
**Test** : `test_calibration.py` — (i) Brier parfait (probs=outcomes) → 0 ; (ii) ECE modèle sur-confiant > ECE calibré ; (iii) `knows_that_it_knows` élevé après entraînement ; (iv) `should_abstain` True sous seuil ; (v) tie avec `KnowledgeBase` abstention existante.

#### 13. `explainer.py` — Explication structurée (M9)
**Exigence** : EX-B191 (le modèle doit réfléchir, raisonner, comprendre réellement), M9.
**API** :
```python
@dataclass
class Explanation:
    claim: str
    premises: list[str]      # étapes intermédiaires
    evidence: list[str]      # citations / sources
    confidence: float
    counter_example: str | None = None

class Explainer:
    def __init__(self, verifier, knowledge_base): ...
    def explain(self, question: str, answer: str) -> Explanation: ...
    def why(self, question: str, answer: str) -> list[str]: ...   # chaîne causale
    def why_not(self, question: str, alternative: str) -> str: ... # contre-argument
    def decompose(self, problem: str) -> list[str]: ...           # macro → micro
```
**Test** : `test_explainer.py` — (i) `explain("2+2", "4")` ≥ 1 prémisse ; (ii) `why` produit chaîne vérifiable ; (iii) `why_not` identifie erreur ; (iv) `decompose` produit sous-problèmes (EX-B : décomposition macro→micro).

#### 14. `multihop_rag.py` — RAG multi-hop + re-ranking (M11)
**Exigence** : M11. `document_learner.py` fait single-hop retrieval.
**API** :
```python
class MultiHopRAG:
    def __init__(self, knowledge_base, max_hops: int = 3): ...
    def retrieve(self, query: str, k: int = 5) -> list: ...   # hop 1
    def expand_query(self, query: str, prev_docs) -> list[str]: ...  # hop 2+
    def rerank(self, query: str, docs: list) -> list: ...  # cross-encoder score
    def answer_with_citations(self, query: str) -> tuple[str, list[str]]: ...
    def needs_more_hops(self, query: str, confidence: float) -> bool: ...
```
**Test** : `test_multihop_rag.py` — (i) question multi-hop "Qui a fondé la ville natale d'Einstein ?" → 2+ hops ; (ii) re-ranking améliore précision vs retrieval brut ; (iii) citations présentes ; (iv) `needs_more_hops` True si confiance faible.

#### 15. `graph_algorithms.py` — Dijkstra / A* / BFS / DFS (G-ALGO)
**Exigence** : EX-B297 (Dijkstra), EX-B298 (A*), EX-B (algorithmique). Manquants alors que EX-B les cite nommément.
**API** :
```python
class WeightedGraph:
    def __init__(self): self.adj = {}
    def add_edge(self, u, v, w): ...
    def neighbors(self, u): ...

def dijkstra(graph, source) -> dict: ...       # distances minimales
def a_star(graph, source, goal, heuristic) -> list: ...
def bfs(graph, source) -> list: ...
def dfs(graph, source) -> list: ...
def shortest_path(graph, source, goal) -> list: ...
```
**Test** : `test_graph_algorithms.py` — (i) Dijkstra sur graphe jouet = distance attendue ; (ii) A* trouve chemin optimal avec heuristique admissible ; (iii) BFS ordre correct ; (iv) graphe déconnecté géré ; (v) complexité mesurée O((V+E) log V).

---

### MOYENNE

#### 16. `signal_processing.py` — Fourier pédagogique + PID + spectre (SIG)
**Exigence** : EX-B289 (résidus), EX-B292 (analyse vibratoire), EX-B293 (Fourier), EX-B294-295 (signaux carré/triangulaire, spectre), EX-B296 (PID). Le `SpectralCoreBlock` utilise FFT en interne mais ne l'**expose pas comme primitive apprise**.
**API** :
```python
def dft(signal: list[complex]) -> list[complex]: ...   # DFT naïf O(N²)
def fft(signal: list[complex]) -> list[complex]: ...    # FFT Cooley-Tukey O(N log N)
def spectrum(signal, sample_rate) -> tuple[list, list]: ...  # freqs, mags
def square_wave(t, freq) -> float: ...
def triangular_wave(t, freq) -> float: ...

class PIDController:
    def __init__(self, kp, ki, kd): ...
    def update(self, setpoint, measured, dt) -> float: ...
    def reset(self): ...

def residue_theorem_example() -> dict:  # intégrale complexe pédagogique
    ...
```
**Test** : `test_signal_processing.py` — (i) `dft` == `fft` (à tolérance) ; (ii) Parseval vérifié ; (iii) spectre sinusoïde = pic à la fréquence ; (iv) PID converge vers setpoint ; (v) `fft` vs `np.fft` concordent.

#### 17. `jacobian.py` — Jacobien + changement de variables (JAC)
**Exigence** : EX-B289 (théorème des résidus), EX-B290 (diagonalisation), EX-B299-301 (Jacobien, ratio aires, changement de variables).
**API** :
```python
def jacobian(f: callable, x: list[float], eps: float = 1e-6) -> list[list[float]]: ...  # différences finies
def jacobian_symbolic(exprs: list, vars: list) -> list: ...  # via SymPy
def determinant(matrix) -> float: ...
def change_of_variables_integral(f, transform, bounds) -> float: ...  # × |det J|
def diagonalize(matrix) -> tuple: ...   # valeurs/vecteurs propres (SymPy)
```
**Test** : `test_jacobian.py` — (i) Jacobien identité = matrice identité ; (ii) `det(J)` cohérent avec aire transformée ; (iii) changement de variables polaire vérifié ; (iv) diagonalisation matrice symétrique réelle = valeurs propres réelles.

#### 18. `sarcasm.py` — Sarcasme / ironie (M3)
**Exigence** : M3.
**API** :
```python
class SarcasmDetector:
    def __init__(self): ...   # features : contradiction littéral/contexte, ponctuation, intensité
    def detect(self, text: str, context: str = "") -> dict:  # {score, markers}
        ...
    def is_ironic(self, text: str) -> bool: ...
```
**Test** : `test_sarcasm.py` — (i) "Oh génial, encore une réunion" → score élevé (contexte négatif + littéral positif) ; (ii) littéral "génial" sans contexte → score bas ; (iii) marqueurs détectés ("oh", "encore", ponctuation).

#### 19. `world_model.py` — World model neuronal JEPA (M12)
**Exigence** : M12, EX-B205. `world.py` est procédural (NPC scripts), pas un **modèle prédictif neuronal**.
**API** :
```python
class JEPAPredictor(nn.Module):
    """Joint Embedding Predictive Architecture (LeCun 2022) — prédit embedding t+1."""
    def __init__(self, d_model: int = 256): ...
    def forward(self, state_t, action) -> torch.Tensor: ...   # prédit state_{t+1} dans latent
    def loss(self, state_t, action, state_t1) -> torch.Tensor: ...  # cosinus

class WorldModel:
    def __init__(self, predictor: JEPAPredictor): ...
    def imagine(self, state, actions: list) -> list[torch.Tensor]: ...  # rollouts
    def plan(self, goal, state, horizon: int = 5) -> list: ...
```
**Test** : `test_world_model.py` — (i) loss décroît sur trajectoires synthétiques ; (ii) `imagine` horizon 3 stable ; (iii) prédiction t+1 corrélée à ground truth ; (iv) plan atteint le but sur tâche jouet.

#### 20. `osint_scanner.py` — OSINT scanner (OSINT)
**Exigence** : EX-T12 (OSINT + Pentest expert), EX-B199 (OSINT), EX-B261 (SharadKumar97/OSINT-SPY à intégrer).
**API** :
```python
class OSINTScanner:
    def __init__(self, safe_mode: bool = True): ...  # safe_mode = pas de requêtes réseau réelles hors allowlist
    def scan_domain(self, domain: str) -> dict: ...   # DNS, whois (cache), registres publics
    def find_leaks(self, email: str) -> list: ...     # HIBP-style (cache local)
    def extract_metadata(self, text: str) -> dict: ...  # emails, phones, IBAN, hashes
    def report(self, target: str) -> dict: ...
```
**Test** : `test_osint_scanner.py` — (i) extraction emails/IBAN/téléphones d'un texte ; (ii) DNS cache OK ; (iii) safe_mode bloque domaine non allowlisté ; (iv) `report` structuré JSON.

---

## 4. Actions transverses (à mener en parallèle du top 20)

### 4.1. CRITIQUE — Corriger les tautologies du `domain_trainer.py`
Les fonctions `evaluate_rule`, `evaluate_all_domains`, `reasoning_bench_aime` comparent `apply == apply` (audit vague 1 §2.3). Elles gonflent artificiellement le BENCH_LEVEL.
- **Action** : réimplémenter `evaluate_all_domains` via `neural_multihop.neural_holdout_eval` (le core neural, pas `lambda`). Re-tester sur hold-out jamais vu.
- **Test de non-régression** : hold-out 97-100% maintenu ; si chute, le score précédent était cosmétique.
- **Impact honnêteté** : BENCH_LEVEL pourrait baisser de 94.9 à ~85 — c'est **souhaitable** (crédibilité).

### 4.2. CRITIQUE — Brancher `chemistry.py`, `genetics.py`, `finance.py` avec tests dédiés
Ces 3 modules existent et sont exportés mais **n'ont aucun test** (`test_chemistry.py` etc. absents).
- **Action** : créer `test_chemistry.py` (balance H₂O, molar_mass NaCl), `test_genetics.py` (Punnett Aa×Aa = 1:2:1), `test_finance.py` (compound_interest vérifié analytiquement).
- **Priorité** : HAUTE (un module exporté sans test est un gap de crédibilité).

### 4.3. HAUTE — Couronne compositionnelle multi-opérateurs
Le crown-jewel actuel (C1/C2/C10) prouve add/mul/linop séparément. Le cahier des charges (EX-M33, EX-M34) exige la **généralisation par composition inter-opérateurs**.
- **Action** : étendre `neural_multihop.py` avec `neural_inter_op_eval` : grok add, puis mul, puis tester `(a+b)*c` jamais vu. Crown-jewel +99.5 pt étendu.
- **Test** : hold-out inter-op ≥ 90%.

### 4.4. MOYENNE — Bench PUBLIC reproductible (M25)
Le bench actuel est interne. M25 exige un bench **public** (avec seeds, requirements figés, README).
- **Action** : `bench_public.py` + `bench_public_README.md` documentant comment reproduire BENCH_LEVEL sur machine vierge.

---

## 5. Honnêteté : ce qui reste bloqué par corpus (exclu de la vague 2)

Ces 6 gaps NE PEUVENT PAS être résolus par densification seule. Tout "solution" sans télécharger les datasets reste cosmétique (cf. CAPABILITIES.md §"Limites honnêtes").

| Gap | Corpus requis | État maximal atteignable sans corpus |
|---|---|---|
| H4 Vidéo | VideoMME / Kinetics | Moving-MNIST (maquette) — déjà fait |
| H5 Parole | LibriSpeech / CommonVoice | VAD RMS + stub formant — déjà fait |
| H6 Object detection | COCO / OpenImages | Centre+rayon conceptuel — déjà fait |
| H7 OCR | IAM / ICDAR | Non implémentable honnêtement |
| H8 Radar | SENTINEL-1 | Energy detection conceptuel — déjà fait |
| H3 Code échelle | GitHub patches | Template+exec 12/12 — déjà fait |

---

## 6. Ordre de bataille suggéré (sprints)

**Sprint 1 (CRITIQUE, ~5 capacités)** : `hypercomplex.py` → `bpe_tokenizer.py` → `syntax_parser.py` → `mcts_planner.py` → `continual_learning.py`. + action transverse tautologies.

**Sprint 2 (HAUTE, ~6 capacités)** : `phonology.py` → `etymology.py` → `semantic_traits.py` → `collocations.py` → `rl_posttraining.py` → `curiosity.py`.

**Sprint 3 (HAUTE fin, ~4 capacités)** : `calibration.py` → `explainer.py` → `multihop_rag.py` → `graph_algorithms.py`.

**Sprint 4 (MOYENNE, ~5 capacités)** : `signal_processing.py` → `jacobian.py` → `sarcasm.py` → `world_model.py` → `osint_scanner.py`.

**Sprint 5 (transverse)** : tests chemistry/genetics/finance + crown inter-op + bench public.

**Total vague 2** : 20 capacités + 4 actions transverses = ~24 livrables. Objectif capacités RÉELLES : ~80 (vague 1) → ~140-160 (vague 2).

---

## 7. Références cahier des charges couvertes par la vague 2

- EX-T6 (phonologie) → `phonology.py` ✓
- EX-T7 (étymologie) → `etymology.py` ✓
- EX-T8 (syntaxe) → `syntax_parser.py` + `semantic_traits.py` + `collocations.py` ✓
- EX-T12 (OSINT) → `osint_scanner.py` ✓
- EX-B173/185 (curiosité, meta-learning) → `curiosity.py` ✓
- EX-B187 (continual learning) → `continual_learning.py` ✓
- EX-B191 (comprendre) → `explainer.py` ✓
- EX-B192/198 (grammaire) → `syntax_parser.py` ✓
- EX-B202/186 (conscience des lacunes) → `calibration.py` ✓
- EX-B205 (planification) → `mcts_planner.py` + `world_model.py` ✓
- EX-B213 (auto-amélioration) → `rl_posttraining.py` ✓
- EX-B256 (tokenizer) → `bpe_tokenizer.py` ✓
- EX-B284-288 (quaternions et algèbres) → `hypercomplex.py` ✓
- EX-B289-301 (résidus, Jacobien, diagonalisation) → `jacobian.py` + `signal_processing.py` ✓
- EX-B292-296 (Fourier, PID, spectre) → `signal_processing.py` ✓
- EX-B297-298 (Dijkstra, A*) → `graph_algorithms.py` ✓
- C5 (grammaire) → `syntax_parser.py` ✓
- H18 (calibration) → `calibration.py` ✓
- M3 (sarcasme) → `sarcasm.py` ✓
- M9 (explication) → `explainer.py` ✓
- M11 (RAG multi-hop) → `multihop_rag.py` ✓
- M12 (world model) → `world_model.py` ✓
- M16 (BPE) → `bpe_tokenizer.py` ✓
- M19 (EWC) → `continual_learning.py` ✓
- M20 (DPO/GRPO) → `rl_posttraining.py` ✓
- M21 (curiosité) → `curiosity.py` ✓

**Couverture cahier des charges après vague 2** : ~95% des exigences implémentables sans corpus (les 5% restants = les 6 modalités corpus + frontend cockpit qui nécessite UI séparée).

---

## 8. Critères de "RÉEL" (anti-tautologie, audit vague 1 §2.1)

Chaque capacité de la vague 2 doit satisfaire :
1. **Poids entraînés** (ou règles vérifiables pour les modules symboliques) — pas de string hardcodée.
2. **Métrique numérique** (acc, loss, Brier, ECE...) sur jeu de test.
3. **Hold-out** jamais vu (généralisation, pas mémorisation).
4. **Comparaison à baseline** (random, sans la méthode) — doit battre.
5. **Test pytest dédié** dans `test_<module>.py`, export dans `__init__.py`, entrée dans `STATUS.md`.

Tout module ne respectant pas ces 5 critères sera marqué **STUB** et ne comptera pas dans le BENCH_LEVEL.

---

**Fin du rapport.** Ce document guide la vague 2 : 20 capacités + 4 actions transverses, toutes implémentables sans corpus externe, conformes à `SpectralCoreBlock` et au paradigme primitives → grok → composer → généraliser.
