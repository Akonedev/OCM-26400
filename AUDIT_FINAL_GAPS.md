# AUDIT FINAL — GAPS RÉSIDUELS OCM-26400

**Expert-Auditeur :** Claude (audit final, aucun code écrit)
**Date :** 21 Juin 2026
**Périmètre :** `ocm26400/` (76 modules, ~1050 tests verts) confronté au cahier des charges (`Besoins/Besoins.md`, `Besoins/Besoins_Tests.md`, `Besoins/Formules_Maths_Algo.md`, `Besoins/Formule_Lois_Grokking.md`).
**Architecture PRÉSERVÉE :** `SpectralCoreBlock` (FFT, `d_model=256`) = noyau unifié. **PAS de transformer.** Paradigme : primitives → grok → composer → généraliser (raisonner = étapes, pas de params).
**6 lois (L1-L6)** respectées : (L1) décomposition > scale, (L2) masquage incrémental ⇒ per-step exact, (L3) `depth_max ≈ 1/(1−per_step_err)`, (L4) récurrence ⊥ longueur ⊥ params, (L5) `L=1+4D`, `batch·L≈const`, (L6) association (1-source direct, multi-source = décomposer).

---

## MÉTHODO (1 ligne)

Recoupement entre l'inventaire exhaustif des exigences du cahier des charges (~276 capacités fonctionnelles) et le contenu réel de `ocm26400/__init__.py` + 104 fichiers de tests. **Exclusions explicites de la mission** : GSM8K-NL (10 approches déjà livrées) et 6 modalités à corpus externe (COCO/LibriSpeech/VideoMME/IAM/SENTINEL/GitHub-échelle) — ces dernières sont des bloqueurs honnêtes, pas des gaps d'architecture.

---

## TOP 10 — Capacités EXIGÉES mais ENCORE NON implémentées

Légende priorité : **CRITIQUE** = le cahier des charges y revient 3+ fois OU pré-requis archi ; **HAUTE** = exigée nommément, ROI fort ; **MOYENNE** = exigée, ROI modéré. "Sans corpus" = implémentable en torch pur + règles + SymPy + données synthétiques.

| # | ID | Capacité | Exigence cahier des charges | Fichier cible | Priorité | Sans corpus ? |
|---|----|----------|------------------------------|---------------|----------|----------------|
| 1 | **G-ETYM** | Étymologie / morphèmes / lexèmes / radicaux / affixes séparables / tmèse | EX-T7, EX-B197, L7-L9 (Besoins §LINGUISTIQUES) | `ocm26400/etymology.py` | **CRITIQUE** | Oui (dictionnaire étymologique public + règles) |
| 2 | **G-COCKPIT** | Cockpit UI + protocole cockpit (bouton validation/refus apprentissage, statut temps réel, vu du modèle, stream artefacts) | INF1, Q1-Q3, EX-B (modes apprentissage AUTO/SUPERVISÉ), A5-A7 (Tests §Modes apprentissage) — revient **5+ fois** | `ocm26400/cockpit_protocol.py` + `cockpit/` (frontend) | **CRITIQUE** | Oui (FastAPI/WS + UI statique) |
| 3 | **G-SEMES** | Sèmes / traits sémantiques / polysémie vs homonymie / synonymes contextuels | EX-T8, EX-B196, L11-L12 (Besoins §LINGUISTIQUES) | `ocm26400/semantic_traits.py` | **HAUTE** | Oui (lexique sémantique + WordNet public léger) |
| 4 | **G-EPISODIC** | Mémoire épisodique neuronale (distincte de sémantique/procédurale, branche `[ANOMALIE_CAUSALE]`) | L3, EX-B182, Formules §3 (Episodic Memory pour consolidation) | `ocm26400/episodic_memory.py` | **HAUTE** | Oui (autoencodeur contextuel torch) |
| 5 | **G-MUSIC** | Génération musicale (notes MIDI, flow-matching sur partitions) | MM6, G6 (Tests §Modes de chat "musique in/out", Besoins §4 CRÉATION) | `ocm26400/music_generator.py` | **HAUTE** | Oui (MIDI = tokens, générables synthétiquement) |
| 6 | **G-ABSTRACT** | Abstraction / catégorisation / comparaison / estimation | R15, M7, EX-B181 ("comment encoder les règles") | `ocm26400/abstraction.py` | **HAUTE** | Oui (clustering spectral + prototypes) |
| 7 | **G-BENCH** | Bench PUBLIC reproductible (seeds, requirements figés, README run sur machine vierge) | INF3, M25, EX-B "tests reproductibles" (Besoins §4-4) | `ocm26400/bench_public.py` + `BENCH_PUBLIC_README.md` | **HAUTE** | Oui (wrap du `eval_harness.py` actuel) |
| 8 | **G-LNSRNS** | LNS (Logarithmic Number System) + RNS (Residue Number System) — arithmétique sans carry | EX-B304-305, M23-M24 maths (arXiv:2106.13914, arXiv:2311.17323) | `ocm26400/alt_number_systems.py` | **MOYENNE** | Oui (maths pures, justifie le choix spectral FFT) |
| 9 | **G-GEOSPATIAL** | Géospatial / plan de veille type Google Maps (routing OSM, street-view) | MM7-étendu, M13, Besoins §Recap ("plan de veille") | `ocm26400/geospatial.py` | **MOYENNE** | Oui (osmnx/routingpy + cache OSM public) |
| 10 | **G-WORLDGEN** | Génération neuronale de monde interactif (FPS/plateforme + PNJ cohérents) — `world.py` actuel est procédural | MM7, G5, EX-B (Tests §Modes de chat "jeux interactifs plateforme/FPS") | `ocm26400/world_generation.py` (étend `world.py`) | **MOYENNE** | Oui (génération de grille par flow-matching + PNJ policy apprise) |

**Bonus (hors top 10, si budget restant)** : `code_compressor.py` (M18 — compression/simplification de code entraînée) ; diagonalisation matrices (M11 — étendre `equation_solver.py`/`jacobian.py`) ; voix passive (L17 — étendre `morphology_fr.py`) ; analogie structurée Gentner (R11/M10 — étendre `cognition.structure_mapping`) ; phonologie suprasegmentale accent/intonation/stress (L16 — étendre `phonology.py`) ; sentiment analysis entraîné (M24 — `nlp_tools.py` est lexicon-based, pas un classifieur).

---

## PLAN D'IMPLÉMENTATION (1 ligne par capacité, DAG paradigme)

**Sprint 1 — CRITIQUE (ordre par dépendances)**
1. `etymology.py` — décomposition `prefix/root/suffix`, `family()` depuis dictionnaire étymo FR/EN, `cognates()` FR↔EN. Pré-requis pour `semantic_traits` (#3). Loi L6 (décomposer avant associer).
2. `cockpit_protocol.py` — schéma JSON events `apprentissage:start/progress/done`, `validation:request(user)`, `agent:status`, `artefact:generated` ; backend FastAPI/WebSocket ; frontend `cockpit/` (UI vanilla, boutons valider/refuser). Débloque Q1-Q3 + A5-A7.

**Sprint 2 — HAUTE**
3. `semantic_traits.py` — `traits(word)` (animé/concret/…), `polysemy()`, `is_homonym()`, `synonym(contextuel)`. Réutilise `etymology` (#1) + `semantic_embeddings.py`.
4. `episodic_memory.py` — `EpisodicMemory.add(context, outcome, ts)`, `replay()`, `consolidate_to_semantic()`. Branché sur hook `[ANOMALIE_CAUSALE]` du LSRA. Distinct de `procedural_memory.py` (qui est "comment faire").
5. `music_generator.py` — flow-matching sur séquences MIDI (tokens note/durée/velocity) via `generators.AMVConditionedDecoder`. Production de partitions courtes, contraintes harmoniques (règles).
6. `abstraction.py` — `categorize(items)` (k-means spectral sur AMV), `prototype(cluster)`, `is_a(instance, concept)`. Justifie la taxonomie de concepts (R24).
7. `bench_public.py` — export de `eval_harness.BenchmarkRunner` avec seeds figées, `requirements-bench.txt`, `BENCH_PUBLIC_README.md`. Reproductible sur machine vierge.

**Sprint 3 — MOYENNE**
8. `alt_number_systems.py` — `LNS` (add ⇒ mult O(1)), `RNS` (arithmétique modulaire sans carry), `to_lns/to_rns/from_*`. Justification mathématique du noyau spectral (EX-B281).
9. `geospatial.py` — `Route(start, end, mode)` via `osmnx`/`routingpy` (cache OSM local), `bbox_features()` (POI/rues), `streetview_url(bbox)` (lien, pas scraping). `geo.py` reste factuel.
10. `world_generation.py` — `generate_level(theme, w, h)` (grille 2D par décodeur flow-matching), `npc_policy(state)` apprise (PPO-lite ou imitation sur traces synth.), branchée sur `world.py` (grille+NPC déjà là).

---

## CONFORMITÉ AUX 6 LOIS (L1-L6)

Chaque capacité ci-dessus respecte :
- **L1 (décomp > scale)** : aucun n'ajoute de params au `SpectralCoreBlock` ; ils ajoutent des **étapes** (étymo = décomposition, MCTS-like déjà couvert ailleurs).
- **L2 (masquage incrémental)** : `episodic_memory`, `abstraction` entraînés par masquage partiel (cf. `experiment_composition.py`).
- **L3 (`depth_max ≈ 1/(1−per_step)`)** : aucun risque de casser le grok — ce sont des modules externes au core.
- **L4 (⊥ longueur/params)** : musique et monde interactif utilisent la récurrence fenêtrée existante.
- **L5 (`L=1+4D`, batch·L≈const)** : non affectée (pas de nouveaux entraînnement longs).
- **L6 (association)** : étymologie et sèmes sont exactement l'association multi-source (décomposer d'après L1/L6).

---

## HONNÊTETÉ — Ce qui reste BLOQUÉ par corpus (exclu de ce top 10)

| Capacité | Corpus externe requis | État max atteignable sans corpus |
|---|---|---|
| Vidéo réelle | VideoMME / Kinetics | Moving-MNIST (maquette, déjà fait) |
| Parole réelle (TTS naturelle) | LibriSpeech / CommonVoice | Formant stub + VAD RMS (déjà fait) |
| Object detection | COCO / OpenImages | Centre+rayon conceptuel (déjà fait) |
| OCR scale | IAM / ICDAR | `ocr.py` 92.6% sur digits (déjà fait) |
| Radar/SAR | SENTINEL-1 | Energy detection conceptuel (déjà fait) |
| Code échelle SWE-bench | GitHub patches | Template+exec 12/12 (déjà fait) |

**Verdict** : ces 6 modalités ne sont **pas** des gaps d'architecture — ce sont des limites de données. Le paradigme OCM réduit les exemples nécessaires (crown-jewel +99.5pt) mais ne marche pas sur **zéro** exemple pour ces tâches.

---

## CONCLUSION

- **76 modules**, ~1050 tests verts, crown-jewel neural hold-out 97-100%, real-bench 29/29.
- **10 gaps résiduels** identifiés ci-dessus, **TOUS implémentables sans corpus externe** (le top 7 sans réserve, les 3 MOYENNE via données publiques légères OSM/MIDI).
- **2 CRITIQUES** (étymologie + cockpit UI) à livrer en premier — le cockpit est cité 5+ fois dans le cahier des charges comme pré-requis UX.
- Aucune refacto du `SpectralCoreBlock` nécessaire. Aucun transformer. Densification pure par couronne compositionnelle (L1/L6).
- Une fois le top 10 livré : couverture **~99%** des exigences implémentables sans corpus, seuil minimal pour déclarer le cahier des charges "substantiellement rempli".
