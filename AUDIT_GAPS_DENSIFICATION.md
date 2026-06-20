# AUDIT GAPS & DENSIFICATION — OCM-26400

**Expert-Auditeur :** Claude (rapport d'audit, aucune écriture de code).
**Date :** 20 juin 2026.
**Périmètre :** `ocm26400/` confronté au cahier des charges (`Besoins/Besoins.md`, `Besoins/Besoins_Tests.md`, `Besoins/Besoins_Maths.md`).
**Architecture PRESERVÉE :** `SpectralCoreBlock` (FFT) est le noyau unifié. Aucune refacto. Paradigme : **primitives → grok → composer → généraliser** (raisonner = ajouter des étapes, pas des params ; d≈256).

---

## MÉTHODOLOGIE D'AUDIT (honnête, pas du marketing)

Trois classes de couverture distinguées pour ÉVITER le greenwashing :

| Tag | Définition | Compté comme |
|---|---|---|
| **RÉEL** | Le module entraîne des poids / exécute du vrai signal / appelle un vrai OS ou HTTP, et le test mesure une métrique (accuracy/MSE qui baisse/gap > 0). | Couvert. |
| **STUB** | Interface + callable + assert de forme (shape/exists/non-None). Aucun poids entraîné, aucune métrique de qualité. | Couvert en interface, **mais pas une capacité réelle**. |
| **TAUTOLOGIQUE** | Le test vérifie qu'une fonction déterministe est cohérente avec elle-même (`apply == apply`, `verify == output`). Ne prouve AUCUNE compétence du modèle. | NON couvert (le "100% maîtrise" affiché est trompeur). |

Le rapport STATUS revendique "94.9/100 BENCH_LEVEL" et "91/91 règles maîtrisées 100%" — cet audit montre qu'une partie de ces chiffres relève des classes STUB/TAUTOLOGIQUE. Les capacités **réellement démontrées** sont listées en §2 avec leur métrique.

---

## 1. CAPACITÉS EXIGÉES (extraites du cahier des charges)

Exhaustif. Numérotation préfixée par domaine (L=linguistique, M=maths, P=physique, etc.).

### 1.1. APPRENTISSAGE DES PRIMITIVES (Besoins.md §1 + Besoins_Tests.md)

| # | Capacité exigée | Source |
|---|---|---|
| L1 | Vocabulaire anglais complet (~1M de formes) | BT§1 |
| L2 | Vocabulaire français (~60K mots) | BT§1 |
| L3 | Grammaire complète (toutes les règles) | BT§1 |
| L4 | Conjugaison complète (tous verbes, tous temps, voix/modes/aspect/personnes) | BT§1 |
| L5 | Phonologie / phonèmes / IPA | BT§1 |
| L6 | Morphologie (dérivation, flexion, composition) | BT§1 |
| L7 | Étymologie (radical, affixe, morphème, lexème) | BT§1 |
| L8 | Voyelles/consonnes, diphtongues, quantité vocalique | BT§1 |
| L9 | Affixes (préfixe/suffixe/désinence, de classe/sémantiques/lexicaux/séparables/tmèse) | BT§1 |
| L10 | Syntaxe (sujet-verbe-complément, rôle syntaxique) | BT§1 |
| L11 | Sémantique (sens, nuances, polysémie, homonymie, traits sémantiques/sèmes) | BT§1 |
| L12 | Synonymes/antonymes/hyperonymes/hyponymes/méronymes/holonymes | BT§1 |
| L13 | Mots composés, mots-valises, collocations, locutions figées | BT§1 |
| L14 | Néologismes, emprunts | BT§1 |
| L15 | Registre, fréquence d'usage | Besoins§associations |
| L16 | Phonologie suprasegmentale (accent, intonation, stress) | BT§1 |
| L17 | Conjugaison : voix active vs passive | BT étendu |
| L18 | Conjugaison : modes (indicatif, subjonctif, conditionnel, impératif) | BT étendu |
| L19 | Conjugaison : aspects (perfectif vs imperfectif) | BT étendu |
| L20 | Genre et nombre (masc/fém/neutre, sg/pl) | BT étendu |
| L21 | Conjugaison FR 3 groupes + tous temps | BT étendu |
| L22 | Verbes irréguliers EN + FR | BT étendu |
| L23 | Consonant doubling (CVC → CVC+Cing) | BT étendu |
| L24 | Capture SIMULTANÉE de toutes les features en une passe | BT§20 |

### 1.2. DOMAINES DE CONNAISSANCE (Besoins_Tests.md §2 — ~30 domaines)

| # | Capacité exigée |
|---|---|
| D1 | Mathématiques (arithmétique, algèbre, géométrie, analyse, probas, stats) |
| D2 | Physique (mécanique, quantique, thermo, EM, ondes, optique, nucléaire, relat, fluides, particules, acoustique) |
| D3 | Chimie (réactions, catalyse, dissolution, équilibres) |
| D4 | Biologie (ADN, mutation, transcription, cellule) |
| D5 | Médecine (diagnostic, prescription) |
| D6 | Pharmacologie (dose, métabolisation) |
| D7 | Histoire |
| D8 | Géographie |
| D9 | Sociologie / Psychologie / Neuroscience |
| D10 | Littérature / Écriture créative |
| D11 | Musique (composition, théorie) |
| D12 | Économie / Finance |
| D13 | Robotique (DH, PID) |
| D14 | Mécanique classique |
| D15 | IoT (Arduino, Raspberry Pi, capteurs, protocoles) |
| D16 | Mobile dev (Android, iOS) |
| D17 | Mobile use (Android, iOS) |
| D18 | Radar/satellite (détection, exploitation) |
| D19 | Object detection (objets, personnes, plaques d'immatriculation) |
| D20 | Object following (suivi temporel) |
| D21 | Object detection par sondes (WiFi, radio, Bluetooth) |
| D22 | Développement : Python, TypeScript, React, Vue.js, JavaScript, Java, Kotlin, CSS |
| D23 | Architectures, DevOps (CI/CD, Docker, Kubernetes) |
| D24 | Sécurité (OWASP Top10, Pentest, CVE, SQL injection, XSS, CSRF, buffer overflow, priv-esc) |
| D25 | OSINT (recon, corrélation, vérification) |
| D26 | Computer use complet |
| D27 | Browser use complet |
| D28 | Protocoles réseau : OSI 7 couches + TCP/IP + HTTP/HTTPS/DNS/TLS/DHCP/OSPF/BGP/NAT/Pare-feu/ports |
| D29 | Astronomie / Géologie / Météorologie / Botanique / Dentisterie / Écologie / CS théorique |
| D30 | Capacité JEPA-like prédictive basée physique (sans Frankenstein) |

### 1.3. APPRENTISSAGE / AUTO-APPRENTISSAGE (Besoins.md §process + Besoins_Tests.md)

| # | Capacité exigée |
|---|---|
| A1 | Apprentissage à partir d'une source fournie (PDF, URL, image, vidéo, YouTube) |
| A2 | Apprentissage ciblé (une donnée précise d'une page, pas tout le site) |
| A3 | Comprendre avant d'apprendre (règles obligatoires) |
| A4 | Généraliser après compréhension |
| A5 | Mode apprentissage AUTO (sans accord user) |
| A6 | Mode apprentissage SUPERVISÉ (bouton validation/refus user) |
| A7 | Notification de statut d'apprentissage (en cours / terminé) |
| A8 | Recherche multi-source + vérification cohérence/véracité |
| A9 | Apprentissage à la volée (temps réel) |
| A10 | Apprentissage incrémental (anti catastrophic forgetting) |
| A11 | Few-shot / Zero-shot learning |
| A12 | Meta-learning (apprendre à apprendre) |
| A13 | Continual learning (sans reset) |
| A14 | Apprentissage actif (curiosité, demande de données) |
| A15 | Apprentissage par renforcement (RL post-training) |
| A16 | Sleep / consolidation (rétrospection, entropie, extraction règle, sommeil profond) |
| A17 | Détecter ce que le modèle NE sait PAS (conscience épistémique) |
| A18 | Auto-amélioration continue |
| A19 | Apprentissage depuis DB SQL/NoSQL, API REST, dépôt GitHub |
| A20 | Extraction de formules/axiomes/algorithmes/principes/paradigmes/innovations |

### 1.4. RAISONNEMENT (Besoins.md §process + Besoins§comprendre)

| # | Capacité exigée |
|---|---|
| R1 | Comprendre le prompt |
| R2 | Raisonner longuement (récurrence fenêtrée profonde) |
| R3 | Intelligence = compréhension, pas mémorisation |
| R4 | Chain-of-Thought étape par étape |
| R5 | Test-time compute (backtrack, self-critique) |
| R6 | Raisonnement causal |
| R7 | Raisonnement multi-hop |
| R8 | Résolution d'équation symbolique |
| R9 | Planification (décomposition but → sous-buts) |
| R10 | Déduction logique (syllogisme) |
| R11 | Analogie (transfert de structure) |
| R12 | Compression d'information |
| R13 | Détection d'anomalie |
| R14 | Probabilité / incertitude |
| R15 | Abstraction / catégorisation / comparaison / estimation |
| R16 | Théorie de l'esprit (Sally-Anne) |
| R17 | Sens commun |
| R18 | Sarcasme / ironie |
| R19 | Raisonnement spatial |
| R20 | Raisonnement moral / éthique |
| R21 | Raisonnement scientifique (hypothèse → test → conclusion) |
| R22 | Raisonnement par élimination |
| R23 | Imagination contrefactuelle |
| R24 | Structure hiérarchique (taxonomie de concepts) |
| R25 | Explication (pourquoi cette réponse) |

### 1.5. MODALITÉS (Omni-In / Omni-Out — Besoins_Tests.md + Besoins§Recap)

| # | Capacité exigée |
|---|---|
| MM1 | Texte in/out |
| MM2 | Audio in/out (STT/TTS/STS streaming temps réel, VAD) |
| MM3 | Image in/out (classification + génération) |
| MM4 | Vidéo in/out (cohérence temporelle) |
| MM5 | 3D in/out (voxels, maillages) |
| MM6 | Musique in/out |
| MM7 | Monde interactif in/out (géospatial, PNJ, FPS/plateforme) |
| MM8 | Génération 3D par flux (voxel) |
| MM9 | Combinaison multi-modale simultanée |
| MM10 | Alignement amodal (4+ modalités → même vecteur) |

### 1.6. CRÉATION D'ARTEFACTS (Besoins_Tests.md §4 + Besoins§5)

| # | Capacité exigée |
|---|---|
| G1 | Génération de texte (code, doc, explications) |
| G2 | Génération d'images |
| G3 | Génération de vidéos |
| G4 | Génération 3D |
| G5 | Génération de mondes interactifs |
| G6 | Génération musicale |
| G7 | Génération de slides / présentation |
| G8 | Génération PDF |
| G9 | Génération de schémas / diagrammes |
| G10 | Génération d'infographies |
| G11 | Génération de data-visualisation |
| G12 | Génération de code multi-langage (HTML/CSS/JS/Python/TS/React/Vue/Java/Kotlin) |

### 1.7. AGENTS / SWARM / SKILLS / OUTILS (Besoins.md §2-3)

| # | Capacité exigée |
|---|---|
| AG1 | Orchestrateur + sous-agents experts seniors |
| AG2 | Swarm d'agents parallèles (jusqu'à 1000+) |
| AG3 | Communication inter-agents (broadcast, dialogue multi-tours) |
| AG4 | Mémoire partagée cohérente entre agents |
| AG5 | Auto-critique (Devil's Advocate) + jury de validation |
| AG6 | Création de skill à la volée (si manquant) |
| AG7 | Skills experts production-grade (≥24, qualité check) |
| AG8 | Prompts experts par domaine (≥22) |
| AG9 | Tool policy apprise (généralise, pas câblée) |
| AG10 | Agents coordonnés, ne perdent jamais le fil |
| AG11 | Work autonomously long-terme |

### 1.8. MÉMOIRE / CONTEXTE (Besoins.md)

| # | Capacité exigée |
|---|---|
| ME1 | Mémoire épisodique |
| ME2 | Mémoire sémantique |
| ME3 | Mémoire procédurale |
| ME4 | Mémoire de travail / scratchpad |
| ME5 | Gestion du contexte (long-context) |
| ME6 | Consolidation épisodique → sémantique |

### 1.9. INFRA / COCKPIT / SÉCURITÉ

| # | Capacité exigée |
|---|---|
| INF1 | Cockpit UI (bouton validation/refus apprentissage, statut, vu du modèle) |
| INF2 | Streaming output (token par token) |
| INF3 | Bench reproductible, mesure du niveau d'intelligence |
| INF4 | Ports en `.env`, jamais dans le code |
| INF5 | Sécurité anti-injection (tous outils) |
| INF6 | Searxng / recherche locale (web) |
| INF7 | RAG (chunking, citations, abstention) |

**TOTAL exigé (comptage granulaire) : ~190 capacités « principales », ~530+ si on décline chaque sous-capacité (temps de conjugaison × groupe, chaque langage de code, chaque protocole OSI, chaque règle de domaine, etc.) — cohérent avec les « ~528 capacités non couvertes » évoquées.**

---

## 2. CAPACITÉS DÉJÀ COUVERTES (372 tests + STATUS)

### 2.1. Couverture RÉELLE (poids entraînés / vrai signal / métrique mesurée)

| # | Capacité | Fichier | Preuve mesurée |
|---|---|---|---|
| 1 | AMV-256 quad-partition (ent/prop/op/meta) | `amv.py` | roundtrip + partition test |
| 2 | ACSP loss (L_align + L_step + L_sparse + L_consist InfoNCE) | `acsp.py`, `infonce.py` | loss differentiable end-to-end |
| 3 | SpectralCoreBlock (FFT filter complexe appris) | `spectral_core.py` | shape + Parseval |
| 4 | ReasonerBlock + LSRA loop + gate calibrée | `reasoner.py` | AUROC 1.0 valides vs OOD |
| 5 | **Crown-jewel décomposition Z₁₁** | `experiment_composition.py` | **gap +99.5pt (one-shot 0.5% vs décomp 100%)** — preuve cardinale |
| 6 | Crown-jewel linguistique (préfixe+stem+suffixe) | `experiment_linguistic.py` | gap +100pt sur 12 triples neufs |
| 7 | Survie crown-jewel sous dictionnaire dense (P2) | `experiment_linguistic_dense.py` | +100pt one-hot→dense |
| 8 | Scaling V>64 (Z₁₂₀) | `experiment_vocab_scale.py` | grok règle 99.7% brut |
| 9 | Apprentissage par ACSP différentiable (Gumbel ST) | `diff_decode.py` | loss descend à 0 sur 1 batch |
| 10 | Vocabulaire anglais RÉEL 370K → 1M+ flexions | `experiment_real_vocab.py`, `experiment_vocab_1m.py` | retrieval@1 100%, OOV 0% |
| 11 | Vocabulaire bilingue RÉEL 591K (EN 315K + FR 276K) | `experiment_bilingual_vocab.py` | retrieval 100% |
| 12 | Alignement amodal sur vues linguistiques RÉELLES | `real_linguistic.py` | retrieval 62-79% paires riches |
| 13 | Encodeurs audio Mel/STFT, image patches, vidéo, Conv3D voxel | `multimodal_encoders.py` | pipelines réels (validés signaux synth.) |
| 14 | MNIST 8×8 classification réelle (sklearn) | `experiment_real_vision.py` | **acc 90.9%** |
| 15 | MNIST 8×8 génération flow-matching RÉELLE | `generators.py`, `experiment_mnist28.py` | MSE baisse après entraînement |
| 16 | VAD voix (RMS énergie, segmentation fin de tour) | `voice.py` | vrai traitement signal |
| 17 | Sleep/consolidation (extraction règle linéaire-modulaire) | `sleep.py` | règle extraite, compression ×27 |
| 18 | Self-correction (re-raisonnement rattrape erreurs) | `self_correction.py` | 79% → 100% |
| 19 | WebFetchTool HTTP RÉEL (urllib + SSRF) | `web_tools.py` | fetch Wikipedia réel |
| 20 | ShellTool computer-use RÉEL (subprocess, sans shell=True) | `computer_use.py` | exécute vraies commandes |
| 21 | GUITool computer-use (pyautogui + allowlist) | `computer_use.py` | interface réelle (nécessite display) |
| 22 | OmniModel UNIFIÉ (1 noyau spectral, joint loss) | `omni.py` | differentiable end-to-end |
| 23 | MCP adapter (expose outils natifs via MCP) | `mcp_adapter.py` | 8 tests adapter |
| 24 | Capstone primitives → génère op^k profondeur 8 | `experiment_omni_generate.py` | 100% sur chaînes neuves |
| 25 | Récurrence fenêtrée profonde (op^k depth 2-5) | `experiment_recursion.py` | 100% sur chaînes non vues |
| 26 | Multi-rule (add/mul/linop conjointes) | `omni_rules.py` | cross-rule >85% |
| 27 | Terminal-Bench style (ShellTool exécute 10 cmds) | `bench_runner.py` | 100% mesuré |
| 28 | MorphologyVerifier (past/gerund/third par op_id) | `morphology.py` | 1 block 3 temps |

### 2.2. Couverture STUB (interface + assert shape, pas de poids)

Listée honnêtement : ~250 tests sur 372 sont dans cette catégorie. Ils valident que l'INTERFACE existe et est appelable, mais **ne prouvent pas la compétence**. Le STATUS les compte comme "capacités couvertes" — cet audit les marque **partiellement couverts (interface seulement)**.

Sont dans cette catégorie : tous les tests "domaine_XXX" (231-256), la plupart des tests "raisonnement_XXX" (201-227), les tests "théorie_esprit / sens_commun / sarcasme / moral / scientifique" (293-298), génération musicale/PDF/slides/schema (308-313), les tests d'extraction (351-356), etc.

### 2.3. Couverture TAUTOLOGIQUE (à NE PAS compter)

Deux zones critiques où le "100%" affiché est artificiel :

**A) `domain_trainer.py::evaluate_rule`** — compare `rule.apply(*args)` à `rule.apply(*args)` (toujours égal) et `rule.verify(args, gold)` qui est défini comme `apply == output` (toujours vrai pour le vrai gold). Le "91/91 règles maîtrisées 100%" affiché dans STATUS ne teste **aucune compétence d'un modèle** : il vérifie que la fonction Python `lambda` est cohérente avec elle-même. **Ce score est à RETIRER du discours "modèle entraîné sur tous les domaines"**.

**B) `domain_trainer.py::reasoning_bench_aime`** — compose `add.apply(a,b)` puis `add.apply(s1,c)` etc., puis `add.verify` (toujours vrai). Le "100% AIME-style" est aussi tautologique : il applique la formule `(a+b) mod 11` et vérifie qu'elle est égale à elle-même. **Aucun modèle n'a raisonné.**

**C) Beaucoup de "rules" dans `rules.py` sont symboliques mod-11**, par ex. `"gravity" = lambda m1,m2: (m1*m2) % n` ou `"ohm" = lambda u,r: (u * pow(r,-1,n)) % n`. Ces règles sont des jouets qui ont la *forme* d'une loi physique mais pas sa *sémantique* (pas d'unités SI, pas d'ordre de grandeur, pas de contraintes physiques). Utiles pour démontrer le MÉCANISME de composition, mais **ne constituent pas une "connaissance" de la physique**.

### 2.4. Capacités que le STATUS revendique mais qui sont sur-évaluées

| Revendication STATUS | Réalité |
|---|---|
| "94.9/100 BENCH_LEVEL" | Calcul agrégé incluant les scores tautologiques + interfaces stub. À recaluler en excluant A/B/C ci-dessus. |
| "91/91 règles maîtrisées (100%) sur 30/30 domaines" | Tautologique (§2.3 A). |
| "Raisonnement AIME-style 100% sur 50 chaînes profondeur 3" | Tautologique (§2.3 B). |
| "QCM GPQA/HLE style 97.1%" | Mix réel (règles symboliques vérifiées) + QCM fabriqués sur mesure. Pas le dataset GPQA réel. |
| "30 domaines couverts" | 30 catégories existent mais les règles sont des `lambda` jouets mod-11. |

---

## 3. GAPS — capacités exigées MAIS non réellement couvertes

### 3.1. CRITIQUE (le cahier des charges en dépend directement ; impossible de déclarer "rempli" sans)

| Gap | Capacité manquante | État actuel | Type |
|---|---|---|---|
| **C1** | **Raisonnement AIME/RÉEL par le core neural** (pas la lambda) | Le "100% AIME" est tautologique. Aucun test ne montre le ReasonerBlock grokké résolvant une chaîne arithmétique de profondeur 3+ sur un hold-out. | IMPLÉMENTABLE (crown-jewel étendu). |
| **C2** | **Compétence de domaine RÉELLE** (pas `apply==apply`) | `evaluate_rule` tautologique. Il faut mesurer : (i) le ReasonerBlock predit `compose(a,b)` sur hold-out par domaine, (ii) verify REJETTE un faux. | IMPLÉMENTABLE (crown-jewel par domaine). |
| **C3** | **Règles physiques/mathÉS RÉELLES** (avec unités, sémantique) | Les 91 règles sont `(αa+βb) mod 11`. F=ma réel existe (1 règle) mais isolée. Pas de système d'unités, pas de dimensional analysis. | MIXTE : règles codables + corpus scolaire. |
| **C4** | **Conjugaison FR complète** (3 groupes, tous temps, modes, voix, aspects) | MorphologyVerifier couvre 3 temps EN. FR quasi-absent (test 74/103/134 existent mais stubs). | IMPLÉMENTABLE (dictionnaire verbes FR public). |
| **C5** | **Grammaire/Raisonnement linguistique RÉEL** (analyse syntaxique, dépendances) | PhraseComposer compose des IDs d'entiers. Pas de parser syntaxique. Pas de règles S-V-C apprises par le core. | MIXTE. |
| **C6** | **Génération de TEXTE libre** (phrases anglaises/françaises cohérentes) | AUCUN. Le `OmniModel` génère audio/image (flow-matching) mais pas de décodeur de texte entraîné. Besoins§4 exigence #1 d'artefact. | CRITIQUE — gap fonctionnel majeur. |
| **C7** | **Apprentissage depuis PDF** | `parse_pdf` existe (PyMuPDF) mais ne déclenche PAS d'apprentissage (n'écrit pas dans la KB). | IMPLÉMENTABLE (câbler parse_pdf → KB). |
| **C8** | **Cycle URL → KB → retrieval end-to-end RÉEL** | URLMemory stocke le texte brut, pas des faits exploitables. Pas de retrieval sémantique sur le texte appris (test 350 stub). | IMPLÉMENTABLE (chunking + AMV). |
| **C9** | **Décodeur de texte flow-matching** (génération de mots/phrases) | CompositionalVocabulary decode_word existe (IDs), pas un générateur entraîné. | IMPLÉMENTABLE (flow-matching sur tokens). |
| **C10** | **Multi-hop raisonnement par le core** (pas lambda) | Test 201 = liste hardcodée `[4,7,3,8]`. Le ReasonerBlock n'est pas testé sur multi-hop. | IMPLÉMENTABLE (lsra_solve étendu). |

### 3.2. HAUTE (capacités centrales du cahier des charges, actuellement stub)

| Gap | Capacité manquante | État actuel | Type |
|---|---|---|---|
| H1 | Browser use INTERACTIF (clics, JS, formulaires, login) | WebFetchTool = GET HTTP passif. Pas de Playwright/Selenium. | Intégration runtime (lib externe). |
| H2 | Computer use GUI ENTRAÎNÉ (pas juste pyautogui) | GUITool exécute pyautogui. Pas de policy apprise pour clic/scroll. | MIXTE. |
| H3 | Génération de code ENTRAÎNÉE multi-langage | Tests 140/275-284 = stubs (string return). Pas de LLM générant du code. | CORPUS (GitHub patches). |
| H4 | Vidéo classification/génération sur VRAIES vidéos | `experiment_real_multimodal.py` utilise signaux synth. Moving-MNIST = игрушка. | CORPUS (VideoMME/MMM-Pro). |
| H5 | Audio classification/génération sur VRAIE parole | AudioMel encodeur OK, mais TTS = formant stub. Pas de Whisper/Tacotron. | CORPUS (LibriSpeech). |
| H6 | Object detection (bounding boxes, mAP) | Test 137/158/198 = conceptuel (centre+rayon). Pas de YOLO-like. | CORPUS (COCO). |
| H7 | OCR (image → texte) | Non implémenté. | CORPUS (IAM/OCR). |
| H8 | Radar/satellite exploitation | Test 143/197 = conceptuel (energy detection). Pas de SAR processing. | CORPUS (SENTINEL). |
| H9 | Apprentissage YouTube (transcript → KB) | Test 189/195 = interface stub. Pas de vrai fetch YouTube. | IMPLÉMENTABLE (yt-dlp). |
| H10 | RAG avec chunking + embeddings + citations | `KnowledgeBase` retrieval cosinus sur mots-id. Pas de chunking texte, pas de citations source. | IMPLÉMENTABLE. |
| H11 | Few-shot / in-context learning par le core | Test 302 = hardcodé `ratios={2.0}`. Pas d'In-Context appris. | IMPLÉMENTABLE. |
| H12 | Mémoire procédurale (comment faire) | Test 211 = liste Python. Pas distinguée de sémantique. | IMPLÉMENTABLE. |
| H13 | Apprentissage supervisé : bouton/cockpit UI | Tests 111/320-324 = simulés Python. Pas de cockpit UI réel. | Intégration frontend. |
| H14 | Génération de slides/PDF/schéma/data-viz | Tests 309-313 = stubs. Pas de générateur réel. | Intégration (pandoc/matplotlib). |
| H15 | Pentest / OWASP Top 10 DÉTECTION réelle | Tests 285-291 = strings regex simples. Pas de scanner (cf. Aikido). | Intégration lib externe. |
| H16 | Streaming output token-par-token | Test 317 = stub `yield`. Pas de vrai stream du ReasonerBlock. | IMPLÉMENTABLE. |
| H17 | Embeddings sémantiques de texte (pas hashes) | `real_linguistic._feature_bag` = hash de caractères. Pas de sens réel. | CORPUS (Word2Vec/BERT léger). |
| H18 | Conscience épistémique mesurable | KB threshold = proxy grossier. Pas de calibration Brier/calibration proper. | IMPLÉMENTABLE. |
| H19 | Sommeil profond (multi-passes, entropie, sommeil paradoxal) | `sleep.py` = 1 passe extraction linéaire-mod. Pas de multi-phase. | IMPLÉMENTABLE. |
| H20 | IoT/Mobile/Robotique RÉELS | Tests 113-116/191-193 = skills textuels. Pas de binding Arduino/Ros/Android. | Intégration hardware. |

### 3.3. MOYENNE (utiles, mais secondaires ; interfaces déjà là)

| Gap | Capacité |
|---|---|
| M1 | Théorie de l'esprit Sally-Anne RÉELLE (pas string hardcodée) |
| M2 | Sens commun base faits réelle (ConceptNet intégration) |
| M3 | Sarcasme/ironie classifieur entraîné |
| M4 | Raisonnement moral (cadre éthique explicite + règles) |
| M5 | Raisonnement spatial géométrique (pas juste strings gauche/droite) |
| M6 | Raisonnement scientifique (hypothèse-test sur données réelles) |
| M7 | Abstraction/catégorisation entraînée |
| M8 | Estimation/probabilité calibrée |
| M9 | Explication (pourquoi cette réponse) |
| M10 | Analogie structurée (Gentner-style) |
| M11 | RAG avancé (multi-hop retrieval, re-ranking) |
| M12 | World model neuronal JEPA-prédictif RÉEL (test 188 = 10 pas, à étendre) |
| M13 | Direction/navigation (OSM/Routing API) |
| M14 | Géospatial street-view 3D réel |
| M15 | Affixes séparables et tmèse / phonologie suprasegmentale RÉELLE |
| M16 | Tokenizer BPE/sentence-piece branchable |
| M17 | Équation solver symbolique (SymPy) |
| M18 | Compression/simplification de code entraînée |
| M19 | Continual learning EWC/SI (pas juste KB store) |
| M20 | RL post-training DPO/GRPO (test 65 = DPO conceptuel) |
| M21 | Curiosité/exploration active (bonheur intrinsèque) |
| M22 | Streaming audio temps-réel full-duplex |
| M23 | Traduction FR↔EN entraînée (test 172 = KB lookup) |
| M24 | Sentiment analysis entraîné |
| M25 | Bench reproductible public (le LEVEL actuel est interne) |

### 3.4. Synthèse chiffrée du GAP

- Capacités exigées (granulaire) : ~530.
- Capacités RÉELLEMENT couvertes (RÉEL, avec métrique) : ~28 (les 28 de §2.1).
- Capacités STUB (interface exists) : ~200.
- Capacités TAUTOLOGIQUES (à ne PAS compter) : ~40.
- Capacités NON couvertes du tout (vides) : ~260.

**GAP réel à combler : ~300 capacités** (les ~200 stub à élever au rang RÉEL + les ~260 absentes, en chevauchement sur certains items). Le chiffre "528" cité par l'utilisateur correspond aux capacités non-RÉELLES (stub + tautologiques + absentes), soit 530 − 28 ≈ **502**. Cohérent.

---

## 4. PLAN D'IMPLÉMENTATION (top 15, concret)

Ordre imposé par le DAG paradigme : **primitives → grok → composer → généraliser**. Chaque item respecte `SpectralCoreBlock` comme noyau.

### TOP 15 — priorisées CRITIQUE → HAUTE

#### 1. [CRITIQUE] C1+C2+C10 — Crown-jewel MULTI-DOMAINE + MULTI-HOP par le core neural
- **Fichier** : `ocm26400/experiment_multidomain_crown.py` (nouveau).
- **Implémentation** : Pour 5 domaines (math add/mul, physics force/velocity, grammar past/plural, logic and/xor, chemistry react), entraîner le `ReasonerBlock` à `compose(a,b)` sur un échantillon, mesurer **gap one-shot vs décomp** sur un hold-out (le crown-jewel généralisé). Mesurer profondeur 3-4.
- **Test** : `test_multidomain_crown.py` — assert gap_décomp > 50pt pour chaque domaine + profondeur 3 accuracy > 80%.
- **Effet** : remplace les scores tautologiques (§2.3 A/B) par des preuves réelles.

#### 2. [CRITIQUE] C6+C9 — Décodeur de TEXTE flow-matching
- **Fichier** : `ocm26400/text_decoder.py` (nouveau), étend `generators.AMVConditionedDecoder` à des séquences de tokens.
- **Implémentation** : `x_dim = vocab_size`. Conditionné par AMV, génère une séquence de tokens via flow-matching + teacher-forcing. Test sur generation de mots anglais depuis leur AMV sémantique.
- **Test** : `test_text_decoder.py` — MSE baisse + decoded tokens == gold > 60% sur 100 mots hold-out.
- **Effet** : ouvre la génération de texte (artefact #1 exigé).

#### 3. [CRITIQUE] C3 — Règles physiques RÉELLES avec dimensional analysis
- **Fichier** : `ocm26400/rules_real.py` (nouveau).
- **Implémentation** : `Rule` étendue avec `units` (SI) + `dim_check`. Lois : F=ma, v=d/t, E=½mv², PV=nRT, F=G·m1m2/r², etc. Vérification dimensionnelle explicite (rejette `force + velocity`). `Rule.apply` produit valeur+unité.
- **Test** : `test_rules_real.py` — 20 lois physiques, `dim_check` accepte bonnes combinaisons, rejette mauvaises.
- **Effet** : remplace les `lambda mod-11` jouets par de la physique vérifiable.

#### 4. [CRITIQUE] C4 — Conjugaison FR complète
- **Fichier** : `ocm26400/morphology_fr.py` (nouveau).
- **Implémentation** : Charger dictionnaire public de conjugaison FR (Bruel/Benjamin ou `mlconjug3`), 3 groupes × 8 temps × 6 personnes. `MorphologyVerifier` étendu avec op_id par temps/groupe/personne. Règles de transformation (terminaisons) apprises par le core.
- **Test** : `test_morphology_fr.py` — 100 verbes aléatoires × 4 temps, accuracy > 90%.
- **Effet** : comble L4/L17/L18/L19/L21.

#### 5. [CRITIQUE] C7+C8 — Apprentissage PDF + cycle URL→KB RÉEL
- **Fichier** : `ocm26400/document_learner.py` (nouveau), étend `web_tools`.
- **Implémentation** : `DocumentLearner.learn_pdf(path)` → `parse_pdf` → chunking par phrase → encode chaque chunk en AMV (via `ModalityEncoder` texte) → stocke dans `KnowledgeBase`. Retrieval cosinus + retourne citation (page/phrase).
- **Test** : `test_document_learner.py` — apprend un PDF de test, répond à 5 questions avec citation correcte.
- **Effet** : comble A1/A2/A8/C7/C8.

#### 6. [CRITIQUE] H10 — RAG avec chunking + embeddings + citations
- **Fichier** : `ocm26400/rag.py` (nouveau).
- **Implémentation** : `Chunker` (sliding window 200 mots, overlap 50), `RagIndex` (KB de chunks), `retrieve(query, k=5)` retourne chunks + scores + source_uri. Branché sur `LearningAgent` (mode "je sais pas → RAG → cite").
- **Test** : `test_rag.py` — 3 docs indexés, 5 questions, top-1 retrieval sur le bon doc > 80%.
- **Effet** : comble INF7/H10.

#### 7. [HAUTE] H1 — Browser use INTERACTIF (Playwright)
- **Fichier** : `ocm26400/browser_tool.py` (nouveau).
- **Implémentation** : Wrapper autour de `playwright.sync_api`. `BrowserTool.navigate/click/fill/submit/screenshot`. Sécurité : allowlist domains + sandbox. Expose via MCP adapter.
- **Test** : `test_browser_tool.py` — navigate example.com, fetch titre, fill un form test.
- **Effet** : comble D27/H1.

#### 8. [HAUTE] H11 — Few-shot / in-context learning par le core
- **Fichier** : `ocm26400/in_context.py` (nouveau).
- **Implémentation** : `InContextLearner` : construit une séquence (ex1, ex2, ex3, query) → SpectralCoreBlock → prédit output. Entraîné sur familles de règles (chaque épisode = nouvelle règle). Mesure généralisation à nouvelle règle non vue (true few-shot).
- **Test** : `test_in_context.py` — 5-shot sur nouvelle règle, accuracy > 60%.
- **Effet** : comble R3/A11.

#### 9. [HAUTE] H19 — Sommeil multi-phases (léger + paradoxal + profond)
- **Fichier** : `ocm26400/sleep.py` (extension).
- **Implémentation** : `sleep_light` (replay + replay-shuffling), `sleep_paradoxal` (génération de variantes + consolidation créative), `sleep_deep` (extraction règle courante + généralisation à tous les domaines). Mesurer compression × N + nouvelles règles extraites.
- **Test** : `test_sleep_phases.py` — après 3 phases, KB compressée + 1 nouvelle règle correcte extraite.
- **Effet** : comble A16 (cahier des charges "plusieurs phases de sommeil").

#### 10. [HAUTE] H17 — Embeddings sémantiques RÉELS (mini BERT/word2vec)
- **Fichier** : `ocm26400/semantic_encoder.py` (nouveau).
- **Implémentation** : Soit (a) entraîner un mini skip-gram sur le corpus `real_vocab_dataset`, soit (b) charger `sentence-transformers/all-MiniLM-L6-v2`. Brancher dans `ModalityEncoder` comme vue "sémantique". Remplace le hash de caractères.
- **Test** : `test_semantic_encoder.py` — similarité cat→dog > cat→car.
- **Effet** : comble H17, ouvre la voie à RAG qualitatif.

#### 11. [HAUTE] H14 — Génération d'artefacts concrets (slides/PDF/schéma)
- **Fichier** : `ocm26400/artifact_gen.py` (nouveau).
- **Implémentation** : `SlidesGenerator` (python-pptx), `PdfGenerator` (reportlab), `SchemaGenerator` (graphviz via subprocess), `DataVizGenerator` (matplotlib). Branchés sur la sortie du text_decoder (#2).
- **Test** : `test_artifact_gen.py` — génère 1 PDF + 1 PPTX + 1 PNG schema, vérifie structure.
- **Effet** : comble G7-G11.

#### 12. [HAUTE] H3 (partiel) — Génération de code multi-langage entraînée
- **Fichier** : `ocm26400/code_generator.py` (nouveau).
- **Implémentation** : Corpus = small set of Python/JS snippets (générés par le teacher = cette session). Fine-tune le text_decoder (#2) sur ces snippets. Spécialisation par prompt de langage. Couvre Python/JS/HTML/CSS dans un premier temps.
- **Test** : `test_code_generator.py` — génère 5 fonctions Python simples, syntax-valid (compile) > 70%.
- **Effet** : comble D22/G12 (partiel).

#### 13. [HAUTE] H9 — Apprentissage YouTube transcript
- **Fichier** : `ocm26400/youtube_learner.py` (nouveau).
- **Implémentation** : `YouTubeLearner.learn(url)` → `yt-dlp` (subprocess, allowlist) pour récupérer les sous-titres → transcript → DocumentLearner (#5). Extraction ciblée (filter par timestamp/keyword).
- **Test** : `test_youtube_learner.py` — apprend 1 vidéo TED (sous-titres publics), répond à 3 questions.
- **Effet** : comble A1 (video source).

#### 14. [HAUTE] H15 — Sécurité OWASP scanner RÉEL
- **Fichier** : `ocm26400/security_scanner.py` (nouveau).
- **Implémentation** : Intégration de règles OWASP Top 10 codées (pattern matching sur code source fourni). Détecte SQL injection (string concat dans query), XSS (echo sans escape), CSRF (no token), hardcoded secrets (regex), path traversal. Report structuré.
- **Test** : `test_security_scanner.py` — 10 snippets vulnérables + 10 safe, F1 > 0.8.
- **Effet** : remplace les tests 285-291 stubs.

#### 15. [HAUTE] H16+INF2 — Streaming output + cockpit status
- **Fichier** : `ocm26400/streaming.py` (nouveau) + `ocm26400/cockpit_protocol.py`.
- **Implémentation** : `stream_generate(prompt)` generator yield token-par-token depuis le ReasonerBlock (LSRA loop affiche l'état latent à chaque itération). `CockpitProtocol` : events `apprentissage:start/progress/done`, `validation:request(user)`, `agent:status`. JSON over stdout/WebSocket.
- **Test** : `test_streaming.py` — yield count == expected tokens, status events well-formed.
- **Effet** : comble INF1/INF2/H16, prérequis cockpit UI.

### 4.1. Suite (post top-15) —摩尔蓬松 pour densifier vers 530

Une fois le top-15 livré, les catégories suivantes se traitent par **batch** (même pattern) :

- **Batch linguistique** (L) : conjugaison voix passive (op_id), modes, aspects, affixes séparables, phonologie suprasegmentale — un `op_id` par dimension dans `MorphologyVerifier`, entraîner le core sur chaque `op_id`. ~12 capacités.
- **Batch domaines** (D) : pour chacun des 30 domaines, écrire 5+ règles RÉELLES (pas mod-11) avec dimensional check — étend `rules_real.py`. ~150 capacités.
- **Batch raisonnement** (R) : each test 201-225 à réimplémenter via `ReasonerBlock` (pas hardcoded) — multi-hop, equation, planification, syllogisme, etc. ~25 capacités.
- **Batch génération** (G) : each test 308-313 à élever à génération neurale — musique (MIDI via flow-matching sur notes), schema (graphviz), infographie (matplotlib). ~6 capacités.
- **Batch multimodal RÉEL** (MM) : brancher datasets réels (LibriSpeech, COCO-MNIST, Moving-MNIST → VideoMME). Nécessite corpus —Downloader + adaptateur. ~10 capacités.

### 4.2. Honnêteté : ce qui NÉCESSITE un corpus externe (pas implémentable seul)

Capacités qui, **même avec le paradigme compositionnel**, ne peuvent pas être "grokkées" sans données réelles étiquetées :

- H4 (vidéo réelle) : nécessite VideoMME/ Kinetics.
- H5 (parole réelle) : nécessite LibriSpeech/CommonVoice.
- H6 (object detection) : nécessite COCO/OpenImages.
- H7 (OCR) : nécessite IAM/ICDAR.
- H8 (radar/SAR) : nécessite SENTINEL-1.
- H3 (code à l'échelle SWE-bench) : nécessite corpus GitHub.

**Recommandation** : pour ces 6, le cahier des charges ne peut PAS être "rempli" par densification interne seule. Il faut soit (a) déclarer un sous-périmètre honnête (ex. : "object detection sur MNIST digits uniquement"), soit (b) brancher un dataset public via un adaptateur. Le paradigme OCM reste valable pour **apprendre ces tâches avec peu d'exemples** (crown-jewel), mais pas sans **aucun** exemple.

---

## 5. RECOMMANDATIONS TRANSVERSES (audit)

1. **Recalculer le BENCH_LEVEL sans les scores tautologiques** (§2.3 A/B/C). Le "94.9/100" actuel sur-évalue. Un BENCH_LEVEL recalculé (excluant `domain_trainer.evaluate_all_domains` et `reasoning_bench_aime`) serait plus défendable scientifiquement.
2. **Externaliser les règles physiques réelles** (top-15 #3) AVANT de continuer à densifier les règles mod-11. Sinon le "30 domaines" restera cosmétique.
3. **Brancher un cockpit UI minimal** (top-15 #15) — le cahier des charges y revient 5+ fois (bouton validation, statut apprentissage, voir ce que le modèle génère).
4. **Définir un sous-périmètre PUBLIC honnête** pour les benchmarks : "OCM-26400 rivalise sur tâches compositionnelles vérifiables (arithmétique modulaire, morphologie, raisonnement multi-étapes), pas sur VideoMME/COCO/SWE-bench qui nécessitent corpus". Cela cadre les attentes vs frontières (Claude/GPT-4).
5. **Tests de non-régression sur les 28 capacités RÉELLES** (§2.1) avant tout ajout — ce sont les seules qui prouvent quelque chose. Ne pas les casser en densifiant.
6. **Tagger chaque test** avec `[RÉEL]` / `[STUB]` / `[TAUTO]` dans le docstring — l'audit a révélé que sans ce tagging, le STATUS mélange allègrement. Un simple `# COVERAGE: STUB` en commentaire permettrait à la prochaine itération de prioriser.

---

## 6. CONCLUSION

- **28 capacités RÉELLEMENT démontrées** (avec métrique) sur ~530 exigées. Le crown-jewel compositionnel (+99.5pt) est la preuve cardinale du paradigme — il est solide et reproductible.
- **~200 stubs** sont à élever au rang RÉEL — chacun est un petit chantier (1 fichier + 1 test + 1 entraînement court).
- **~40 tests tautologiques** sont à RETIRER du discours "100% maîtrise" (en particulier `domain_trainer.py` et `reasoning_bench_aime`).
- **~260 capacités absentes** dont les 15 prioritaires du §4 ouvrent les plus grosses (texte/PDF/RAG/browser/code/FR/sometil multi-phase).
- Le paradigme est **sain** (primitives → grok → composer → généraliser, `SpectralCoreBlock` noyau unifié, d≈256). Aucune refacto d'architecture nécessaire — juste de la **densification ciblée** par couronne (crown-jewel) sur les dimensions manquantes.

Le top-15 ci-dessus est l'ordre imposé par le DAG paradigme et le juge (densification par grok compositionnel). Une fois livré, le score RÉEL (non tautologique) passerait de 28 à ~80-100 capacités démontrées, soit le seuil minimal pour déclarer "densification substantielle du cahier des charges".
