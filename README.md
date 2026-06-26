# OCM-26400 — Omni-Cognitive Mentalese Architecture

> **Compréhension > Mémoire, TOUJOURS.** Un modèle spectral unifié (FFT) qui grok les règles et génère depuis la compréhension.

[![Tests](https://img.shields.io/badge/tests-1137%2F1137-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## Qu'est-ce qu'OCM-26400 ?

OCM-26400 est une architecture d'IA **unifiée** basée sur un **mélangeur spectral FFT**
(`SpectralCoreBlock`) qui remplace l'attention des transformers. Un seul noyau de
**675K paramètres fixes** traite tous les domaines : maths, logique, langage, physique,
audio, image, vidéo, 3D, monde physique.

### Le crown-jewel : décomposition > one-shot

À paramètres constants, la **décomposition** (calculer l'intermédiaire puis le final)
généralise à **100%** sur des compositions jamais vues, tandis que le one-shot échoue à
**0.5%**. Ce gap de **+99.5 points** est le fondement du projet.

## Démarrage rapide

```bash
# Pre-training automatique (curriculum v4 complet)
python auto_pretrain.py --device cpu --steps 2000

# Training cross-modal sur vraies données
python auto_train.py --device cuda --steps 10000

# Fine-tuning spécialisé
python auto_finetune.py --domain audio --steps 5000

# Tester tous les modes
python test_omni.py bench
```

## Architecture

```
SpectralCoreBlock (FFT bidirectionnel, O(L log L))
├── in_proj → LayerNorm → rfft → filter (appris) → irfft → out_proj
├── résiduel spectral (stabilité Parseval)
└── FFN (4× expansion, GELU)

AMV-256 = [ent(64) | prop(64) | op(64) | meta(64)]
          entité    propriété  opérateur  méta(confidence, source, consist)

LSRA : v(t+1) = Block(v(t)), stop quand sigmoid(meta[0]) ≥ 0.9
```

## Résultats

| Domaine | Score | Méthode |
|---------|-------|---------|
| Arithmétique (crown-jewel) | 100% | FFT grok + décomposition |
| Logique (AND/OR/NOT/IMP/IFF) | 100% | IDs + FFT grok |
| Morphologie (EN/FR) | 100% | char-level grok règles |
| Video / 3D / World | 100% | règles arithmétiques sur IDs |
| Audio (génération) | 97% | règles phonétiques → Mel généré |
| Image (classification) | 89.5% | cross-modal simultané |
| Image (génération) | 78% | flow-matching concept→patches |
| Tests automatisés | 1137/1137 | suite complète |

## Principes

1. **Compréhension > Mémoire** — le grokking spectral encode la règle, pas l'instance
2. **Zéro texte comme cœur** — tout est IDs numériques manipulés par le noyau FFT
3. **Profondeur > Params** — raisonner = ajouter des étapes (L4), pas des poids
4. **IDs numériques** — le grokking marche sur des associations entre NOMBRES
5. **Capture simultanée** — toutes les modalités en une passe pour les associations
6. **Génération depuis règles** — créer depuis la compréhension, pas depuis la mémoire

## Documentation

- [Publication scientifique](PUBLICATION_OCM26400.md)
- [Formules et découvertes](FORMULAS_AND_DISCOVERIES.md)
- [Conditions de grokking](GROKKING_CONDITIONS.md)
- [Capacités mesurées](CAPABILITIES.md)
- [Procédures (1194 lignes)](PROCEDURES.md)
- [Besoins et spec](Besoins/)

## Licence

MIT — voir [LICENSE](LICENSE)

## Auteur

**akone** — Recherches MathBase / Hermes
