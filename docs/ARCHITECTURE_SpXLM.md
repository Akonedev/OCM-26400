# ARCHITECTURE SpXLM — Rapport du College d'Experts
# Spectral eXponential LSTM Language Model
# Date: 9 Juin 2026

## COMPOSANTS RETENUS (apres debat + DA)

| Composant | Source | Verdict | Raison |
|---|---|---|---|
| xLSTM exponential gating (sLSTM) | E1 | GARDER | Empiriquement valide, simple, robuste |
| Diffusion Head | E1 | REJETER | Trop lent, instable a petite echelle |
| FFT Spectral Mixing | E2 | GARDER (complement) | Capture patterns globaux, O(N*logN) |
| Quaternions | E2 | REJETER | Pas de preuve en NLP |
| CfC | E2 | REJETER | Pas de kernels ROCm |
| BPE reduit (4K) | E3 | GARDER | Economie massive embedding (13M liberes) |
| Hopfield Memory | E3 | REJETER | Trop complexe, convergence couteuse |
| Local KV Attention | E3 | GARDER | Simple, efficace, adressage contenu |
| Dual Process | E3 | REJETER | Non prouve a cette echelle |
| Curriculum Training | E3 | GARDER | Eprouve, pas risque |
| Weight Tying | Tous | GARDER | Gratuit, +0.1-0.3 PPL |
| SwiGLU FFN | Standard | GARDER | +0.2-0.5 PPL vs ReLU |

## SCHEMA ARCHITECTURAL

```
Input Tokens (seq_len=256)
        |
        v
[EMBEDDING] vocab=4000, dim=512 -> 2.048M params
[+ RoPE positional encoding] -> ~0 params (fixed)
        |
        v
[BLOCK x6] ~2.1M params each = 12.6M total
  |
  +-- LayerNorm
  +-- xLSTM-sLSTM (exponential gating, hidden=512) ~1.05M/couche
  +-- Residual +
  +-- LayerNorm
  +-- Spectral Mixing (FFT + Linear + iFFT) ~0.26M/couche
  +-- Local KV Attention (window=32, heads=8) ~0.21M/couche
  +-- SwiGLU FFN (512->1024->512) ~0.53M/couche
  +-- Residual +
        |
        v
[OUTPUT HEAD] LayerNorm -> Linear -> Softmax
  Weight tying avec embedding -> 0 params additionnels
        |
        v
Output logits (256, 4000)
```

## BUDGET PARAMS

| Composant | Params |
|---|---|
| Embedding (incl. tied head) | 2.048M |
| xLSTM x6 | 6.306M |
| LayerNorm x13 | 0.013M |
| Spectral MLP x6 | 1.576M |
| Local Attention x6 | 1.573M |
| SwiGLU FFN x6 | 3.150M |
| Output LayerNorm | 0.001M |
| **TOTAL** | **~14.77M** |
| **Marge** | **~1.23M** |

## PPL ESTIMEE

- Baseline MiniGPT: 5.07
- Gains cumules: -0.68
- **PPL estimee: ~4.39**
- **Fourchette: 4.2 - 4.7**
- Score de confiance: 7.5/10

## RISQUES + MITIGATION

1. xLSTM < attendu a petite echelle -> fallback mLSTM
2. FFT + RNN mauvaise synergie -> ablation, tester separement
3. BPE 4K trop restrictif -> augmenter a 6K
4. Convergence difficile -> warmup separe (RNN seul puis + FFT)
5. ROCm compatibilite -> kernels standards PyTorch优先

## PLAN IMPLEMENTATION

Phase 1: Baseline xLSTM pur (6 couches, 512 dims, BPE 4K) -> PPL < 5.5
Phase 2: + Spectral Mixing -> PPL < 5.0
Phase 3: + Local Attention + SwiGLU -> PPL < 4.7
Phase 4: Curriculum training + tuning -> PPL < 4.5
