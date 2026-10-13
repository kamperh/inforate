# Wordphn-BPE n-gram sweep on LibriSpeech 960h

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes of that one shuffle. 5000 whole-unit BPE merges are learned fresh from each fraction's own training subset only (never from dev-clean, never reused across fractions), with `wordphn` units as the atomic symbols, so a unit is a collocation written as phones ("of the" is `AH-V DH-AH`). Vocabulary is built from each cell's training subset only. Evaluated on dev-clean. See design.md, "The wordphn-BPE unit".

### Bit rate (bits/s)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 35.68 | 33.15 | 33.16 | 33.24 | 33.32 |
| 50% | 35.37 | 32.23 | 32.18 | 32.26 | 32.35 |
| 75% | 35.28 | 31.82 | 31.70 | 31.78 | 31.87 |
| 100% | 35.19 | 31.49 | 31.32 | 31.41 | 31.49 |

### Perplexity (per unit, not comparable across fractions)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 10962.66 | 5660.77 | 5679.66 | 5799.25 | 5921.25 |
| 50% | 10196.58 | 4496.46 | 4429.34 | 4528.63 | 4630.01 |
| 75% | 10014.87 | 4049.41 | 3924.36 | 4016.61 | 4108.85 |
| 100% | 9776.88 | 3714.75 | 3554.27 | 3636.75 | 3721.80 |

### Top-order discount used (Chen & Goodman; see design.md)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 0.57 | 0.81 | 0.90 | 0.90 | 0.90 |
| 50% | 0.58 | 0.79 | 0.90 | 0.90 | 0.90 |
| 75% | 0.58 | 0.77 | 0.90 | 0.90 | 0.90 |
| 100% | 0.59 | 0.77 | 0.90 | 0.90 | 0.90 |
