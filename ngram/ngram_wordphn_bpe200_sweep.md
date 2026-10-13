# Wordphn-BPE n-gram sweep on LibriSpeech 960h

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes of that one shuffle. 200 whole-unit BPE merges are learned fresh from each fraction's own training subset only (never from dev-clean, never reused across fractions), with `wordphn` units as the atomic symbols, so a unit is a collocation written as phones ("of the" is `AH-V DH-AH`). Vocabulary is built from each cell's training subset only. Evaluated on dev-clean. See design.md, "The wordphn-BPE unit".

### Bit rate (bits/s)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 36.77 | 32.95 | 32.70 | 32.83 | 32.95 |
| 50% | 36.48 | 32.16 | 31.76 | 31.90 | 32.02 |
| 75% | 36.39 | 31.82 | 31.32 | 31.46 | 31.58 |
| 100% | 36.31 | 31.55 | 30.97 | 31.10 | 31.23 |

### Perplexity (per unit, not comparable across fractions)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 3561.34 | 1521.78 | 1441.46 | 1483.78 | 1523.45 |
| 50% | 3343.90 | 1280.35 | 1171.18 | 1207.27 | 1240.78 |
| 75% | 3287.61 | 1188.75 | 1063.85 | 1096.35 | 1127.19 |
| 100% | 3220.94 | 1117.96 | 983.11 | 1012.09 | 1040.46 |

### Top-order discount used (Chen & Goodman; see design.md)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 0.57 | 0.76 | 0.90 | 0.90 | 0.90 |
| 50% | 0.58 | 0.75 | 0.89 | 0.90 | 0.90 |
| 75% | 0.58 | 0.74 | 0.88 | 0.90 | 0.90 |
| 100% | 0.59 | 0.73 | 0.87 | 0.90 | 0.90 |
