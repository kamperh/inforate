# Word-BPE n-gram sweep on LibriSpeech 960h

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes of that one shuffle. 200 word-level BPE merges are learned fresh from each fraction's own training subset only (never from dev-clean, never reused across fractions). Vocabulary is built from each cell's training subset only. Evaluated on dev-clean.

### Bit rate (bits/s)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 36.51 | 32.60 | 32.32 | 32.44 | 32.57 |
| 50% | 36.17 | 31.78 | 31.34 | 31.48 | 31.61 |
| 75% | 36.07 | 31.42 | 30.88 | 31.02 | 31.15 |
| 100% | 35.97 | 31.13 | 30.52 | 30.65 | 30.78 |

### Perplexity (per word-BPE unit, not comparable across fractions)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 3649.64 | 1516.77 | 1423.40 | 1465.39 | 1505.84 |
| 50% | 3393.59 | 1263.53 | 1146.17 | 1181.98 | 1216.00 |
| 75% | 3310.58 | 1164.41 | 1032.88 | 1065.27 | 1096.08 |
| 100% | 3238.47 | 1092.18 | 952.18 | 981.09 | 1009.38 |

### Top-order discount used (Chen & Goodman; see design.md)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 0.58 | 0.76 | 0.90 | 0.90 | 0.90 |
| 50% | 0.58 | 0.75 | 0.89 | 0.90 | 0.90 |
| 75% | 0.59 | 0.74 | 0.88 | 0.90 | 0.90 |
| 100% | 0.60 | 0.73 | 0.87 | 0.90 | 0.90 |
