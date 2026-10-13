# Subwordphnx-BPE n-gram sweep on LibriSpeech 960h

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes of that one shuffle. 158 subword BPE merges over phones are learned fresh from each fraction's own training subset only (never from dev-clean, never reused across fractions), with every merge confined to a single word, so a piece is a run of phones inside one word ("the" is the single piece `DH-AH`). That is the merge count the neural unit has at vocabulary 200, which is what makes the two stages the same unit. Vocabulary is built from each cell's training subset only. Evaluated on dev-clean. See design.md, "The subwordphnx-BPE unit".

### Bit rate (bits/s)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 52.98 | 43.23 | 36.26 | 34.37 | 34.57 |
| 50% | 53.13 | 43.30 | 35.94 | 33.59 | 33.72 |
| 75% | 53.12 | 43.30 | 35.77 | 33.17 | 33.21 |
| 100% | 53.12 | 43.30 | 35.66 | 32.88 | 32.87 |

### Perplexity (per piece, not comparable across fractions)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 142.41 | 57.19 | 29.79 | 24.94 | 25.42 |
| 50% | 143.16 | 57.19 | 28.74 | 23.08 | 23.36 |
| 75% | 143.03 | 57.15 | 28.27 | 22.17 | 22.27 |
| 100% | 143.03 | 57.13 | 27.97 | 21.58 | 21.56 |

### Top-order discount used (Chen & Goodman; see design.md)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 0.75 | 0.41 | 0.58 | 0.73 | 0.84 |
| 50% | 0.75 | 0.42 | 0.56 | 0.71 | 0.82 |
| 75% | 0.75 | 0.42 | 0.55 | 0.69 | 0.80 |
| 100% | 0.75 | 0.42 | 0.54 | 0.68 | 0.79 |
