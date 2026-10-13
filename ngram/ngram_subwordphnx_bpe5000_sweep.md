# Subwordphnx-BPE n-gram sweep on LibriSpeech 960h

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes of that one shuffle. 4,958 subword BPE merges over phones are learned fresh from each fraction's own training subset only (never from dev-clean, never reused across fractions), with every merge confined to a single word, so a piece is a run of phones inside one word ("the" is the single piece `DH-AH`). That is the merge count the neural unit has at vocabulary 5,000, which is what makes the two stages the same unit. Vocabulary is built from each cell's training subset only. Evaluated on dev-clean. See design.md, "The subwordphnx-BPE unit".

### Bit rate (bits/s)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 41.46 | 33.71 | 32.36 | 32.49 | 32.62 |
| 50% | 41.49 | 33.37 | 31.67 | 31.76 | 31.91 |
| 75% | 41.47 | 33.20 | 31.31 | 31.38 | 31.54 |
| 100% | 41.46 | 33.08 | 31.05 | 31.09 | 31.24 |

### Perplexity (per piece, not comparable across fractions)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 1085.92 | 293.69 | 233.87 | 239.08 | 244.57 |
| 50% | 1083.74 | 276.13 | 207.09 | 210.60 | 215.81 |
| 75% | 1084.95 | 269.08 | 195.72 | 198.23 | 203.41 |
| 100% | 1085.20 | 264.20 | 187.52 | 188.82 | 193.83 |

### Top-order discount used (Chen & Goodman; see design.md)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 0.38 | 0.68 | 0.83 | 0.90 | 0.90 |
| 50% | 0.54 | 0.67 | 0.81 | 0.90 | 0.90 |
| 75% | 0.20 | 0.66 | 0.80 | 0.90 | 0.90 |
| 100% | 0.11 | 0.65 | 0.79 | 0.90 | 0.90 |
