# Subwordphnx-BPE n-gram sweep on LibriSpeech 960h

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes of that one shuffle. 1,958 subword BPE merges over phones are learned fresh from each fraction's own training subset only (never from dev-clean, never reused across fractions), with every merge confined to a single word, so a piece is a run of phones inside one word ("the" is the single piece `DH-AH`). That is the merge count the neural unit has at vocabulary 2,000, which is what makes the two stages the same unit. Vocabulary is built from each cell's training subset only. Evaluated on dev-clean. See design.md, "The subwordphnx-BPE unit".

### Bit rate (bits/s)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 44.80 | 35.48 | 32.85 | 32.89 | 33.04 |
| 50% | 44.81 | 35.21 | 32.18 | 32.15 | 32.33 |
| 75% | 44.78 | 35.10 | 31.84 | 31.77 | 31.95 |
| 100% | 44.78 | 35.03 | 31.59 | 31.48 | 31.67 |

### Perplexity (per piece, not comparable across fractions)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 727.89 | 184.77 | 125.62 | 126.26 | 129.19 |
| 50% | 728.14 | 177.61 | 113.71 | 113.24 | 116.17 |
| 75% | 725.26 | 174.78 | 108.15 | 107.00 | 109.98 |
| 100% | 726.76 | 173.09 | 104.42 | 102.68 | 105.59 |

### Top-order discount used (Chen & Goodman; see design.md)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 0.50 | 0.62 | 0.79 | 0.90 | 0.90 |
| 50% | 0.75 | 0.60 | 0.77 | 0.89 | 0.90 |
| 75% | 0.75 | 0.58 | 0.76 | 0.88 | 0.90 |
| 100% | 0.75 | 0.57 | 0.75 | 0.87 | 0.90 |
