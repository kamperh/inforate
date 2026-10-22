# Subwordphnx-BPE n-gram sweep on LibriSpeech 960h

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes of that one shuffle. 49,958 subword BPE merges over phones are learned fresh from each fraction's own training subset only (never from dev-clean, never reused across fractions), with every merge confined to a single word, so a piece is a run of phones inside one word ("the" is the single piece `DH-AH`). That is the merge count the neural unit has at vocabulary 50,000, which is what makes the two stages the same unit. Vocabulary is built from each cell's training subset only. Evaluated on dev-clean. See design.md, "The subwordphnx-BPE unit".

### Bit rate (bits/s)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 37.19 | 32.56 | 32.21 | 32.37 | 32.52 |
| 50% | 36.99 | 31.93 | 31.44 | 31.58 | 31.74 |
| 75% | 36.93 | 31.66 | 31.04 | 31.18 | 31.33 |
| 100% | 36.90 | 31.45 | 30.75 | 30.87 | 31.02 |

### Perplexity (per piece, not comparable across fractions)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 1561.31 | 624.65 | 583.57 | 602.05 | 619.82 |
| 50% | 1554.65 | 568.96 | 515.83 | 531.10 | 547.50 |
| 75% | 1545.22 | 541.12 | 479.14 | 492.05 | 507.33 |
| 100% | 1539.41 | 520.82 | 453.22 | 463.60 | 478.07 |

### Top-order discount used (Chen & Goodman; see design.md)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 0.11 | 0.76 | 0.88 | 0.90 | 0.90 |
| 50% | 0.12 | 0.74 | 0.86 | 0.90 | 0.90 |
| 75% | 0.40 | 0.73 | 0.85 | 0.90 | 0.90 |
| 100% | 0.41 | 0.72 | 0.85 | 0.90 | 0.90 |
