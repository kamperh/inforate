# Word-BPE n-gram sweep on LibriSpeech 960h

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes of that one shuffle. 1000 word-level BPE merges are learned fresh from each fraction's own training subset only (never from dev-clean, never reused across fractions). Vocabulary is built from each cell's training subset only. Evaluated on dev-clean.

### Bit rate (bits/s)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 36.04 | 32.61 | 32.48 | 32.59 | 32.69 |
| 50% | 35.70 | 31.71 | 31.46 | 31.57 | 31.68 |
| 75% | 35.59 | 31.32 | 30.97 | 31.08 | 31.19 |
| 100% | 35.49 | 31.01 | 30.59 | 30.70 | 30.81 |

### Perplexity (per word-BPE unit, not comparable across fractions)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 6065.60 | 2649.77 | 2567.05 | 2633.64 | 2699.69 |
| 50% | 5596.53 | 2136.32 | 2007.48 | 2062.86 | 2117.08 |
| 75% | 5450.78 | 1941.66 | 1783.04 | 1833.24 | 1881.69 |
| 100% | 5323.43 | 1803.42 | 1626.45 | 1671.77 | 1716.23 |

### Top-order discount used (Chen & Goodman; see design.md)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 0.58 | 0.78 | 0.90 | 0.90 | 0.90 |
| 50% | 0.58 | 0.76 | 0.90 | 0.90 | 0.90 |
| 75% | 0.59 | 0.75 | 0.90 | 0.90 | 0.90 |
| 100% | 0.60 | 0.74 | 0.90 | 0.90 | 0.90 |
