# Word-BPE n-gram sweep on LibriSpeech 960h

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes of that one shuffle. 5000 word-level BPE merges are learned fresh from each fraction's own training subset only (never from dev-clean, never reused across fractions). Vocabulary is built from each cell's training subset only. Evaluated on dev-clean.

### Bit rate (bits/s)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 35.39 | 32.79 | 32.80 | 32.88 | 32.96 |
| 50% | 35.05 | 31.85 | 31.78 | 31.86 | 31.95 |
| 75% | 34.93 | 31.40 | 31.26 | 31.35 | 31.44 |
| 100% | 34.83 | 31.05 | 30.87 | 30.96 | 31.05 |

### Perplexity (per word-BPE unit, not comparable across fractions)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 11742.43 | 5907.78 | 5915.93 | 6044.43 | 6175.04 |
| 50% | 10761.47 | 4613.13 | 4525.80 | 4631.37 | 4738.40 |
| 75% | 10477.87 | 4104.85 | 3960.51 | 4056.88 | 4153.36 |
| 100% | 10196.06 | 3744.04 | 3563.09 | 3648.93 | 3737.66 |

### Top-order discount used (Chen & Goodman; see design.md)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 0.58 | 0.81 | 0.90 | 0.90 | 0.90 |
| 50% | 0.58 | 0.79 | 0.90 | 0.90 | 0.90 |
| 75% | 0.59 | 0.77 | 0.90 | 0.90 | 0.90 |
| 100% | 0.60 | 0.77 | 0.90 | 0.90 | 0.90 |
