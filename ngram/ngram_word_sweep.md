# Word-level n-gram sweep on LibriSpeech 960h

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes of that one shuffle. Vocabulary is built from each cell's training subset only. Evaluated on dev-clean.

### Bit rate (bits/s)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 37.12 | 32.66 | 32.30 | 32.46 | 32.62 |
| 50% | 36.79 | 31.90 | 31.39 | 31.54 | 31.70 |
| 75% | 36.68 | 31.58 | 30.96 | 31.09 | 31.26 |
| 100% | 36.59 | 31.32 | 30.62 | 30.74 | 30.91 |

### Perplexity

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 1798.88 | 730.16 | 679.56 | 702.09 | 724.29 |
| 50% | 1682.81 | 626.66 | 565.65 | 582.66 | 602.10 |
| 75% | 1646.55 | 587.18 | 517.83 | 532.24 | 550.41 |
| 100% | 1613.97 | 557.54 | 484.11 | 495.79 | 513.02 |

### Top-order discount used (Chen & Goodman; see design.md)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 0.58 | 0.76 | 0.87 | 0.90 | 0.90 |
| 50% | 0.58 | 0.75 | 0.86 | 0.90 | 0.90 |
| 75% | 0.59 | 0.74 | 0.85 | 0.90 | 0.90 |
| 100% | 0.60 | 0.74 | 0.84 | 0.90 | 0.90 |
