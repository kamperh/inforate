# Wordphn-BPE n-gram sweep on LibriSpeech 960h

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes of that one shuffle. 1000 whole-unit BPE merges are learned fresh from each fraction's own training subset only (never from dev-clean, never reused across fractions), with `wordphn` units as the atomic symbols, so a unit is a collocation written as phones ("of the" is `AH-V DH-AH`). Vocabulary is built from each cell's training subset only. Evaluated on dev-clean. See design.md, "The wordphn-BPE unit".

### Bit rate (bits/s)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 36.29 | 32.98 | 32.87 | 32.97 | 33.07 |
| 50% | 35.99 | 32.12 | 31.89 | 32.00 | 32.10 |
| 75% | 35.91 | 31.74 | 31.42 | 31.53 | 31.63 |
| 100% | 35.82 | 31.45 | 31.05 | 31.16 | 31.26 |

### Perplexity (per unit, not comparable across fractions)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 5830.08 | 2645.50 | 2574.27 | 2639.25 | 2703.46 |
| 50% | 5451.62 | 2160.54 | 2042.48 | 2096.94 | 2150.51 |
| 75% | 5333.66 | 1969.57 | 1821.80 | 1870.15 | 1918.40 |
| 100% | 5235.33 | 1842.40 | 1672.77 | 1717.41 | 1762.01 |

### Top-order discount used (Chen & Goodman; see design.md)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 0.57 | 0.78 | 0.90 | 0.90 | 0.90 |
| 50% | 0.58 | 0.76 | 0.90 | 0.90 | 0.90 |
| 75% | 0.58 | 0.75 | 0.90 | 0.90 | 0.90 |
| 100% | 0.59 | 0.74 | 0.90 | 0.90 | 0.90 |
