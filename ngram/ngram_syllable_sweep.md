# Syllable-level n-gram sweep on LibriSpeech 960h

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes of that one shuffle. Vocabulary is built from each cell's training subset only. Evaluated on dev-clean. Out-of-vocabulary syllables are spelled out via their constituent phones (`token.split("-")`), not characters.

### Bit rate (bits/s)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 46.22 | 35.24 | 32.98 | 33.06 | 33.25 |
| 50% | 46.21 | 34.91 | 32.28 | 32.30 | 32.51 |
| 75% | 46.20 | 34.78 | 31.94 | 31.91 | 32.13 |
| 100% | 46.20 | 34.68 | 31.68 | 31.61 | 31.82 |

### Perplexity (per syllable)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 802.88 | 163.92 | 118.20 | 119.43 | 122.83 |
| 50% | 800.63 | 156.28 | 106.75 | 107.02 | 110.32 |
| 75% | 800.48 | 153.33 | 101.59 | 101.16 | 104.41 |
| 100% | 799.50 | 151.08 | 97.82 | 96.85 | 99.93 |

### Top-order discount used (Chen & Goodman; see design.md)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 0.51 | 0.67 | 0.80 | 0.90 | 0.90 |
| 50% | 0.53 | 0.66 | 0.78 | 0.88 | 0.90 |
| 75% | 0.54 | 0.65 | 0.77 | 0.87 | 0.90 |
| 100% | 0.54 | 0.65 | 0.76 | 0.86 | 0.90 |
