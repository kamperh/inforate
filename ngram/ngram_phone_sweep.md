# Phone-level n-gram sweep on LibriSpeech 960h

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes of that one shuffle. Vocabulary is built from each cell's training subset only. Evaluated on dev-clean. Phones are unstressed (standard 39-symbol ARPABET set); see design.md, 'Data'.

### Bit rate (bits/s)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 60.63 | 50.46 | 43.68 | 38.76 | 36.06 |
| 50% | 60.63 | 50.46 | 43.66 | 38.57 | 35.40 |
| 75% | 60.63 | 50.46 | 43.65 | 38.49 | 35.11 |
| 100% | 60.63 | 50.46 | 43.65 | 38.45 | 34.93 |

### Perplexity (per phone)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 30.51 | 17.20 | 11.73 | 8.89 | 7.64 |
| 50% | 30.51 | 17.19 | 11.72 | 8.80 | 7.36 |
| 75% | 30.51 | 17.19 | 11.71 | 8.76 | 7.24 |
| 100% | 30.51 | 17.19 | 11.71 | 8.74 | 7.16 |

### Top-order discount used (Chen & Goodman; see design.md)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 0.75 | 0.49 | 0.47 | 0.52 | 0.60 |
| 50% | 0.75 | 0.47 | 0.49 | 0.51 | 0.58 |
| 75% | 0.75 | 0.59 | 0.48 | 0.50 | 0.57 |
| 100% | 0.75 | 0.69 | 0.48 | 0.50 | 0.56 |
