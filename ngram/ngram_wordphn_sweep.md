# Wordphn n-gram sweep on LibriSpeech 960h

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes of that one shuffle. Vocabulary is built from each cell's training subset only. Evaluated on dev-clean. A unit is one word interval written as its own destressed phones joined by `-` ("abbey" is `AE-B-IY`); see design.md, "The wordphn unit".

### Bit rate (bits/s)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 37.37 | 32.99 | 32.66 | 32.82 | 32.97 |
| 50% | 37.07 | 32.27 | 31.79 | 31.94 | 32.10 |
| 75% | 36.99 | 31.97 | 31.38 | 31.52 | 31.68 |
| 100% | 36.90 | 31.72 | 31.05 | 31.17 | 31.34 |

### Perplexity (per realisation; comparable with the word sweep's, which scores the same token count)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 1888.98 | 780.48 | 731.25 | 755.29 | 778.47 |
| 50% | 1781.41 | 674.88 | 613.09 | 631.94 | 652.25 |
| 75% | 1751.91 | 634.94 | 563.88 | 579.92 | 598.91 |
| 100% | 1721.07 | 604.63 | 528.22 | 541.27 | 559.30 |

### Top-order discount used (Chen & Goodman; see design.md)

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 |
|---|---:|---:|---:|---:|---:|
| 25% | 0.57 | 0.76 | 0.88 | 0.90 | 0.90 |
| 50% | 0.58 | 0.75 | 0.86 | 0.90 | 0.90 |
| 75% | 0.58 | 0.74 | 0.85 | 0.90 | 0.90 |
| 100% | 0.59 | 0.73 | 0.85 | 0.90 | 0.90 |
