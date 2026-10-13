# LibriSpeech / dev-clean statistics

| set | utterances | speech duration | word tokens | word vocab | phone tokens | phone inventory (unstressed) | syllable tokens | syllable inventory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LibriSpeech 960h | 281,239 | 769.8h (2771365s) | 9,403,427 | 89,048 | 33,464,416 | 39 | 13,025,947 | 13,656 |
| dev-clean | 2,703 | 4.4h (15846s) | 54,402 | 8,332 | 194,827 | 39 | 75,910 | 4,082 |

## Words-tier cleanliness check

- LibriSpeech 960h: words-tier text is clean ([a-z']+ only).
- dev-clean: words-tier text is clean ([a-z']+ only).

## Realisation-based word vocabulary (the wordphn unit)

Each word counted by its aligned phones rather than its spelling ("abbey" is `AE-B-IY`); see llm/wordphn_prepare_mmap.py.

- LibriSpeech 960h: 9,403,427 wordphn tokens (identical to word tokens, one unit per word interval), 84,163 realisation types against 89,048 orthographic word types, of which 27,908 occur once. 3,674 word types have more than one realisation, and 7,104 realisations cover more than one word type.
- dev-clean: 54,402 wordphn tokens (identical to word tokens, one unit per word interval), 8,372 realisation types against 8,332 orthographic word types, of which 4,618 occur once. 248 word types have more than one realisation, and 203 realisations cover more than one word type.
