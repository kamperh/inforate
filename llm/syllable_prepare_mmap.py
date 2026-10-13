"""Train a syllable vocabulary (hybrid syllable/phone backoff, min-count 2) on
LibriSpeech 960h training `syllables`-tier sentences, save it, apply random
phone-spelling augmentation to a copy of the training data, then dump a flat token-id
mmap from the augmented data.

Structurally identical to llm/word_prepare_mmap.py (see that file's docstring for the
full reasoning behind the vocabulary rule and the augmentation mechanics decision); the
differences are exactly `SyllableTokenizer`'s two differences from `WordTokenizer`
(llm/tokenisers.py): the finer unit is phones, not characters, and splitting is
`token.split("-")`, not `list(token)`, since the `syllables` tier already writes each
syllable as its phones joined by "-".

Vocabulary: syllable types occurring at least MIN_COUNT times in training data only
(`data_common.build_min_count_vocab`). Rarer training types and unseen dev-clean
syllables are spelled out as phones by `SyllableTokenizer` itself (no separate `<unk>`
case).

Random augmentation is deliberately handled here, not inside `SyllableTokenizer`:
independently for AUGMENT_RATE of *all* syllable occurrences in the *training* data only
(regardless of whether the syllable is in-vocab), replace the syllable with its own
phones before encoding. Validation/evaluation never applies this:
`SyllableTokenizer.encode()` only ever runs the deterministic in-vocab/spell-out rule,
identically at train and test time.
"""

from pathlib import Path

import numpy as np

from data_common import (
    TRAIN_SPLITS,
    augment_with_spellout,
    build_min_count_vocab,
    read_tier,
    write_train_mmap,
)
from tokenisers import SyllableTokenizer

OUT_DIR = Path(__file__).resolve().parent / "data" / "syllable-mmap"
VOCAB_PATH = OUT_DIR / "vocab.json"
MIN_COUNT = 2
AUGMENT_RATE = 0.02
AUGMENT_SEED = 0
DTYPE = np.uint16


if __name__ == "__main__":
    print("Reading syllables tier for:", TRAIN_SPLITS)
    utterances = read_tier(TRAIN_SPLITS, "syllables")
    print(f"Utterances: {len(utterances):,}")

    syllable_vocab = build_min_count_vocab(
        [tokens for tokens, _ in utterances.values()], min_count=MIN_COUNT
    )
    print(f"Syllable vocab (count >= {MIN_COUNT}): {len(syllable_vocab):,} types")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SyllableTokenizer.save_vocab(syllable_vocab, VOCAB_PATH)
    print("Saved vocab to:", VOCAB_PATH)

    tokenizer = SyllableTokenizer(VOCAB_PATH)
    assert tokenizer.vocab_size < np.iinfo(DTYPE).max
    print(f"Full tokeniser vocab (syllables + phones + specials): {tokenizer.vocab_size:,}")

    print(f"Random augmentation: {AUGMENT_RATE:.0%} of occurrences, seed={AUGMENT_SEED}")
    augmented_utterances = augment_with_spellout(
        utterances, AUGMENT_RATE, AUGMENT_SEED, split_fn=lambda t: t.split("-")
    )

    write_train_mmap(tokenizer, augmented_utterances, OUT_DIR, "syllable", dtype=DTYPE)
