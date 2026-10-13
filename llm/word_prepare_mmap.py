"""Train a word vocabulary (hybrid word/character backoff, min-count 2) on LibriSpeech
960h training `words`-tier sentences, save it, apply random character-spelling
augmentation to a copy of the training data, then dump a flat token-id mmap from the
augmented data.

Vocabulary: word types occurring at least MIN_COUNT times in training data only
(`data_common.build_min_count_vocab`). Rarer training types and unseen dev-clean words
are spelled out as characters by `WordTokenizer` itself (no separate `<unk>` case; see
llm/tokenisers.py).

Random augmentation is deliberately handled here, not inside `WordTokenizer`:
independently for AUGMENT_RATE of *all* word occurrences in the *training* data only
(regardless of whether the word is in-vocab), replace the word with its own characters
before encoding, so the model gets meaningfully more exposure to the character fallback
pathway than the ~0.33%-of-tokens deterministic OOV rate alone would give it. Applied
once here, when the mmap is built, not re-rolled per training epoch: re-rolling would
need on-the-fly retokenisation during training, which this project's flat pre-built mmap
pipeline (`data.TokenizedUnitsChunkedMMapDataset`) doesn't support, and a one-time
random split still gives the fallback pathway far more examples than the deterministic
rule alone. Validation/evaluation never applies this: `WordTokenizer.encode()` only ever
runs the deterministic in-vocab/spell-out rule, identically at train and test time.
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
from tokenisers import WordTokenizer

OUT_DIR = Path(__file__).resolve().parent / "data" / "word-mmap"
VOCAB_PATH = OUT_DIR / "vocab.json"
MIN_COUNT = 2
AUGMENT_RATE = 0.02
AUGMENT_SEED = 0
DTYPE = np.uint16


if __name__ == "__main__":
    print("Reading words tier for:", TRAIN_SPLITS)
    utterances = read_tier(TRAIN_SPLITS, "words")
    print(f"Utterances: {len(utterances):,}")

    word_vocab = build_min_count_vocab(
        [tokens for tokens, _ in utterances.values()], min_count=MIN_COUNT
    )
    print(f"Word vocab (count >= {MIN_COUNT}): {len(word_vocab):,} types")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WordTokenizer.save_vocab(word_vocab, VOCAB_PATH)
    print("Saved vocab to:", VOCAB_PATH)

    tokenizer = WordTokenizer(VOCAB_PATH)
    assert tokenizer.vocab_size < np.iinfo(DTYPE).max
    print(
        f"Full tokeniser vocab (words + characters + specials): {tokenizer.vocab_size:,}"
    )

    print(
        f"Random augmentation: {AUGMENT_RATE:.0%} of occurrences, seed={AUGMENT_SEED}"
    )
    augmented_utterances = augment_with_spellout(
        utterances, AUGMENT_RATE, AUGMENT_SEED, split_fn=list
    )

    write_train_mmap(tokenizer, augmented_utterances, OUT_DIR, "word", dtype=DTYPE)
