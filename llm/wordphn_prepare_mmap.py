"""Train a wordphn vocabulary (hybrid wordphn/phone backoff, min-count 2) on LibriSpeech
960h training word-phone sentences, save it, apply random phone-spelling augmentation to
a copy of the training data, then dump a flat token-id mmap from the augmented data.

The wordphn unit is a whole word identified by its *realised phones* rather than its
spelling: "abbey" is `AE-B-IY`. No TextGrid carries such a tier, so
`data_common.word_phone_intervals` derives it by grouping the `phones` tier under the
`words` tier's own time boundaries (see that function for the grouping rule and why the
word interval's times are kept). Everything downstream is then the syllable unit's
pipeline unchanged, since the token format is the same: destressed ARPABET joined by
"-", backing off to `~`-marked phones.

Structurally identical to llm/syllable_prepare_mmap.py (and so to
llm/word_prepare_mmap.py, whose docstring has the full reasoning behind the vocabulary
rule and the augmentation mechanics decision); the only differences are the derived tier
and `WordPhoneTokenizer` in place of `SyllableTokenizer`.

Vocabulary: wordphn types occurring at least MIN_COUNT times in training data only
(`data_common.build_min_count_vocab`). Rarer training types and unseen dev-clean
realisations are spelled out as phones by `WordPhoneTokenizer` itself (no separate
`<unk>` case).

Random augmentation is deliberately handled here, not inside `WordPhoneTokenizer`:
independently for AUGMENT_RATE of *all* wordphn occurrences in the *training* data only
(regardless of whether the unit is in-vocab), replace the unit with its own phones before
encoding. Validation/evaluation never applies this: `WordPhoneTokenizer.encode()` only
ever runs the deterministic in-vocab/spell-out rule, identically at train and test time.
"""

from pathlib import Path

import numpy as np

from data_common import (
    TRAIN_SPLITS,
    WORDPHN_TIER,
    augment_with_spellout,
    build_min_count_vocab,
    read_tier,
    write_train_mmap,
)
from tokenisers import WordPhoneTokenizer

OUT_DIR = Path(__file__).resolve().parent / "data" / "wordphn-mmap"
VOCAB_PATH = OUT_DIR / "vocab.json"
MIN_COUNT = 2
AUGMENT_RATE = 0.02
AUGMENT_SEED = 0
DTYPE = np.uint16


if __name__ == "__main__":
    print("Deriving word-phone units for:", TRAIN_SPLITS)
    utterances = read_tier(TRAIN_SPLITS, WORDPHN_TIER)
    print(f"Utterances: {len(utterances):,}")

    wordphn_vocab = build_min_count_vocab(
        [tokens for tokens, _ in utterances.values()], min_count=MIN_COUNT
    )
    print(f"Wordphn vocab (count >= {MIN_COUNT}): {len(wordphn_vocab):,} types")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WordPhoneTokenizer.save_vocab(wordphn_vocab, VOCAB_PATH)
    print("Saved vocab to:", VOCAB_PATH)

    tokenizer = WordPhoneTokenizer(VOCAB_PATH)
    assert tokenizer.vocab_size < np.iinfo(DTYPE).max
    print(f"Full tokeniser vocab (wordphn + phones + specials): {tokenizer.vocab_size:,}")

    print(f"Random augmentation: {AUGMENT_RATE:.0%} of occurrences, seed={AUGMENT_SEED}")
    augmented_utterances = augment_with_spellout(
        utterances, AUGMENT_RATE, AUGMENT_SEED, split_fn=lambda t: t.split("-")
    )

    write_train_mmap(tokenizer, augmented_utterances, OUT_DIR, "wordphn", dtype=DTYPE)
