"""Learn N_MERGES word-level BPE merges (ngram/bpe.py's BPETokeniser, not subword BPE)
on LibriSpeech 960h training `words`-tier sentences, save the merges plus the resulting
vocabulary, apply random character-spelling augmentation to a copy of the training data,
then dump a flat token-id mmap from the augmented data.

N_MERGES is a command-line argument (--n-merges, default 200), not a constant to edit by
hand: unlike llm/subword_bpe_prepare_mmap.py (one fixed vocab size) or
ngram/ngram_bpe_sweep.py (a batch sweep across merge counts in one run), this script is
meant to be re-run by hand once per merge count of interest (200/1000/5000, matching
ngram/ngram_bpe_sweep.py's own tested values), so a flag avoids editing the file -- and
risking forgetting to -- before each run. Every output filename embeds N_MERGES
(data/word-bpe-mmap/word-bpe{n}-*, checkpoints/opt125m-word-bpe{n}-librispeech/ from
llm/word_bpe_train.py), so different merge counts' outputs never collide.

Vocabulary and backoff: see `WordBpeTokenizer`'s docstring in llm/tokenisers.py. In
short: every unit `BPETokeniser.encode()` can produce over the training corpus (every
training word type plus the learned merges), computed once here (not recomputed on each
load); anything else is a genuinely unseen word, spelled out as its own characters.

Random augmentation: independently for AUGMENT_RATE of all *word* occurrences in the
*training* data only, regardless of whether the word is in-vocab, replace the word with
its own characters before BPE segmentation. Applied at the word level, not the BPE-unit
level, because word-BPE's OOV/backoff granularity is always a single word (merges only
ever fire on pairs of *known* training words, so an OOV case never involves a multi-word
merged unit; see `WordBpeTokenizer`'s docstring) -- the same granularity the
deterministic backoff rule itself already operates on. Spelled-out characters are
re-joined into the sentence before `BPETokeniser.encode()` runs on it, exactly like a
genuine sentence would be, so they can in principle interact with BPE's own merge
algorithm (e.g. a lone augmented character could coincidentally match one side of a
learned merge with its neighbour); this is a rare, low-consequence edge case, not fixed
here, since avoiding it entirely would mean bypassing `WordBpeTokenizer.encode()`'s
normal segmentation path. Validation/evaluation never applies this:
`WordBpeTokenizer.encode()` only ever runs the deterministic in-vocab/spell-out rule,
identically at train and test time.
"""

import argparse
from pathlib import Path

import numpy as np

from data_common import TRAIN_SPLITS, augment_with_spellout, read_tier, write_train_mmap
from tokenisers import WordBpeTokenizer

OUT_DIR = Path(__file__).resolve().parent / "data" / "word-bpe-mmap"
# Uint32, not every other unit's uint16. At --min-count 2 the vocabulary is 58,206 to
# 63,002 across the tested merge counts, which would fit uint16's 65,535 ceiling, but a
# lower threshold or a higher merge count pushes it over. The existing mmaps and
# checkpoints are uint32, so changing it would mean regenerating and retraining.
DTYPE = np.uint32
AUGMENT_RATE = 0.02
AUGMENT_SEED = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-merges",
        type=int,
        default=200,
        help="Number of word-level BPE merges to learn (default: 200; "
        "ngram/ngram_bpe_sweep.py also tested 1000 and 5000)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Keep only BPE units occurring at least this many times in "
        "training (default: 2, matching word/syllable's MIN_COUNT so "
        "word-BPE's vocabulary is comparable with word's). Dropping the "
        "threshold costs 4.99 bits/s at 200 merges. Changing this changes the "
        "vocabulary, so the mmap must be regenerated and the model retrained.",
    )
    args = parser.parse_args()
    n_merges = args.n_merges
    unit_name = f"word-bpe{n_merges}"
    saved_path = OUT_DIR / f"{unit_name}-vocab.json"

    print("Reading words tier for:", TRAIN_SPLITS)
    utterances = read_tier(TRAIN_SPLITS, "words")
    print(f"Utterances: {len(utterances):,}")

    train_sentences = [" ".join(tokens) for tokens, _ in utterances.values()]

    print(f"Learning {n_merges} word-level BPE merges...", flush=True)
    bpe, vocab = WordBpeTokenizer.train_new(
        train_sentences, n_merges, min_count=args.min_count
    )
    print(
        f"Learned {len(bpe.merges)} merges; vocabulary {len(vocab):,} units "
        f"(min_count={args.min_count})"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WordBpeTokenizer.save(bpe, vocab, saved_path)
    print("Saved merges + vocab to:", saved_path)

    tokenizer = WordBpeTokenizer(saved_path)
    assert tokenizer.vocab_size < np.iinfo(DTYPE).max
    print(
        f"Full tokeniser vocab (BPE units + characters + specials): {tokenizer.vocab_size:,}"
    )

    print(
        f"Random augmentation: {AUGMENT_RATE:.0%} of occurrences, seed={AUGMENT_SEED}"
    )
    augmented_utterances = augment_with_spellout(
        utterances, AUGMENT_RATE, AUGMENT_SEED, split_fn=list
    )

    write_train_mmap(tokenizer, augmented_utterances, OUT_DIR, unit_name, dtype=DTYPE)
