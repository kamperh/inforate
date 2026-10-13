"""Learn N_MERGES word-level BPE merges (ngram/bpe.py's BPETokeniser, not subword BPE)
over the *wordphn* unit rather than the orthographic word, save the merges plus the
resulting vocabulary, apply random phone-spelling augmentation to a copy of the training
data, then dump a flat token-id mmap from the augmented data.

The wordphn unit identifies a word by its realised phones instead of its spelling
("abbey" is `AE-B-IY`; see `data_common.word_phone_intervals`), so a merged unit here is
a collocation written out as phones: "of the" becomes `AH-V DH-AH`. The BPE is fitted
afresh on the wordphn stream, not inherited from llm/word_bpe_prepare_mmap.py's
word-level merges, so each unit's merges are learned on that unit's own stream, which is
what keeps the word-BPE and wordphn-BPE series parallel. Merges are therefore chosen
over *realisations*, which cuts both ways: pair counts pool across homophones ("to the",
"too the" and "two the" are all `T-UW DH-AH`) and dilute across pronunciation variants
("of the" competes as both `AH-V DH-AH` and `AH-V DH-IY`).

The two separators do not collide: a space joins the words inside a merged unit, "-"
joins the phones inside a word, and a realisation never contains a space, so
`BPETokeniser` runs unmodified.

N_MERGES is a command-line argument (--n-merges, default 200), matching
llm/word_bpe_prepare_mmap.py: the script is meant to be re-run by hand once per merge
count of interest, and every output filename embeds it
(data/wordphn-bpe-mmap/wordphn-bpe{n}-*, checkpoints/opt125m-wordphn-bpe{n}-librispeech/
from llm/wordphn_bpe_train.py), so different merge counts' outputs never collide.

Vocabulary and backoff: see `WordPhoneBpeTokenizer`'s docstring in llm/tokenisers.py. In
short: every unit `BPETokeniser.encode()` can produce over the training corpus, computed
once here (not recomputed on each load); anything else backs off to its component
wordphn units, and a component that is itself out of vocabulary is spelled out as its
own phones.

Random augmentation: independently for AUGMENT_RATE of all *wordphn* occurrences in the
*training* data only, regardless of whether the unit is in-vocab, replace the unit with
its own phones before BPE segmentation. Applied at the wordphn level, not the BPE-unit
level, because that is the granularity the deterministic backoff rule itself operates on
-- the same reasoning llm/word_bpe_prepare_mmap.py gives for augmenting at the word
level. Validation/evaluation never applies this:
`WordPhoneBpeTokenizer.encode()` only ever runs the deterministic in-vocab/back-off
rule, identically at train and test time.
"""

import argparse
from pathlib import Path

import numpy as np

from data_common import (
    TRAIN_SPLITS,
    WORDPHN_TIER,
    augment_with_spellout,
    read_tier,
    write_train_mmap,
)
from tokenisers import WordPhoneBpeTokenizer

OUT_DIR = Path(__file__).resolve().parent / "data" / "wordphn-bpe-mmap"
# Uint32, matching the word-BPE family rather than wordphn's uint16. At --min-count 2
# every merge count run here fits uint16's 65,535 ceiling (61,294 at 5000 merges), but
# a lower threshold pushes it over, and consistency inside the BPE family is worth more
# than the 20 MB this costs.
DTYPE = np.uint32
AUGMENT_RATE = 0.02
AUGMENT_SEED = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-merges",
        type=int,
        default=200,
        help="Number of wordphn-level BPE merges to learn (default: 200; the neural "
        "stage has run 200, 1000 and 5000, matching word-BPE)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Keep only BPE units occurring at least this many times in training "
        "(default: 2, matching word/wordphn/word-BPE's own threshold so the "
        "vocabularies stay comparable). Changing this changes the vocabulary, so the "
        "mmap must be regenerated and the model retrained.",
    )
    args = parser.parse_args()
    n_merges = args.n_merges
    unit_name = f"wordphn-bpe{n_merges}"
    saved_path = OUT_DIR / f"{unit_name}-vocab.json"

    print("Reading derived wordphn units for:", TRAIN_SPLITS)
    utterances = read_tier(TRAIN_SPLITS, WORDPHN_TIER)
    print(f"Utterances: {len(utterances):,}")

    train_sentences = [" ".join(tokens) for tokens, _ in utterances.values()]

    print(f"Learning {n_merges} wordphn-level BPE merges...", flush=True)
    bpe, vocab = WordPhoneBpeTokenizer.train_new(
        train_sentences, n_merges, min_count=args.min_count
    )
    print(
        f"Learned {len(bpe.merges)} merges; vocabulary {len(vocab):,} units "
        f"(min_count={args.min_count}, {sum(1 for u in vocab if ' ' in u):,} of them "
        "multi-word)"
    )
    print("First merges:")
    for a, b in bpe.merges[:8]:
        print(f"  {' '.join(a)} + {' '.join(b)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WordPhoneBpeTokenizer.save(bpe, vocab, saved_path)
    print("Saved merges + vocab to:", saved_path)

    tokenizer = WordPhoneBpeTokenizer(saved_path)
    assert tokenizer.vocab_size < np.iinfo(DTYPE).max
    print(
        "Full tokeniser vocab (BPE units + phones + specials): "
        f"{tokenizer.vocab_size:,}"
    )

    print(
        f"Random augmentation: {AUGMENT_RATE:.0%} of occurrences, "
        f"seed={AUGMENT_SEED}"
    )
    augmented_utterances = augment_with_spellout(
        utterances, AUGMENT_RATE, AUGMENT_SEED, split_fn=lambda t: t.split("-")
    )

    write_train_mmap(tokenizer, augmented_utterances, OUT_DIR, unit_name, dtype=DTYPE)
