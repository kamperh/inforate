"""
Word-level BPE n-gram sweep over LibriSpeech 960h fractions, a direct parallel to
ngram/ngram_word_sweep.py.

BPE here is applied at the *word* level: repeatedly merging the most frequent adjacent
pair of units (starting from whole words) into a single unit, so a "word-BPE" token is a
tuple of one or more whole words (e.g. "of the"), not a subword piece in the usual NLP
sense.

For each of 25/50/75/100% of pooled LibriSpeech 960h training data (nested prefixes of
one fixed shuffle), --n-merges merges are learned from *that fraction's own* training
subset only (never from dev-clean, and never reused across fractions, per the
"tokenisers are factories re-fit per fold/split" design principle), then for each n-gram
order 1-5, fits `AbsoluteDiscountingLM` on the BPE-tokenised training subset, scores
every dev-clean utterance (out-of-vocabulary units are spelled out via `SpellingLM` over
*characters*, not their constituent words, so nothing is skipped), and reports bit rate
and perplexity from the same per-utterance bit totals.

Shared sweep machinery (data loading, fitting/scoring, table formatting) lives in
`ngram/sweep_common.py`. Unlike the other three sweeps, tokenisation here can't be a
fixed pre-split, since merges (and hence encoding) depend on the fraction: BPE
learning/encoding happens once per fraction in `main`.

Vocabulary at every cell is built from that cell's training subset only, never from
dev-clean.
"""

import argparse
from pathlib import Path

from bpe import BPETokeniser
from sweep_common import (
    DATA_ROOT,
    DEV_SPLIT,
    FRACTIONS,
    TRAIN_SPLITS,
    format_table,
    load_utterances,
    pooled_train_keys,
    run_order_sweep,
)

DEFAULT_N_MERGES = 5000


def spelling_of(token):
    """Word-BPE unit -> characters, matching `ngram_word_sweep.py`."""
    return list(token)


def write_report(bitrate_table, ppl_table, discount_table, n_train_total, n_merges):
    lines = [
        "# Word-BPE n-gram sweep on LibriSpeech 960h\n",
        f"Pooled and shuffled (seed=0) {n_train_total:,} training utterances; "
        "fractions are nested prefixes of that one shuffle. "
        f"{n_merges} word-level BPE merges are learned fresh from each "
        "fraction's own training subset only (never from dev-clean, never "
        "reused across fractions). Vocabulary is built from each cell's "
        "training subset only. Evaluated on dev-clean.\n",
        format_table("Bit rate (bits/s)", bitrate_table, lambda v: f"{v:.2f}"),
        format_table(
            "Perplexity (per word-BPE unit, not comparable across fractions)",
            ppl_table,
            lambda v: f"{v:.2f}",
        ),
        format_table(
            "Top-order discount used (Chen & Goodman; see design.md)",
            discount_table,
            lambda v: f"{v:.2f}",
        ),
    ]
    out_path = Path(__file__).resolve().parent / f"ngram_bpe{n_merges}_sweep.md"
    print("Writing:", out_path)
    out_path.write_text("\n".join(lines))


def main(n_merges):
    print("Reading:", DATA_ROOT)
    train_utterances = load_utterances(TRAIN_SPLITS, "words")
    dev_utterances = load_utterances([DEV_SPLIT], "words")
    print(f"Pooled train utterances: {len(train_utterances):,}")
    print(f"Dev-clean utterances: {len(dev_utterances):,}")

    keys = pooled_train_keys(train_utterances)
    n_total = len(keys)

    bitrate_table = {}
    ppl_table = {}
    discount_table = {}
    for fraction in FRACTIONS:
        n_take = int(round(fraction * n_total))
        subset_keys = keys[:n_take]
        train_sentences = [train_utterances[k][0] for k in subset_keys]
        print(f"\nFraction {fraction:.0%}: {len(subset_keys):,} utterances")

        print(f"  learning {n_merges} BPE merges...", flush=True)
        tokeniser = BPETokeniser(n_merges).learn(train_sentences)
        print(f"  learned {len(tokeniser.merges)} merges")

        train_token_seqs = [tokeniser.encode(s) for s in train_sentences]
        dev_token_seqs = {
            key: (tokeniser.encode(sentence), duration)
            for key, (sentence, duration) in dev_utterances.items()
        }
        n_dev_tokens = sum(len(toks) for toks, _ in dev_token_seqs.values())

        run_order_sweep(
            fraction,
            train_token_seqs,
            dev_token_seqs,
            spelling_of,
            n_dev_tokens,
            "BPE",
            bitrate_table,
            ppl_table,
            discount_table,
        )

    write_report(bitrate_table, ppl_table, discount_table, n_total, n_merges)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-merges",
        type=int,
        default=DEFAULT_N_MERGES,
        help=f"Number of word-level BPE merges (default: {DEFAULT_N_MERGES}). "
        "Was a module constant edited by hand before each run; a flag matches "
        "llm/word_bpe_prepare_mmap.py and removes the risk of forgetting. "
        "Writes ngram/ngram_bpe<n>_sweep.md.",
    )
    main(parser.parse_args().n_merges)
