"""
Wordphn n-gram sweep over LibriSpeech 960h fractions, a direct parallel to
ngram/ngram_word_sweep.py.

The `wordphn` unit keeps word-sized spans but identifies each one by **its realised
phones rather than its spelling**, so "abbey" is `AE-B-IY`. No TextGrid carries such a
tier, so it is derived from the `words` and `phones` tiers by
`sweep_common.load_wordphn_utterances`.

For each of 25/50/75/100% of pooled LibriSpeech 960h training data (nested prefixes of
one fixed shuffle) and each n-gram order 1-5, fits `AbsoluteDiscountingLM` on
realisation-tokenised sentences, scores every dev-clean utterance (out-of-vocabulary
realisations are spelled out via `SpellingLM` over their constituent phones, so nothing
is skipped), and reports bit rate and perplexity from the same per-utterance bit totals.

**This unit is matched to word on everything but identity.** There is one unit per word
interval and the derived intervals keep the `words` tier's own times, so the token count
and the speech-second denominator are both bit-identical to `ngram_word_sweep.py`'s. Any
difference in bit rate is about *what* the unit is, not how many there are or how long
they last, and perplexity is comparable between the two sweeps for once, since they
score the same number of tokens.

Shared sweep machinery (data loading, fitting/scoring, table formatting) lives in
`ngram/sweep_common.py`; only tokenisation and `spelling_of` are specific to this
script.

Vocabulary at every cell is built from that cell's training subset only, never from
dev-clean.
"""

from pathlib import Path

from sweep_common import (
    DATA_ROOT,
    DEV_SPLIT,
    FRACTIONS,
    TRAIN_SPLITS,
    WORDPHN_TIER,
    format_table,
    load_utterances,
    pooled_train_keys,
    run_order_sweep,
    tokenize_dev,
    tokenize_train_subset,
)

OUT_PATH = Path(__file__).resolve().parent / "ngram_wordphn_sweep.md"


def spelling_of(token):
    """Realisation -> phones; free, since the unit is already a hyphen-joined phone
    sequence. The same rule the syllable sweep uses, one level coarser.
    """
    return token.split("-")


def write_report(bitrate_table, ppl_table, discount_table, n_train_total):
    lines = [
        "# Wordphn n-gram sweep on LibriSpeech 960h\n",
        f"Pooled and shuffled (seed=0) {n_train_total:,} training utterances; "
        "fractions are nested prefixes of that one shuffle. Vocabulary is built "
        "from each cell's training subset only. Evaluated on dev-clean. A unit is "
        "one word interval written as its own destressed phones joined by `-` "
        '("abbey" is `AE-B-IY`); see design.md, "The wordphn unit".\n',
        format_table("Bit rate (bits/s)", bitrate_table, lambda v: f"{v:.2f}"),
        format_table(
            "Perplexity (per realisation; comparable with the word sweep's, which "
            "scores the same token count)",
            ppl_table,
            lambda v: f"{v:.2f}",
        ),
        format_table(
            "Top-order discount used (Chen & Goodman; see design.md)",
            discount_table,
            lambda v: f"{v:.2f}",
        ),
    ]
    print("Writing:", OUT_PATH)
    OUT_PATH.write_text("\n".join(lines))


def main():
    print("Reading:", DATA_ROOT)
    train_utterances = load_utterances(TRAIN_SPLITS, WORDPHN_TIER)
    dev_utterances = load_utterances([DEV_SPLIT], WORDPHN_TIER)
    print(f"Pooled train utterances: {len(train_utterances):,}")
    print(f"Dev-clean utterances: {len(dev_utterances):,}")

    dev_token_seqs, n_dev_tokens = tokenize_dev(dev_utterances)
    print(f"Dev-clean realisations: {n_dev_tokens:,}")

    keys = pooled_train_keys(train_utterances)
    n_total = len(keys)

    bitrate_table = {}
    ppl_table = {}
    discount_table = {}
    for fraction in FRACTIONS:
        n_take = int(round(fraction * n_total))
        subset_keys = keys[:n_take]
        train_token_seqs = tokenize_train_subset(train_utterances, subset_keys)
        print(f"\nFraction {fraction:.0%}: {len(subset_keys):,} utterances")

        run_order_sweep(
            fraction,
            train_token_seqs,
            dev_token_seqs,
            spelling_of,
            n_dev_tokens,
            "wordphn",
            bitrate_table,
            ppl_table,
            discount_table,
        )

    write_report(bitrate_table, ppl_table, discount_table, n_total)


if __name__ == "__main__":
    main()
