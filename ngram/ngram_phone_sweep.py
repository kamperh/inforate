"""
Phone-level n-gram sweep over LibriSpeech 960h fractions, a direct parallel to
ngram/ngram_word_sweep.py.

For each of 25/50/75/100% of pooled LibriSpeech 960h training data (nested prefixes of
one fixed shuffle) and each n-gram order 1-5, fits `AbsoluteDiscountingLM` on
phone-tokenised sentences (reconstructed from the `phones` tier, stress digits
stripped), scores every dev-clean utterance (out-of-vocabulary phones are spelled out
via `SpellingLM` over their own letters, so nothing is skipped, though this should
essentially never fire given the closed inventory), and reports bit rate and perplexity
from the same per-utterance bit totals.

**Unstressed phones.** The raw `phones` tier carries ARPABET stress digits
(`AH0`/`AH1`/`AH2`, 69 symbols); stress is stripped here to the standard 39-symbol set.
The `syllables` tier is already destressed at the source, so the syllable sweep's
phone-level spelling model uses the 39-symbol set too; keeping the stressed set here
would give "phone" two different meanings across sweeps.

Shared sweep machinery (data loading, fitting/scoring, table formatting) lives in
`ngram/sweep_common.py`; only tokenisation and `spelling_of` are specific to this
script. Vocabulary at every cell is built from that cell's training subset only, never
from dev-clean.
"""

from pathlib import Path

from librispeech_stats import destress
from sweep_common import (
    DATA_ROOT,
    DEV_SPLIT,
    FRACTIONS,
    TRAIN_SPLITS,
    format_table,
    load_utterances,
    pooled_train_keys,
    run_order_sweep,
    tokenize_dev,
    tokenize_train_subset,
)

OUT_PATH = Path(__file__).resolve().parent / "ngram_phone_sweep.md"


def spelling_of(token):
    """Phone -> its own letters; a safety net expected never to fire, given the closed
    39-symbol inventory.
    """
    return list(token)


def write_report(bitrate_table, ppl_table, discount_table, n_train_total):
    lines = [
        "# Phone-level n-gram sweep on LibriSpeech 960h\n",
        f"Pooled and shuffled (seed=0) {n_train_total:,} training utterances; "
        "fractions are nested prefixes of that one shuffle. Vocabulary is built "
        "from each cell's training subset only. Evaluated on dev-clean. Phones "
        "are unstressed (standard 39-symbol ARPABET set); see design.md, 'Data'.\n",
        format_table("Bit rate (bits/s)", bitrate_table, lambda v: f"{v:.2f}"),
        format_table("Perplexity (per phone)", ppl_table, lambda v: f"{v:.2f}"),
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
    train_utterances = load_utterances(TRAIN_SPLITS, "phones", destress)
    dev_utterances = load_utterances([DEV_SPLIT], "phones", destress)
    print(f"Pooled train utterances: {len(train_utterances):,}")
    print(f"Dev-clean utterances: {len(dev_utterances):,}")

    dev_token_seqs, n_dev_phone_tokens = tokenize_dev(dev_utterances)

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
            n_dev_phone_tokens,
            "phone",
            bitrate_table,
            ppl_table,
            discount_table,
        )

    write_report(bitrate_table, ppl_table, discount_table, n_total)


if __name__ == "__main__":
    main()
