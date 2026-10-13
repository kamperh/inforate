"""
Wordphn-BPE n-gram sweep over LibriSpeech 960h fractions, standing to
ngram/ngram_wordphn_sweep.py exactly as ngram/ngram_bpe_sweep.py stands to
ngram/ngram_word_sweep.py.

BPE is applied at the whole-unit level, with `wordphn` units as the atomic symbols, so a
unit here is a collocation written out as phones: "of the" becomes `AH-V DH-AH`. The two
separators do not collide -- a space joins the words inside a merged unit and "-" joins
the phones inside a word, and a realisation never contains a space -- so `BPETokeniser`
runs unmodified, as it does in the neural stage.

Merges are chosen over *realisations*, which cuts both ways: pair counts pool across
homophones ("to the", "too the" and "two the" are all `T-UW DH-AH`) and dilute across
pronunciation variants ("of the" competes as both `AH-V DH-AH` and `AH-V DH-IY`).

For each of 25/50/75/100% of pooled LibriSpeech 960h training data (nested prefixes of
one fixed shuffle), --n-merges merges are learned from *that fraction's own* training
subset only (never from dev-clean, and never reused across fractions, per the
"tokenisers are factories re-fit per fold/split" design principle), then for each n-gram
order 1-5, fits `AbsoluteDiscountingLM` on the BPE-tokenised training subset, scores
every dev-clean utterance (out-of-vocabulary units are spelled out via `SpellingLM` over
*phones*, see `spelling_of`), and reports bit rate and perplexity from the same
per-utterance bit totals.

Shared sweep machinery lives in `ngram/sweep_common.py`. As in `ngram_bpe_sweep.py`,
tokenisation cannot be a fixed pre-split, since merges (and hence encoding) depend on
the fraction: BPE learning and encoding happen once per fraction in `main`.

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
    WORDPHN_TIER,
    format_table,
    load_utterances,
    pooled_train_keys,
    run_order_sweep,
)

DEFAULT_N_MERGES = 200
WORD_BOUNDARY = " "  # Separates two realisations inside a merged unit


def spelling_of(token):
    """Wordphn-BPE unit -> phones, keeping the word boundary as its own symbol.

    The exact analogue of `ngram_bpe_sweep.py`'s `list(token)`, which flattens a merged
    unit to its finest subunits and keeps the space between words as just another
    character. Dropping the boundary instead would make `AH-V DH-AH` and the
    single-word `AH-V-DH-AH` spell out identically, which is the invalid-bound trap
    design.md records under "Per-unit spelling": a spell-out must transmit enough to
    identify the unit uniquely.

    In practice the boundary branch never fires. An out-of-vocabulary unit is always a
    single never-merged unseen realisation, because merges fire only on pairs of
    realisations already seen in training and `ngram/` applies no minimum-count
    threshold, so a merged unit is in vocabulary by construction. Kept anyway, since the
    rule should be correct rather than only correct for this corpus.
    """
    pieces = []
    for i, realisation in enumerate(token.split(WORD_BOUNDARY)):
        if i:
            pieces.append(WORD_BOUNDARY)
        pieces.extend(realisation.split("-"))
    return pieces


def write_report(bitrate_table, ppl_table, discount_table, n_train_total, n_merges):
    lines = [
        "# Wordphn-BPE n-gram sweep on LibriSpeech 960h\n",
        f"Pooled and shuffled (seed=0) {n_train_total:,} training utterances; "
        "fractions are nested prefixes of that one shuffle. "
        f"{n_merges} whole-unit BPE merges are learned fresh from each "
        "fraction's own training subset only (never from dev-clean, never "
        "reused across fractions), with `wordphn` units as the atomic symbols, "
        'so a unit is a collocation written as phones ("of the" is '
        "`AH-V DH-AH`). Vocabulary is built from each cell's training subset "
        'only. Evaluated on dev-clean. See design.md, "The wordphn-BPE unit".\n',
        format_table("Bit rate (bits/s)", bitrate_table, lambda v: f"{v:.2f}"),
        format_table(
            "Perplexity (per unit, not comparable across fractions)",
            ppl_table,
            lambda v: f"{v:.2f}",
        ),
        format_table(
            "Top-order discount used (Chen & Goodman; see design.md)",
            discount_table,
            lambda v: f"{v:.2f}",
        ),
    ]
    out_path = Path(__file__).resolve().parent / f"ngram_wordphn_bpe{n_merges}_sweep.md"
    print("Writing:", out_path)
    out_path.write_text("\n".join(lines))


def main(n_merges):
    print("Reading:", DATA_ROOT)
    train_utterances = load_utterances(TRAIN_SPLITS, WORDPHN_TIER)
    dev_utterances = load_utterances([DEV_SPLIT], WORDPHN_TIER)
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
            "wordphn-BPE",
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
        help=f"Number of whole-unit BPE merges (default: {DEFAULT_N_MERGES}). "
        "Run at 200, 1000 and 5000, matching ngram_bpe_sweep.py and the neural "
        "stage. Writes ngram/ngram_wordphn_bpe<n>_sweep.md.",
    )
    main(parser.parse_args().n_merges)
