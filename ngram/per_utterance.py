"""Dump per-utterance bits for the n-gram stage, so `utils/bootstrap.py` can put
confidence intervals on the results.

`sweep_common.fit_and_score` already computes `{key: (bits, duration)}` for
every dev-clean utterance; the sweeps just aggregate it away through
`ngram_lm.bitrate` and keep only the table cell. This script re-fits the one
cell every headline claim in `results.md` actually uses -- **order 3 at 100%
data** -- for each unit, and writes the per-utterance dict out instead.

Only that cell, deliberately: each (unit, order) pair is a separate fit, so
covering all five orders would be five times the compute for intervals on
comparisons nobody is making. Bit rates here must reproduce the order-3 100%
cells in `results.md` exactly, which is the check that the dumped numbers
describe the same quantity the tables report.

Why bootstrap rather than seeds for this stage: an n-gram fit is deterministic
given its training data. There is no initialisation and no SGD, so at 100% data
re-running with a different seed changes nothing, and resampling the evaluation
set is the only available source of error bars.
"""

import argparse
import json
from functools import partial
from pathlib import Path

from bpe import BPETokeniser, SubwordPhoneXBPETokeniser
from librispeech_stats import destress
from ngram_bpe_sweep import spelling_of as bpe_spelling_of
from ngram_lm import bitrate, build_spelling_lm
from ngram_phone_sweep import spelling_of as phone_spelling_of
from ngram_subwordphnx_bpe_sweep import merges_for
from ngram_subwordphnx_bpe_sweep import spelling_of as subwordphnx_bpe_spelling_of
from ngram_syllable_sweep import spelling_of as syllable_spelling_of
from ngram_word_sweep import spelling_of as word_spelling_of
from ngram_wordphn_bpe_sweep import spelling_of as wordphn_bpe_spelling_of
from ngram_wordphn_sweep import spelling_of as wordphn_spelling_of
from sweep_common import (
    DEV_SPLIT,
    TRAIN_SPLITS,
    WORDPHN_TIER,
    fit_and_score,
    load_utterances,
    pooled_train_keys,
    tokenize_dev,
    tokenize_train_subset,
)

OUT_DIR = Path(__file__).resolve().parent / "per_utterance"
ORDER = 3
FRACTION = 1.0

# (tier, token_transform, spelling_of, tokeniser factory). `spelling_of` is imported
# from each sweep rather than restated, so the two can never diverge: the sweeps'
# `main()` is guarded, so importing costs nothing. A second copy here would be
# a fresh way for a spell-out rule to drift out of step, which has bitten this
# project before.
#
# The factory is `None` for a unit whose tier is already tokenised, and otherwise a
# zero-argument callable giving a *fresh* tokeniser, since merges are re-fit per
# fraction. Which class it builds is part of the unit: the two BPE families merge whole
# words upwards into collocations, subwordphnx-BPE merges phones upwards into subword
# pieces.
UNITS = {
    "word": ("words", None, word_spelling_of, None),
    "syllable": ("syllables", None, syllable_spelling_of, None),
    "phone": ("phones", destress, phone_spelling_of, None),
    "word-bpe-200": ("words", None, bpe_spelling_of, partial(BPETokeniser, 200)),
    "word-bpe-1000": ("words", None, bpe_spelling_of, partial(BPETokeniser, 1000)),
    "word-bpe-5000": ("words", None, bpe_spelling_of, partial(BPETokeniser, 5000)),
    "wordphn": (WORDPHN_TIER, None, wordphn_spelling_of, None),
    "wordphn-bpe-200": (
        WORDPHN_TIER, None, wordphn_bpe_spelling_of, partial(BPETokeniser, 200)
    ),
    "wordphn-bpe-1000": (
        WORDPHN_TIER, None, wordphn_bpe_spelling_of, partial(BPETokeniser, 1000)
    ),
    "wordphn-bpe-5000": (
        WORDPHN_TIER, None, wordphn_bpe_spelling_of, partial(BPETokeniser, 5000)
    ),
    # One entry per vocabulary size, built in a loop rather than written out four
    # times: these differ only in the merge count, as llm/eval_bitrate.py's own
    # subwordphnx entries do
    **{
        f"subwordphnx-bpe-{v}": (
            WORDPHN_TIER,
            None,
            subwordphnx_bpe_spelling_of,
            partial(SubwordPhoneXBPETokeniser, merges_for(v)),
        )
        for v in (200, 2000, 5000, 50000)
    },
}


def tokenise(unit, train_utterances, dev_utterances, subset_keys):
    """(train token sequences, {key: (tokens, duration)}) for one unit.

    Word/syllable/phone/wordphn are a plain whitespace split of the
    already-tokenised tier, exactly `tokenize_train_subset`/`tokenize_dev`. Every BPE
    unit instead learns its merges from this fraction's own training subset first,
    matching `ngram_bpe_sweep.main`'s "re-fit per fold" discipline. The word-BPE and
    wordphn-BPE families share one `BPETokeniser`, since what differs is only the
    stream it is given: words for one, realisations for the other. subwordphnx-BPE
    uses `SubwordPhoneXBPETokeniser`, which merges in the other direction.
    """
    factory = UNITS[unit][3]
    if factory is None:
        train_seqs = tokenize_train_subset(train_utterances, subset_keys)
        dev_seqs, _ = tokenize_dev(dev_utterances)
        return train_seqs, dev_seqs

    train_sentences = [train_utterances[k][0] for k in subset_keys]
    tokeniser = factory()
    print(f"  learning {tokeniser.n_merges:,} {tokeniser.name} merges...", flush=True)
    tokeniser.learn(train_sentences)
    train_seqs = [tokeniser.encode(s) for s in train_sentences]
    dev_seqs = {
        key: (tokeniser.encode(sentence), duration)
        for key, (sentence, duration) in dev_utterances.items()
    }
    return train_seqs, dev_seqs


def main(units):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for unit in units:
        tier, transform, spelling_of, _ = UNITS[unit]
        print(f"\n=== {unit} (order {ORDER}, {FRACTION:.0%}) ===", flush=True)
        train_utterances = load_utterances(TRAIN_SPLITS, tier, transform)
        dev_utterances = load_utterances([DEV_SPLIT], tier, transform)
        keys = pooled_train_keys(train_utterances)
        subset_keys = keys[: int(round(FRACTION * len(keys)))]

        train_seqs, dev_seqs = tokenise(
            unit, train_utterances, dev_utterances, subset_keys
        )

        vocab = set()
        for tokens in train_seqs:
            vocab.update(tokens)
        spelling_lm = build_spelling_lm(vocab, spelling_of, order=3)

        print(f"  fitting order-{ORDER} LM...", flush=True)
        results, _, _ = fit_and_score(
            ORDER, train_seqs, dev_seqs, spelling_of, spelling_lm
        )

        out = OUT_DIR / f"{unit}-order{ORDER}-{FRACTION:.0%}.json"
        out.write_text(json.dumps({k: [b, d] for k, (b, d) in results.items()}))
        print(f"  bit rate {bitrate(results):.4f} bits/s -> {out.name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--units",
        nargs="+",
        choices=sorted(UNITS),
        default=sorted(UNITS),
        help="Which units to refit and dump (default: all). A fit is deterministic "
        "given its training data, so a dump never needs regenerating unless the unit's "
        "tokenisation or spell-out rule changes; naming a subset avoids spending an "
        "hour recomputing dumps that are already on disk.",
    )
    main(parser.parse_args().units)
