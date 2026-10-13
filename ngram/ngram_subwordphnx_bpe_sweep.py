"""
Subwordphnx-BPE n-gram sweep over LibriSpeech 960h fractions, the n-gram counterpart of
the neural stage's `subwordphnx-BPE` unit.

Subword BPE with *phones* as the atomic symbols and every merge confined to a single
word, so a piece is a run of phones inside one word, written as "-"-joined phones
("the" comes out as the single piece `DH-AH`). This is subword BPE in the usual NLP
sense, unlike `ngram_bpe_sweep.py` and `ngram_wordphn_bpe_sweep.py`, which merge whole
words into collocations: those two grow units upwards from words, this one grows them
upwards from phones. Both arrive at roughly word-sized units from opposite directions.

Reads the derived `wordphones` tier, since that is where the word grouping needed to
confine a merge lives. The phones are the same phones the phone sweep reads and the
durations come from the `words` tier untouched, so this unit's denominator is
bit-identical to the word, wordphn and wordphn-BPE sweeps' and every bit rate stays
directly comparable.

For each of 25/50/75/100% of pooled LibriSpeech 960h training data (nested prefixes of
one fixed shuffle), the merges are learned from *that fraction's own* training subset
only (never from dev-clean, and never reused across fractions, per the "tokenisers are
factories re-fit per fold/split" design principle), then for each n-gram order 1-5,
fits `AbsoluteDiscountingLM` on the piece-tokenised training subset, scores every
dev-clean utterance, and reports bit rate and perplexity from the same per-utterance
bit totals.

**Nothing can go out of vocabulary**, as for `phone` and for the neural unit: every
phone is an atomic symbol, and a piece is either a single phone or the result of a merge
learned on this fraction's training data, so a dev piece is in vocabulary by
construction. `spelling_of` is the same kind of safety net `ngram_phone_sweep.py` keeps.

Shared sweep machinery lives in `ngram/sweep_common.py`. As in `ngram_bpe_sweep.py`,
tokenisation cannot be a fixed pre-split, since the merges (and hence the encoding)
depend on the fraction: BPE learning and encoding happen once per fraction in `main`.

Vocabulary at every cell is built from that cell's training subset only, never from
dev-clean.
"""

import argparse
from pathlib import Path

from bpe import SubwordPhoneXBPETokeniser
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

DEFAULT_VOCAB_SIZE = 2000
# What the neural unit's vocabulary budget spends on things that are not merges: the 39
# destressed ARPABET phones that make up the alphabet, and the three special tokens
# HuggingFace's `BpeTrainer` reserves (<bos>, <pad>, <unk>). Subtracting both makes
# --vocab-size mean the same number of merges here as there -- 1,958 at 2,000 -- even
# though this stage has no special tokens of its own to reserve
N_ALPHABET = 39
N_SPECIALS = 3


def merges_for(vocab_size):
    """Merge count matching the neural unit at the same `--vocab-size`."""
    return vocab_size - N_ALPHABET - N_SPECIALS


def spelling_of(token):
    """Piece -> its own phones; a safety net expected never to fire.

    A piece *is* its phone sequence, so this transmits enough to identify the piece
    uniquely, which is what `design.md`, "Per-unit spelling", requires of a spell-out
    rule. It cannot fire: every phone is an atomic symbol and every merged piece was
    learned from this fraction's own training data.
    """
    return token.split("-")


def write_report(bitrate_table, ppl_table, discount_table, n_train_total, vocab_size):
    n_merges = merges_for(vocab_size)
    lines = [
        "# Subwordphnx-BPE n-gram sweep on LibriSpeech 960h\n",
        f"Pooled and shuffled (seed=0) {n_train_total:,} training utterances; "
        "fractions are nested prefixes of that one shuffle. "
        f"{n_merges:,} subword BPE merges over phones are learned fresh from each "
        "fraction's own training subset only (never from dev-clean, never reused "
        "across fractions), with every merge confined to a single word, so a piece "
        'is a run of phones inside one word ("the" is the single piece `DH-AH`). '
        f"That is the merge count the neural unit has at vocabulary {vocab_size:,}, "
        "which is what makes the two stages the same unit. Vocabulary is built from "
        "each cell's training subset only. Evaluated on dev-clean. See design.md, "
        '"The subwordphnx-BPE unit".\n',
        format_table("Bit rate (bits/s)", bitrate_table, lambda v: f"{v:.2f}"),
        format_table(
            "Perplexity (per piece, not comparable across fractions)",
            ppl_table,
            lambda v: f"{v:.2f}",
        ),
        format_table(
            "Top-order discount used (Chen & Goodman; see design.md)",
            discount_table,
            lambda v: f"{v:.2f}",
        ),
    ]
    out_path = (
        Path(__file__).resolve().parent
        / f"ngram_subwordphnx_bpe{vocab_size}_sweep.md"
    )
    print("Writing:", out_path)
    out_path.write_text("\n".join(lines))


def main(vocab_size):
    n_merges = merges_for(vocab_size)
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

        print(f"  learning {n_merges:,} phone BPE merges...", flush=True)
        tokeniser = SubwordPhoneXBPETokeniser(n_merges).learn(train_sentences)
        print(f"  learned {len(tokeniser.merges):,} merges")
        longest = sorted(
            ("-".join(a + b) for a, b in tokeniser.merges), key=len, reverse=True
        )[:3]
        print(f"  longest pieces: {longest}")

        train_token_seqs = [tokeniser.encode(s) for s in train_sentences]
        dev_token_seqs = {
            key: (tokeniser.encode(sentence), duration)
            for key, (sentence, duration) in dev_utterances.items()
        }
        n_dev_tokens = sum(len(toks) for toks, _ in dev_token_seqs.values())
        print(f"  dev-clean pieces: {n_dev_tokens:,}")

        run_order_sweep(
            fraction,
            train_token_seqs,
            dev_token_seqs,
            spelling_of,
            n_dev_tokens,
            "subwordphnx-BPE",
            bitrate_table,
            ppl_table,
            discount_table,
        )

    write_report(bitrate_table, ppl_table, discount_table, n_total, vocab_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help=f"Target vocabulary size (default: {DEFAULT_VOCAB_SIZE}), named to match "
        "llm/subwordphnx_bpe_train.py's own flag rather than given as a merge count, "
        "so the same number means the same unit in both stages. The merge count is "
        "this minus the 39 phones and the 3 special tokens the neural stage reserves. "
        "Writes ngram/ngram_subwordphnx_bpe<v>_sweep.md.",
    )
    main(parser.parse_args().vocab_size)
