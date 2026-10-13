#!/usr/bin/env python

"""
Show the learned wordphn-level BPE merges and vocabulary, in the order learned.

Reads the saved tokeniser at `llm/data/wordphn-bpe-mmap/wordphn-bpe<n>-vocab.json`,
which `llm/wordphn_bpe_prepare_mmap.py` wrote. Nothing is re-learned here.

A merge joins two existing units into one, starting from whole `wordphn` units -- a
word identified by its realised phones, `AH-V` for "of" -- so a wordphn-BPE unit is a
multi-word expression written as phones (`AH-V DH-AH`, "of the"), never a sub-word
piece. Merge `i` is only meaningful given merges `1..i-1`: later merges combine units
that earlier ones created, which is why the order is worth seeing.

Realisations are hard to read on their own, so each unit is glossed with the spelling
it most often carries on dev-clean; `--no-gloss` skips that read. A glossed word marked
`*` has more than one spelling for that realisation, which is the homophone collapse
the unit is built on (`T-UW` is to, too and two).

The three merge counts are **nested**: 200's merges are the first 200 of 1000's, which
are the first 1000 of 5000's, so the levels are prefixes of one merge sequence rather
than three independent fits. `--compare` shows that and what each level adds.

`ngram/ngram_wordphn_bpe_sweep.py` learns its own merges per data fraction and discards
them rather than saving, so its merges cannot be listed directly. At 100% data it
learns from the same sentences with the same class, differing only in the order they
are fed: `ngram/` shuffles with seed 0, `llm/` reads in filesystem order. `--verify`
re-learns along that path and diffs, which is the check `design.md`, "Open issues",
records as never having been run for this unit.

Examples:

    python utils/inspect_bpe.py                               # first 30 of 200 merges
    python utils/inspect_bpe.py --n-merges 5000 --top 50
    python utils/inspect_bpe.py --n-merges 1000 --all
    python utils/inspect_bpe.py --n-merges 5000 --longest 15  # deepest merge chains
    python utils/inspect_bpe.py --compare                     # the three levels
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # utils/ -> project root
VOCAB_DIR = ROOT / "llm" / "data" / "wordphn-bpe-mmap"
LEVELS = (200, 1000, 5000)


def load(n_merges):
    """(merges, vocab) for one merge count, merges in the order learned.

    Each merge is stored as a pair of units and each unit as a list of realisations, so
    `[["AH-V"], ["DH-AH"]]` becomes `("AH-V", "DH-AH")` here.
    """
    path = VOCAB_DIR / f"wordphn-bpe{n_merges}-vocab.json"
    if not path.exists():
        raise SystemExit(
            f"No saved tokeniser at {path}.\n"
            f"Run: python llm/wordphn_bpe_prepare_mmap.py --n-merges {n_merges}"
        )
    saved = json.loads(path.read_text())
    merges = [(" ".join(a), " ".join(b)) for a, b in saved["merges"]]
    return merges, saved["vocab"]


def load_gloss(split):
    """{realisation: (most frequent spelling, how many spellings)} over one split.

    The `words` and `wordphones` tiers have one interval per word by construction (see
    `llm/data_common.py`, `word_phone_intervals`), so zipping an utterance's two token
    lists pairs each word with its own realisation.
    """
    import sys

    sys.path.insert(0, str(ROOT / "llm"))
    from data_common import WORDPHN_TIER, read_tier

    print(f"Reading {split} for glosses...", flush=True)
    words = read_tier([split], "words")
    realisations = read_tier([split], WORDPHN_TIER)
    spellings = defaultdict(Counter)
    for key, (word_tokens, _) in words.items():
        for word, realisation in zip(word_tokens, realisations[key][0]):
            spellings[realisation][word] += 1
    return {r: (c.most_common(1)[0][0], len(c)) for r, c in spellings.items()}


def gloss(unit, table):
    """`unit` ("AH-V DH-AH") written as its spellings ("of the"), or None if not
    glossing. An unseen realisation shows as `?`, one with several spellings as `word*`.
    """
    if table is None:
        return None
    out = []
    for realisation in unit.split():
        spelling, n_spellings = table.get(realisation, ("?", 1))
        out.append(spelling + "*" if n_spellings > 1 else spelling)
    return " ".join(out)


def print_merges(merges, limit, table):
    width = len(str(min(limit, len(merges))))
    for i, (left, right) in enumerate(merges[:limit], 1):
        unit = left + " " + right
        line = f"  {i:>{width}}. {left!r} + {right!r}  ->  {unit!r}"
        shown = gloss(unit, table)
        print(f"{line}   ({shown})" if shown else line)


def print_longest(vocab, count, table):
    """The units built from the most words, i.e. the deepest merge chains."""
    multi = sorted(
        (u for u in vocab if " " in u),
        key=lambda u: (len(u.split()), u),
        reverse=True,
    )
    for unit in multi[:count]:
        line = f"  {len(unit.split())} words: {unit!r}"
        shown = gloss(unit, table)
        print(f"{line}   ({shown})" if shown else line)


def compare(table):
    """The three merge counts side by side, and what each level adds over the one
    below. The levels are nested, so this is one merge sequence read at three depths.
    """
    levels = {n: load(n) for n in LEVELS}
    print(f"{'level':>6}  {'merges':>6}  {'vocabulary':>10}  {'multi-word':>10}  "
          f"{'deepest':>7}")
    for n, (merges, vocab) in levels.items():
        multi = [u for u in vocab if " " in u]
        deepest = max((len(u.split()) for u in multi), default=1)
        print(f"{n:>6}  {len(merges):>6}  {len(vocab):>10,}  {len(multi):>10,}  "
              f"{deepest:>7}")

    nested = all(
        levels[big][0][: len(levels[small][0])] == levels[small][0]
        for small, big in zip(LEVELS, LEVELS[1:])
    )
    print(f"\nNested (each level's merges are a prefix of the next): {nested}")
    if not nested:
        print("  The levels are independent fits; compare them by set, not position.")

    for small, big in zip(LEVELS, LEVELS[1:]):
        added = levels[big][0][len(levels[small][0]):]
        print(f"\nFirst merges {small} -> {big} adds ({len(added)} in all):")
        print_merges(added, 5, table)


def print_gloss_legend(table, split):
    """How to read the glosses, printed once at the end of whatever was shown."""
    if table is not None:
        print(
            f"\nGlosses are the most frequent spelling on {split}; "
            "* marks a realisation with more than one."
        )


def verify(n_merges, merges):
    """Re-learn along the n-gram path and compare, set and order."""
    import sys

    sys.path.insert(0, str(ROOT / "ngram"))
    from bpe import BPETokeniser
    from sweep_common import (
        TRAIN_SPLITS,
        WORDPHN_TIER,
        load_utterances,
        pooled_train_keys,
    )

    print("Reading wordphones tier for:", TRAIN_SPLITS, flush=True)
    utterances = load_utterances(TRAIN_SPLITS, WORDPHN_TIER)
    keys = pooled_train_keys(utterances)  # Shuffled seed 0, as the sweep does
    sentences = [utterances[k][0] for k in keys]
    print(
        f"Learning {n_merges} merges from {len(sentences):,} sentences...", flush=True
    )

    learned = [
        (" ".join(a), " ".join(b))
        for a, b in BPETokeniser(n_merges).learn(sentences).merges
    ]
    print(f"\n  same set:   {set(learned) == set(merges)}")
    print(f"  same order: {learned == merges}")
    for i, (a, b) in enumerate(zip(learned, merges), 1):
        if a != b:
            print(f"  first order difference at #{i}: ngram {a} vs llm {b}")
            break


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--n-merges",
        type=int,
        default=200,
        help="Which saved tokeniser to read (default: 200; also 1000 and 5000)",
    )
    parser.add_argument(
        "--top", type=int, default=30, help="How many merges to show (default: 30)"
    )
    parser.add_argument("--all", action="store_true", help="Show every merge")
    parser.add_argument(
        "--longest",
        type=int,
        metavar="N",
        help="Instead show the N units built from the most words",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Instead compare the three merge counts and what each level adds",
    )
    parser.add_argument(
        "--no-gloss",
        action="store_true",
        help="Skip the dev-clean read that glosses realisations as spellings",
    )
    parser.add_argument(
        "--gloss-split",
        default="dev-clean",
        help="Split to take glosses from (default: dev-clean; train-clean-100 covers "
        "more rare realisations and takes about ten times as long)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Re-learn along the n-gram path and diff; takes several minutes",
    )
    args = parser.parse_args()

    table = None if args.no_gloss else load_gloss(args.gloss_split)

    if args.compare:
        compare(table)
        print_gloss_legend(table, args.gloss_split)
        return

    merges, vocab = load(args.n_merges)
    multi = sum(1 for u in vocab if " " in u)
    print(
        f"wordphn-bpe{args.n_merges}: {len(merges)} merges, vocabulary {len(vocab):,} "
        f"({multi:,} multi-word, {len(vocab) - multi:,} single-word)\n"
    )

    if args.longest:
        print_longest(vocab, args.longest, table)
    else:
        limit = len(merges) if args.all else args.top
        print(f"Merges 1..{min(limit, len(merges))}, in the order learned:")
        print_merges(merges, limit, table)

    print_gloss_legend(table, args.gloss_split)

    if args.verify:
        print()
        verify(args.n_merges, merges)


if __name__ == "__main__":
    main()
