#!/usr/bin/env python

"""
Show one LibriSpeech utterance as every unit in the project tokenises it.

Reads the utterance's forced alignment and encodes it with each neural-stage
tokeniser that `llm/<unit>_prepare_mmap.py` saved. Nothing is re-learned and no model
is loaded, so this is a seconds-long read rather than a scoring run.

The unit list, each unit's tier and each unit's tokeniser all come from
`llm/eval_bitrate.py`'s own `BASE`, so a unit cannot appear here under a tokenisation
different from the one it was scored with. `DISPLAY_ORDER` only reorders that dict for
reading, and is asserted to cover it exactly: adding a unit to `BASE` fails here until
it is placed.

What the output shows per unit is the **content** tokens. The boundary token every
utterance ends with (`<bos>`, reused as end-of-utterance; see
`llm/data.py`, `TokenizedUnitsUtteranceDataset`) is dropped, since it is one token for
every unit and carries no tokenisation information. A `~`-marked token is a spell-out
piece for an out-of-vocabulary unit, emitted pre-marked: see `design.md`,
"Random augmentation". subwordphnx-BPE pieces are mapped back from the private-use
characters its BPE merges over to readable phones.

An n-gram unit is deliberately not shown. `ngram/`'s BPE sweeps re-fit their merges per
data fraction and discard them (`ngram/per_utterance.py` does the same), so there is no
saved n-gram tokenisation to read; at 100% data the two stages learn from the same
sentences with the same class, differing only in feed order. The non-BPE units are a
plain tier split and so are identical across the stages by construction.

**Per-utterance bit rates are not printed, deliberately.** They are in the dumps under
`llm/per_utterance/` and `ngram/per_utterance/`, but one utterance is far too little to
average over: on `1272-141231-0000` the neural `phone` unit is the *best* of the fifteen
at 23.8 bits/s where over all of dev-clean it is the worst at 29.56. Quote bit rates
from `results.md`, and use this script for tokenisations and token counts only.

Examples:

    python utils/inspect_example.py                       # first dev-clean utterance
    python utils/inspect_example.py 1272-141231-0000
    python utils/inspect_example.py --words 9             # just the first nine words
    python utils/inspect_example.py --markdown --words 9  # a table to paste in a doc
    python utils/inspect_example.py --timings             # word tier with its times
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # utils/ -> project root

# The first dev-clean utterance in the order the sweeps iterate, and the one printed as
# expected output in HuggingFace's Wav2Vec2 documentation
DEFAULT_UTTERANCE = "1272-128104-0000"

# `BASE`'s keys, reordered coarse to fine rather than alphabetically: the three
# single-tier units first, then the sub-word levels, then the collocation families.
# `bars` is False only for `phone`, whose tokens are single symbols, so bars would
# treble the width of the one line that exists to show no grouping at all.
DISPLAY_ORDER = (
    "phone",
    "syllable",
    "word",
    "wordphn",
    "subword-bpe",
    "subwordphnx-bpe-200",
    "subwordphnx-bpe-2000",
    "subwordphnx-bpe-5000",
    "subwordphnx-bpe-50000",
    "word-bpe-200",
    "word-bpe-1000",
    "word-bpe-5000",
    "wordphn-bpe-200",
    "wordphn-bpe-1000",
    "wordphn-bpe-5000",
)
TIERS = ("words", "syllables", "phones")


def add_stage_paths():
    """Put `llm/` and `ngram/` on the import path, so this script can reuse their unit
    definitions instead of restating them.

    Both stages are flat package-less directories imported by their own scripts' sibling
    imports, so a path insert is how anything outside them borrows a helper, as
    `utils/inspect_bpe.py` already does for `llm/data_common.py`. Done on the way into
    `main()` rather than at module scope, so `--help` does not pay for importing torch.
    """
    sys.path.insert(0, str(ROOT / "llm"))
    sys.path.insert(0, str(ROOT / "ngram"))


def load_units():
    """`llm/eval_bitrate.py`'s `BASE`, checked against `DISPLAY_ORDER` and returned in
    that order.
    """
    from eval_bitrate import BASE

    missing = set(BASE) - set(DISPLAY_ORDER)
    unknown = set(DISPLAY_ORDER) - set(BASE)
    assert not missing, f"unplaced units in BASE: {sorted(missing)}"
    assert not unknown, f"DISPLAY_ORDER has units BASE lacks: {sorted(unknown)}"
    return {unit: BASE[unit] for unit in DISPLAY_ORDER}


def find_textgrid(utterance):
    """The one TextGrid for `utterance`, and which split it came from."""
    from data_common import DATA_ROOT

    for split in sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir()):
        hits = list((DATA_ROOT / split).rglob(f"{utterance}.TextGrid"))
        if hits:
            return hits[0], split
    raise SystemExit(f"No alignment found for {utterance!r} under {DATA_ROOT}")


def intervals_for(textgrid, tier_name):
    """The non-empty intervals of one tier, including the derived `wordphn` tier.

    Empty-text intervals are silence, which the alignments already mark and every
    duration in this project already excludes.
    """
    from data_common import WORDPHN_TIER, word_phone_intervals

    if tier_name == WORDPHN_TIER:
        intervals = word_phone_intervals(textgrid)
    else:
        intervals = textgrid.get_tier_by_name(tier_name)
    return [iv for iv in intervals if iv.text]


def cutoff_time(textgrid, n_words):
    """The end time of the `n_words`-th word, or None to keep the whole utterance.

    Truncating by time rather than by token count is what keeps the units aligned with
    each other: a nine-word prefix is the same stretch of speech whether it is being
    counted in words, syllables or phones.
    """
    if n_words is None:
        return None
    words = intervals_for(textgrid, "words")
    if n_words > len(words):
        raise SystemExit(f"--words {n_words} but the utterance has {len(words)} words")
    return words[n_words - 1].end_time


def tokens_for(textgrid, tier_name, cutoff):
    """One tier's token texts, up to `cutoff` seconds.

    The `phones` tier carries ARPABET stress digits, which every unit in the project
    strips, so they are stripped here too through the phone sweep's own `destress`:
    otherwise the tier block would show `IH1` where the `phone` unit shows `IH`.
    """
    from librispeech_stats import destress

    transform = destress if tier_name == "phones" else (lambda text: text)
    intervals = intervals_for(textgrid, tier_name)
    if cutoff is not None:
        intervals = [iv for iv in intervals if iv.end_time <= cutoff + 1e-9]
    return [transform(iv.text) for iv in intervals]


def encode(unit, config, textgrid, cutoff):
    """One unit's content tokens, exactly as the dataset would build them.

    Returns None if the unit's tokeniser has not been saved yet, so a missing
    `<unit>_prepare_mmap.py` run leaves the other fourteen rows readable.
    """
    from tokenisers import SubwordPhoneXBpeTokenizer

    tokens = tokens_for(textgrid, config["tier_name"], cutoff)
    try:
        tokenizer = config["tokenizer"]()
    except (OSError, ValueError) as error:
        print(f"  {unit}: no saved tokeniser ({error})", file=sys.stderr)
        return None
    ids = tokenizer.encode(tokens, is_split_into_words=True)
    pieces = tokenizer.convert_ids_to_tokens(ids)
    # Drop the boundary token the dataset appends, wherever the tokeniser put it
    pieces = [p for p in pieces if p != tokenizer.bos_token]
    if unit.startswith("subwordphnx-bpe"):
        pieces = [SubwordPhoneXBpeTokenizer.as_phones(p) for p in pieces]
    return pieces


def joined(unit, pieces, escape=False):
    """`pieces` as one line. A collocation token contains a space, so the separator has
    to be something else; `phone` is the exception noted at `DISPLAY_ORDER`.
    """
    separator = " " if unit == "phone" else (" \\| " if escape else " | ")
    return separator.join(pieces)


def print_plain(textgrid, split, units, cutoff, n_words, show_timings):
    from data_common import WORDPHN_TIER

    words = intervals_for(textgrid, "words")
    shown = tokens_for(textgrid, "words", cutoff)
    speech = sum(
        iv.duration() for iv in words if cutoff is None or iv.end_time <= cutoff + 1e-9
    )
    span = (cutoff or words[-1].end_time) - words[0].start_time
    print(f"=== {textgrid.filename.stem}  [{split}]")
    print(f"transcript: {' '.join(shown)}")
    suffix = "" if n_words is None else f" of {len(words)}"
    print(f"speech {speech:.2f}s within a {span:.2f}s span, {len(shown)} words{suffix}")

    if show_timings:
        print("\nword tier")
        for iv in words:
            if cutoff is not None and iv.end_time > cutoff + 1e-9:
                break
            print(f"  {iv.start_time:6.2f} {iv.end_time:6.2f}  {iv.text}")

    print("\ntiers")
    for tier_name in TIERS + (WORDPHN_TIER,):
        tokens = tokens_for(textgrid, tier_name, cutoff)
        print(f"  {tier_name:<11} ({len(tokens):>3})  {joined('', tokens)}")

    print("\ntokenisations, content tokens only")
    for unit, config in units.items():
        pieces = encode(unit, config, textgrid, cutoff)
        if pieces is None:
            continue
        print(f"  {unit:<22} ({len(pieces):>3})  {joined(unit, pieces)}")


def print_markdown(textgrid, units, cutoff):
    print("| unit | tokenisation (content tokens only) | n |")
    print("|---|---|---|")
    for unit, config in units.items():
        pieces = encode(unit, config, textgrid, cutoff)
        if pieces is None:
            continue
        line = joined(unit, pieces, escape=True)
        print(f"| {unit} | `{line}` | {len(pieces)} |")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "utterance",
        nargs="?",
        default=DEFAULT_UTTERANCE,
        help=f"Utterance key, any split (default: {DEFAULT_UTTERANCE})",
    )
    parser.add_argument(
        "--words",
        type=int,
        metavar="N",
        help="Show only the first N words, truncating every unit at that word's end "
        "time so the tokenisations stay aligned with each other",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print one markdown table instead, to paste into a document",
    )
    parser.add_argument(
        "--timings",
        action="store_true",
        help="Also print the word tier with its interval times",
    )
    args = parser.parse_args()

    add_stage_paths()
    units = load_units()
    path, split = find_textgrid(args.utterance)

    import tgt

    textgrid = tgt.read_textgrid(path)
    cutoff = cutoff_time(textgrid, args.words)

    if args.markdown:
        print_markdown(textgrid, units, cutoff)
    else:
        print_plain(textgrid, split, units, cutoff, args.words, args.timings)


if __name__ == "__main__":
    main()
