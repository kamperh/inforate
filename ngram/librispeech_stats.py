"""
Statistics for LibriSpeech 960h and dev-clean.

Reads the words/phones/syllables tiers of every TextGrid forced alignment and reports,
per set: utterance count, speech-only duration, token counts and inventory sizes for
each unit. Also checks that `words`-tier text is already clean ([a-z']+ only), which
ngram/ngram_word_sweep.py relies on when reconstructing sentences by plain whitespace
joining.

tgt already drops empty-text (silence) intervals from a tier by default, so every
interval returned is real speech and no filtering is needed: summing interval durations
directly gives speech-only time.

The phone inventory is reported unstressed (stress digits stripped, e.g. `AH0` -> `AH`),
matching ngram/ngram_phone_sweep.py. This only shrinks the inventory (69 -> 39 types);
token counts are unaffected, since destressing collapses types, not occurrences.

A second section reports the **realisation-based** word vocabulary (the `wordphn` unit
trained in llm/): each word counted by its aligned phones rather than its spelling, so
"abbey" is `AE-B-IY`. It quantifies both directions in which the two vocabularies
disagree: one spelling with several realisations, and one realisation covering several
spellings. The grouping is duplicated here in the few lines it takes rather than
imported from llm/data_common.py, matching this project's convention that ngram/ and
qwen/ do not depend on llm/'s layout. `word_phone_units` is also what
ngram/sweep_common.py builds the `wordphn` and `wordphn-BPE` sweeps' own tier from, so
this module is the one definition of the grouping rule inside ngram/ (as it already is
of `destress`, which ngram/ngram_phone_sweep.py imports).
"""

import re
from collections import Counter, defaultdict
from pathlib import Path

import tgt
from tqdm import tqdm

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "librispeech-alignments"
OUT_PATH = Path(__file__).resolve().parent / "librispeech_stats.md"

SETS = {
    "LibriSpeech 960h": [
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
    ],
    "dev-clean": ["dev-clean"],
}

CLEAN_WORD_RE = re.compile(r"^[a-z']+$")
STRESS_RE = re.compile(r"\d+$")

# The threshold the neural stage applies to word, wordphn and syllable, reported here
# alongside the full inventories so the two can be compared; see design.md,
# "Vocabulary rules", for why hapax legomena are cut. ngram/ applies no threshold, so
# for that stage this is a descriptive statistic rather than a vocabulary it uses
MIN_COUNT = 2

# Alignment times are written to 2-3 decimals, so a word's last phone ends exactly at
# the word's own end time; this only guards float round-trip error.
BOUNDARY_EPS = 1e-6


def destress(phone):
    """Strip trailing ARPABET stress digits, e.g. 'AH0' -> 'AH'."""
    return STRESS_RE.sub("", phone)


def word_phone_units(textgrid):
    """[(word, realisation)] for one TextGrid: each `words`-tier interval paired with its
    own phones, destressed and joined by "-" ("abbey" -> "AE-B-IY"). A word's phones are
    the run of `phones`-tier intervals ending at or before the word's own end time; the
    two tiers share their boundaries exactly, verified over all 281,239 training and
    2,703 dev-clean TextGrids with zero mismatches.
    """
    words = list(textgrid.get_tier_by_name("words"))
    phones = list(textgrid.get_tier_by_name("phones"))
    pairs = []
    i = 0
    for word in words:
        start = i
        while i < len(phones) and phones[i].end_time <= word.end_time + BOUNDARY_EPS:
            i += 1
        group = phones[start:i]
        assert group, f"no phones inside word interval {word.text!r}"
        pairs.append((word.text, "-".join(destress(p.text) for p in group)))
    assert i == len(phones), f"{len(phones) - i} phone(s) left over after the last word"
    return pairs


def iter_textgrids(splits):
    paths = []
    for split in splits:
        paths.extend((DATA_ROOT / split).rglob("*.TextGrid"))
    return sorted(paths)


def at_least(types, min_count=MIN_COUNT):
    """How many types in a `Counter` occur at least `min_count` times.

    Equivalent to `len(llm/data_common.build_min_count_vocab(..., min_count))` over the
    same tier, so the 960h row of the table this feeds reproduces each unit's saved
    vocabulary under `llm/data/` exactly, which is the check that the two stages are
    counting the same thing.
    """
    return sum(1 for count in types.values() if count >= min_count)


def stats_for_set(name, splits):
    n_utterances = 0
    speech_seconds = 0.0

    word_tokens = 0
    word_types = Counter()
    dirty_words = set()

    phone_tokens = 0
    phone_types = Counter()

    syll_tokens = 0
    syll_types = Counter()

    wordphn_tokens = 0
    wordphn_types = Counter()
    realisations_of = defaultdict(set)  # word -> its realisations
    words_of = defaultdict(set)  # realisation -> the words realised that way

    paths = iter_textgrids(splits)
    for path in tqdm(paths, desc=name):
        tg = tgt.read_textgrid(path)
        n_utterances += 1

        for iv in tg.get_tier_by_name("words"):
            speech_seconds += iv.duration()
            word_tokens += 1
            word_types[iv.text] += 1
            if not CLEAN_WORD_RE.match(iv.text):
                dirty_words.add(iv.text)

        for iv in tg.get_tier_by_name("phones"):
            phone_tokens += 1
            phone_types[destress(iv.text)] += 1

        for iv in tg.get_tier_by_name("syllables"):
            syll_tokens += 1
            syll_types[iv.text] += 1

        for word, realisation in word_phone_units(tg):
            wordphn_tokens += 1
            wordphn_types[realisation] += 1
            realisations_of[word].add(realisation)
            words_of[realisation].add(word)

    return {
        "name": name,
        "n_utterances": n_utterances,
        "speech_seconds": speech_seconds,
        "word_tokens": word_tokens,
        "word_vocab": len(word_types),
        "phone_tokens": phone_tokens,
        "phone_inventory": len(phone_types),
        "syll_tokens": syll_tokens,
        "syll_inventory": len(syll_types),
        "wordphn_tokens": wordphn_tokens,
        "wordphn_vocab": len(wordphn_types),
        "wordphn_hapax": sum(1 for c in wordphn_types.values() if c == 1),
        "word_vocab_min": at_least(word_types),
        "phone_inventory_min": at_least(phone_types),
        "syll_inventory_min": at_least(syll_types),
        "wordphn_vocab_min": at_least(wordphn_types),
        "multi_realisation_words": sum(1 for r in realisations_of.values() if len(r) > 1),
        "multi_word_realisations": sum(1 for w in words_of.values() if len(w) > 1),
        "dirty_words": sorted(dirty_words)[:20],
        "n_dirty_words": len(dirty_words),
    }


def format_row(s):
    hours = s["speech_seconds"] / 3600
    return (
        f"| {s['name']} | {s['n_utterances']:,} | {hours:.1f}h "
        f"({s['speech_seconds']:.0f}s) | {s['word_tokens']:,} | {s['word_vocab']:,} | "
        f"{s['phone_tokens']:,} | {s['phone_inventory']} | {s['syll_tokens']:,} | "
        f"{s['syll_inventory']:,} |"
    )


def format_min_count_row(s):
    """One row of the min-count table: a separate, narrow table rather than four more
    columns on the main one, which is already nine wide (see `format_realisation_line`).
    """
    return (
        f"| {s['name']} | {s['word_vocab_min']:,} | {s['wordphn_vocab_min']:,} | "
        f"{s['syll_inventory_min']:,} | {s['phone_inventory_min']} |"
    )


def format_realisation_line(s):
    """One bullet summarising the realisation-based ("wordphn") word vocabulary against
    the orthographic one, for the section the main table deliberately does not carry:
    that table is already nine columns, and pandoc switches a markdown table to
    `\\real{...}` column widths once its total text gets long enough (CLAUDE.md).
    """
    return (
        f"{s['name']}: {s['wordphn_tokens']:,} wordphn tokens (identical to word "
        f"tokens, one unit per word interval), {s['wordphn_vocab']:,} realisation "
        f"types against {s['word_vocab']:,} orthographic word types, of which "
        f"{s['wordphn_hapax']:,} occur once. {s['multi_realisation_words']:,} word "
        f"types have more than one realisation, and {s['multi_word_realisations']:,} "
        f"realisations cover more than one word type."
    )


def main():
    print("Reading:", DATA_ROOT)
    results = [stats_for_set(name, splits) for name, splits in SETS.items()]

    header = (
        "| set | utterances | speech duration | word tokens | word vocab | "
        "phone tokens | phone inventory (unstressed) | syllable tokens | "
        "syllable inventory |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    table = header + "\n" + "\n".join(format_row(s) for s in results) + "\n"

    min_count_header = (
        "| set | word | wordphn | syllable | phone |\n"
        "|---|---:|---:|---:|---:|"
    )
    min_count_table = (
        min_count_header
        + "\n"
        + "\n".join(format_min_count_row(s) for s in results)
        + "\n"
    )

    print()
    print(table)
    print()
    print(min_count_table)

    cleanliness_lines = []
    for s in results:
        if s["n_dirty_words"]:
            line = (
                f"{s['name']}: {s['n_dirty_words']} non-clean word types "
                f"(not matching [a-z']+), e.g. {s['dirty_words']}"
            )
        else:
            line = f"{s['name']}: words-tier text is clean ([a-z']+ only)."
        print(line)
        cleanliness_lines.append(line)

    realisation_lines = []
    print()
    for s in results:
        line = format_realisation_line(s)
        print(line)
        realisation_lines.append(line)

    print("Writing:", OUT_PATH)
    with open(OUT_PATH, "w") as f:
        f.write("# LibriSpeech / dev-clean statistics\n\n")
        f.write(table)
        f.write(f"\n## Vocabulary at count >= {MIN_COUNT}\n\n")
        f.write(
            "Types occurring at least "
            f"{MIN_COUNT} times, i.e. the inventory left once hapax legomena are cut. "
            "This is the vocabulary the neural stage builds (`design.md`, "
            "\"Vocabulary rules\"); `ngram/` applies no threshold, so for that stage "
            "the full inventories in the table above are the ones in use. The 960h row "
            "reproduces each unit's saved vocabulary under `llm/data/` exactly.\n\n"
        )
        f.write(min_count_table)
        f.write("\n## Words-tier cleanliness check\n\n")
        for line in cleanliness_lines:
            f.write(f"- {line}\n")
        f.write("\n## Realisation-based word vocabulary (the wordphn unit)\n\n")
        f.write(
            "Each word counted by its aligned phones rather than its spelling "
            '("abbey" is `AE-B-IY`); see llm/wordphn_prepare_mmap.py.\n\n'
        )
        for line in realisation_lines:
            f.write(f"- {line}\n")


if __name__ == "__main__":
    main()
