"""
Shared data-preparation building blocks for llm/<unit>_prepare_mmap.py scripts.

Covers what's common across all five units: reading a LibriSpeech alignment tier,
building a training-only vocabulary (for the hybrid-backoff units, syllable and word),
and writing the flat token-id mmap plus its `.json` metadata that
`TokenizedUnitsChunkedMMapDataset` trains from (see llm/data.py, to be copied from
docs/assets/opt125m-trainer-example/data.py when needed). Validation reads TextGrids
directly via `TokenizedUnitsUtteranceDataset` (tokenising on the fly with whichever
unit's tokeniser), not from a mmap, so this module only prepares the training-side
artefacts.
"""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import tgt
from tqdm import tqdm

from tokenisers import FALLBACK_MARKER

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "librispeech-alignments"
TRAIN_SPLITS = ["train-clean-100", "train-clean-360", "train-other-500"]
DEV_SPLIT = "dev-clean"

# Tier name for the derived `wordphn` unit, which no TextGrid actually carries: word
# spans whose text is replaced by their realised phones (see `word_phone_intervals`).
# Passed as a `tier_name` anywhere a real tier name goes, so `read_tier`,
# `data.TokenizedUnitsUtteranceDataset` and `eval_bitrate.dev_duration` all serve it
# without knowing it is derived.
WORDPHN_TIER = "wordphones"

# Alignment boundary times are written to the TextGrid at 2-3 decimals, so a phone's
# end time matches its word's exactly; this only guards float round-trip error.
BOUNDARY_EPS = 1e-6


def word_phone_intervals(textgrid):
    """The derived `wordphn` tier: one interval per `words`-tier interval, keeping that
    interval's own start and end times but replacing its text with its constituent
    phones, destressed and joined by "-" ("abbey" -> "AE-B-IY"). This is the `syllables`
    tier's existing format (`B-AH-T`) applied one level up, so `WordPhoneTokenizer`
    reuses `SyllableTokenizer` unchanged.

    Phones are grouped by the time boundaries the two tiers already share: a word's
    phones are exactly the run of phone intervals ending at or before the word's own end
    time. Verified over all 281,239 training and 2,703 dev-clean TextGrids with zero
    mismatches, and the resulting raw type count (89,048) reproduces
    `ngram/librispeech_stats.md`'s word vocabulary exactly. The asserts below keep that
    a checked property rather than an assumption.

    **Reusing the word interval's own times is what keeps the denominator comparable**:
    summing `duration()` over this tier gives bit-identical speech time to summing over
    the `words` tier, so `wordphn`'s bit rate divides by exactly the same seconds as
    `word`'s. Stress digits are stripped to match `ArpabetTokenizer` and the already
    destressed `syllables` tier, so the backoff inventory is the same 39 phones.
    """
    words = list(textgrid.get_tier_by_name("words"))
    phones = list(textgrid.get_tier_by_name("phones"))
    intervals = []
    i = 0
    for word in words:
        start = i
        while i < len(phones) and phones[i].end_time <= word.end_time + BOUNDARY_EPS:
            i += 1
        group = phones[start:i]
        assert group, f"no phones inside word interval {word.text!r}"
        assert group[0].start_time >= word.start_time - BOUNDARY_EPS, (
            f"phone {group[0].text!r} starts before word {word.text!r}"
        )
        intervals.append(
            tgt.Interval(
                word.start_time,
                word.end_time,
                "-".join(p.text.rstrip("012") for p in group),
            )
        )
    assert i == len(phones), f"{len(phones) - i} phone(s) left over after the last word"
    return intervals


def iter_tier(root, tier_name, sort=False, desc=None):
    """Yield (utterance key, tier) for every TextGrid under `root` (a single directory),
    reading the tier named `tier_name`. Shared low-level walk behind `read_tier` (below)
    and `llm/data.py`'s `TokenizedUnitsUtteranceDataset`, which otherwise duplicated
    this loop with slightly different behaviour.

    `tier_name` may also be `WORDPHN_TIER`, which no TextGrid carries: that yields the
    derived word-phone intervals instead (see `word_phone_intervals`).

    `sort`, if set, sorts paths before walking (`TokenizedUnitsUtteranceDataset` wants a
    deterministic order for validation; `read_tier` doesn't need one, since downstream
    consumers, e.g. `write_train_mmap`, don't care about utterance order in a flat
    training stream). `desc`, if given, wraps the walk in a tqdm progress bar (forces
    `sort=True`'s list, since tqdm needs a total to show one); leave unset for no
    progress bar.
    """
    paths = Path(root).rglob("*.TextGrid")
    if sort or desc is not None:
        paths = sorted(paths)
    if desc is not None:
        paths = tqdm(paths, desc=desc, leave=False)
    for path in paths:
        textgrid = tgt.read_textgrid(path)
        if tier_name == WORDPHN_TIER:
            # Derived, not a real tier: a plain list of Intervals, which every caller
            # consumes identically (`iv.text`, `iv.duration()`, iteration).
            yield path.stem, word_phone_intervals(textgrid)
        else:
            yield path.stem, textgrid.get_tier_by_name(tier_name)


def read_tier(splits, tier_name, token_transform=None):
    """{utterance key: (tokens, speech-only duration in seconds)} for every TextGrid
    under `splits`, reading the tier named `tier_name`. `tokens` is a list, not a joined
    string (unlike ngram/sweep_common.load_utterances), since that's the shape needed
    for mmap writing and for building/applying a hybrid-backoff vocabulary. tgt already
    drops silence, so summing interval durations gives speech-only time.
    `token_transform`, if given, is applied to each interval's text before collecting
    (e.g. stripping stress digits for phones).
    """
    transform = token_transform or (lambda t: t)
    utterances = {}
    for split in splits:
        for key, tier in iter_tier(DATA_ROOT / split, tier_name):
            tokens = [transform(iv.text) for iv in tier]
            duration = sum(iv.duration() for iv in tier)
            utterances[key] = (tokens, duration)
    return utterances


def build_min_count_vocab(token_lists, min_count=2):
    """Set of types occurring at least `min_count` times across `token_lists` (an
    iterable of token lists). Pass training data only, never dev/test, per the project's
    "vocabulary built from training data only" design principle (CLAUDE.md).
    """
    counts = Counter()
    for tokens in token_lists:
        counts.update(tokens)
    return {t for t, c in counts.items() if c >= min_count}


def augment_with_spellout(utterances, rate, seed, split_fn):
    """Independently, for `rate` of all token occurrences (any utterance, any position),
    replace the token with `split_fn(token)`'s pieces, regardless of whether it's
    in-vocab. Training-only: never applied at validation/evaluation time. Shared by
    `word_prepare_mmap.py`/`word_bpe_prepare_mmap.py` (`split_fn=list`, word ->
    characters) and `syllable_prepare_mmap.py` (`split_fn=lambda t: t.split("-")`,
    syllable -> phones).

    Pieces are emitted **already marked** with `FALLBACK_MARKER`, so they encode to
    the same fallback ids the deterministic spell-out rule produces at test time. Bare
    pieces would not: `_encode_plus` keeps anything already in the unit vocabulary
    whole, and every single letter is itself a LibriSpeech word, so bare characters
    resolved to whole-word ids and the augmentation exercised the fallback pathway not
    at all. Each tokeniser's `_encode_plus` passes a marked piece straight through.
    """
    rng = random.Random(seed)
    augmented = {}
    for key, (tokens, duration) in utterances.items():
        new_tokens = []
        for tok in tokens:
            if rng.random() < rate:
                new_tokens.extend(FALLBACK_MARKER + p for p in split_fn(tok))
            else:
                new_tokens.append(tok)
        augmented[key] = (new_tokens, duration)
    return augmented


def write_train_mmap(tokenizer, utterances, out_dir, name, dtype):
    """Encode every utterance's tokens with `tokenizer.encode()`, *keeping* its leading
    `<bos>`, and write a flat token-id mmap plus its `.json` metadata. `utterances` is a
    `{key: (tokens, duration)}` dict, e.g. from `read_tier`; only the tokens are used
    here, in whatever order `dict.values()` gives (order doesn't matter for a flat
    training stream).

    Keeping each utterance's `<bos>` gives the flat stream a real, densely occurring
    utterance-boundary marker between every utterance, so the model is trained to
    expect one at a genuine utterance end. This matches ngram/'s and qwen/'s convention
    of scoring an explicit end-of-utterance token.
    """
    assert (
        tokenizer.vocab_size < np.iinfo(dtype).max
    ), f"vocab_size={tokenizer.vocab_size} does not fit in dtype {dtype}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{name}-train.bin"

    num_tokens = num_files = 0
    print("Writing:", bin_path)
    with bin_path.open("wb") as out:
        for tokens, _ in tqdm(utterances.values(), desc=f"Encoding ({name})"):
            ids = tokenizer.encode(tokens, is_split_into_words=True)
            arr = np.asarray(ids, dtype=dtype)
            out.write(arr.tobytes())
            num_tokens += int(arr.size)
            num_files += 1

    meta = {
        "num_tokens": num_tokens,
        "num_files": num_files,
        "dtype": np.dtype(dtype).name,
    }
    meta_path = out_dir / f"{name}-train.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {num_tokens} tokens from {num_files} files to {bin_path}")
    return bin_path, meta_path


if __name__ == "__main__":
    # Toy sanity check: build a min-count vocab and round-trip a tiny mmap through a
    # real ArpabetTokenizer, without touching real LibriSpeech data.
    import shutil
    import sys
    import tempfile

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tokenisers import ArpabetTokenizer

    toy_utterances = {
        "u1": (["K", "AE", "T"], 1.0),
        "u2": (["D", "AA", "G"], 1.5),
        "u3": (["K", "AE", "T"], 1.0),  # repeated type, for the vocab check
    }
    vocab = build_min_count_vocab(
        [tokens for tokens, _ in toy_utterances.values()], min_count=2
    )
    assert vocab == {"K", "AE", "T"}, vocab
    print(f"build_min_count_vocab: {vocab} (expected types seen >= 2 times)")

    # Toy sanity check: augment_with_spellout at rate=1.0 (deterministic) must replace
    # every token via split_fn, never touch duration, and preserve keys. Pieces come
    # back marked, so they encode to fallback ids rather than being mistaken for whole
    # units that happen to share the same bare string.
    augmented = augment_with_spellout(toy_utterances, rate=1.0, seed=0, split_fn=list)
    assert augmented["u1"] == (["~K", "~A", "~E", "~T"], 1.0), augmented["u1"]
    assert all(
        p.startswith(FALLBACK_MARKER) for toks, _ in augmented.values() for p in toks
    ), augmented
    assert augmented.keys() == toy_utterances.keys()
    unaugmented = augment_with_spellout(toy_utterances, rate=0.0, seed=0, split_fn=list)
    assert unaugmented == toy_utterances, unaugmented
    print("augment_with_spellout: rate=1.0 spells out every token as marked fallback "
          "pieces, rate=0.0 is a no-op")

    # Toy sanity check: word_phone_intervals groups phones under words by their shared
    # boundaries, destresses them, and keeps the *word* interval's own times (so the
    # derived tier's duration matches the words tier's exactly). Built from a synthetic
    # TextGrid, so this needs no real LibriSpeech file.
    toy_tg = tgt.TextGrid()
    toy_words = tgt.IntervalTier(name="words")
    toy_phones = tgt.IntervalTier(name="phones")
    for start, end, text in [(0.56, 1.25, "northanger"), (1.29, 1.83, "abbey")]:
        toy_words.add_interval(tgt.Interval(start, end, text))
    for start, end, text in [
        (0.56, 0.67, "N"), (0.67, 0.74, "AO2"), (0.74, 0.82, "R"), (0.82, 0.91, "TH"),
        (0.91, 1.05, "AE1"), (1.05, 1.12, "NG"), (1.12, 1.25, "ER0"),
        (1.29, 1.45, "AE1"), (1.45, 1.54, "B"), (1.54, 1.83, "IY0"),
    ]:
        toy_phones.add_interval(tgt.Interval(start, end, text))
    toy_tg.add_tiers([toy_words, toy_phones])

    derived = word_phone_intervals(toy_tg)
    assert [iv.text for iv in derived] == ["N-AO-R-TH-AE-NG-ER", "AE-B-IY"], derived
    assert abs(sum(iv.duration() for iv in derived)
               - sum(iv.duration() for iv in toy_words)) < 1e-12
    print(f"word_phone_intervals: {[iv.text for iv in derived]}, duration matches the "
          f"words tier exactly")

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        tokenizer = ArpabetTokenizer()
        bin_path, meta_path = write_train_mmap(
            tokenizer, toy_utterances, tmp_dir, "toy", dtype=np.uint16
        )
        meta = json.loads(meta_path.read_text())
        assert meta["num_files"] == 3, meta
        # 4 tokens/utterance (<bos> + 3 phones) x 3 utterances: the leading <bos> is now
        # kept, giving the flat stream a real, densely-occurring utterance-boundary
        # marker instead of being dropped.
        assert meta["num_tokens"] == 12, meta
        ids = np.memmap(bin_path, dtype=np.uint16, mode="r")
        assert ids.size == meta["num_tokens"], (ids.size, meta["num_tokens"])
        decoded = tokenizer.decode(ids[:4].tolist())
        print(f"round-tripped first utterance (with boundary token): {decoded!r}")
        assert decoded == "<bos> K AE T", decoded
        print("write_train_mmap round-trip passed.")
    finally:
        shutil.rmtree(tmp_dir)
