"""
Shared data-reading building blocks for qwen/eval_bitrate.py.

Trimmed copy of the relevant parts of llm/data_common.py (DATA_ROOT, iter_tier,
read_tier): only reading a LibriSpeech alignment tier's text and speech-only
duration is needed here, since nothing is trained or tokenised into a mmap in
this directory -- Qwen's own pretrained tokenizer is used directly, and scoring
is inference-only. Duplicated rather than imported from llm/, matching
llm/data_common.py's own stated convention ("kept self-contained... per the
project convention") -- qwen/ shouldn't depend on llm/'s internal file layout.

Also provides `primer_utterance_keys`: a deterministic, fixed selection of
utterance keys from train-clean-100 used to build eval_bitrate.py's primer (see
its module docstring for why the primer is deliberately *not* drawn from
dev-clean).
"""

from pathlib import Path

import tgt
from tqdm import tqdm

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "librispeech-alignments"
TRAIN_SPLITS = ["train-clean-100", "train-clean-360", "train-other-500"]
DEV_SPLIT = "dev-clean"

PRIMER_SPLIT = "train-clean-100"
NUM_PRIMER_UTTERANCES = 8


def iter_tier(root, tier_name, sort=False, desc=None):
    """Yield (utterance key, tier) for every TextGrid under `root` (a single
    directory), reading the tier named `tier_name`. `sort`, if set, sorts paths
    before walking, for a deterministic order (needed for `primer_utterance_keys`
    below, which must return the same utterances every time). `desc`, if given,
    wraps the walk in a tqdm progress bar (forces `sort=True`'s list, since tqdm
    needs a total to show one).
    """
    paths = Path(root).rglob("*.TextGrid")
    if sort or desc is not None:
        paths = sorted(paths)
    if desc is not None:
        paths = tqdm(paths, desc=desc, leave=False)
    for path in paths:
        tier = tgt.read_textgrid(path).get_tier_by_name(tier_name)
        yield path.stem, tier


def read_tier(splits, tier_name, token_transform=None):
    """{utterance key: (tokens, speech-only duration in seconds)} for every
    TextGrid under `splits`, reading the tier named `tier_name`. `tokens` is a
    list, not a joined string. tgt already drops silence, so summing interval
    durations gives speech-only time. `token_transform`, if given, is applied to
    each interval's text before collecting.
    """
    transform = token_transform or (lambda t: t)
    utterances = {}
    for split in splits:
        for key, tier in iter_tier(DATA_ROOT / split, tier_name):
            tokens = [transform(iv.text) for iv in tier]
            duration = sum(iv.duration() for iv in tier)
            utterances[key] = (tokens, duration)
    return utterances


def primer_utterances(n=NUM_PRIMER_UTTERANCES):
    """{utterance key: word tokens} for the first `n` utterances (sorted
    TextGrid path order, so this is deterministic and reproducible) from
    `PRIMER_SPLIT`. Stops reading as soon as `n` are found, rather than walking
    all of train-clean-100. Deliberately drawn from train-clean-100, not
    dev-clean: this keeps the full, unmodified dev-clean set as the
    scored/denominator set, matching every other unit's bit rate in this
    project (ngram/'s sweeps and llm/eval_bitrate.py both score all of
    dev-clean, untouched) -- the predecessor project drew its own primer from
    the same pool it scored, which this deliberately avoids.
    """
    utterances = {}
    for key, tier in iter_tier(DATA_ROOT / PRIMER_SPLIT, "words", sort=True):
        tokens = [iv.text for iv in tier if iv.text]  # tgt keeps silence as "", drop it
        if not tokens:
            continue
        utterances[key] = tokens
        if len(utterances) == n:
            break
    return utterances


if __name__ == "__main__":
    # Toy sanity check: primer keys are deterministic, the right count, and
    # disjoint from dev-clean. No GPU, no model, no real scoring -- just the
    # selection logic.
    primer_a = primer_utterances()
    primer_b = primer_utterances()
    assert list(primer_a.keys()) == list(primer_b.keys()), (
        "primer_utterances is not deterministic"
    )
    assert len(primer_a) == NUM_PRIMER_UTTERANCES, primer_a

    dev_keys = set(read_tier([DEV_SPLIT], "words").keys())
    assert not (set(primer_a) & dev_keys), "primer overlaps dev-clean"

    print(f"primer_utterances keys: {list(primer_a.keys())}")
    print(
        f"{len(primer_a)} deterministic primer utterances from {PRIMER_SPLIT}, "
        f"disjoint from {len(dev_keys)} dev-clean utterances. Self-test passed."
    )
