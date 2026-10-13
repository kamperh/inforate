"""Evaluate a trained checkpoint's bit rate on dev-clean.

For each unit in `UNITS` whose checkpoint exists, loads the trained model, scores every
dev-clean utterance (same `TokenizedUnitsUtteranceDataset` construction used for
validation), converts the summed cross-entropy loss (natural log) to bits, and reports
bit rate (total bits / total dev-clean speech-only seconds, from the same alignment
durations `ngram/ngram_<unit>_sweep.py` uses for its own bit rate) and perplexity
(per-token, not comparable across units). Mirrors
`ngram/ngram_{word,syllable,phone,bpe}_sweep.py`'s report style (module docstring,
`OUT_PATH`, "Reading:"/"Writing:" prints, one markdown table).

Bit rate is `sum(bits) / sum(duration)`, matching `ngram/sweep_common.bitrate()`'s own
definition exactly: a ratio of two independently-computed totals, so utterance-level
pairing between bits and duration isn't needed, only the two totals.

Uses the *final* checkpoint (`{run_name}.pt`), not `{run_name}-best.pt`, per
`trainer.py`'s own `save_checkpoint` comment ("Final weights are the checkpoint used for
downstream scoring").
"""

import argparse
import json
import math
from functools import lru_cache, partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import OPTConfig, OPTForCausalLM

from data import TokenizedUnitsUtteranceDataset
from data_common import DATA_ROOT, DEV_SPLIT, WORDPHN_TIER, read_tier
from tokenisers import (
    ArpabetTokenizer,
    SubwordBpeTokenizer,
    SubwordPhoneXBpeTokenizer,
    SyllableTokenizer,
    WordBpeTokenizer,
    WordPhoneBpeTokenizer,
    WordPhoneTokenizer,
    WordTokenizer,
)
from trainer import collate

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_PATH = Path(__file__).resolve().parent / "eval_bitrate.md"


class StaleCheckpoint(RuntimeError):
    """A checkpoint trained under a vocabulary the current tokeniser no longer produces.
    Raised rather than asserted so a full sweep can skip it and carry on, while an
    explicit `--units` request still fails loudly.
    """


PER_UTTERANCE_DIR = Path(__file__).resolve().parent / "per_utterance"
DEFAULT_BATCH_SIZE = 20
NUM_WORKERS = (
    0  # Matches phone_train.py: multiprocessing not verified in this environment
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Per-base-unit (tier_name, tokenizer, batch_size) shared by every variant of that unit
# below (canonical, "-lr2e-4" archival, and any hyperparameter-sweep entries) -- spread
# into each variant's dict literal via {**BASE["word"], "checkpoint": ...}, so
# tier_name/tokenizer/batch_size are written once per unit instead of once per variant.
# "batch_size" is optional, defaulting to DEFAULT_BATCH_SIZE; large-vocabulary units
# need a smaller one, since cross_entropy's internal float32 softmax buffer scales with
# batch_size x seq_len x vocab_size almost independently of model size. The same issue
# forces word_train.py's batch_size down, at training time.
BASE = {
    "phone": {"tier_name": "phones", "tokenizer": ArpabetTokenizer},
    "subword-bpe": {
        "tier_name": "words",
        "tokenizer": lambda: SubwordBpeTokenizer(
            DATA_DIR / "subword-bpe-mmap" / "tokenizer.json"
        ),
    },
    # One entry per vocabulary size, built in a loop rather than written out three
    # times: these differ only in the saved tokeniser, and the `v=v` default binds the
    # size at definition time so every lambda does not close over the last one.
    # WORDPHN_TIER throughout, since the merges are confined to a word and so the
    # tokeniser needs the grouping; the phones and the seconds are the same either way.
    **{
        f"subwordphnx-bpe-{v}": {
            "tier_name": WORDPHN_TIER,
            "tokenizer": lambda v=v: SubwordPhoneXBpeTokenizer(
                DATA_DIR / "subwordphnx-bpe-mmap" / f"subwordphnx-bpe{v}-tokenizer.json"
            ),
        }
        for v in (200, 2000, 5000)
    },
    "word": {
        "tier_name": "words",
        "tokenizer": lambda: WordTokenizer(DATA_DIR / "word-mmap" / "vocab.json"),
        "batch_size": 4,  # 58,006 vocab; verified empirically, peak 4.21GB at seq_len=2047
    },
    "wordphn": {
        "tier_name": WORDPHN_TIER,
        "tokenizer": lambda: WordPhoneTokenizer(
            DATA_DIR / "wordphn-mmap" / "vocab.json"
        ),
        "batch_size": 4,  # 56,296 vocab, just under word's; same batch size as word
    },
    "syllable": {
        "tier_name": "syllables",
        "tokenizer": lambda: SyllableTokenizer(
            DATA_DIR / "syllable-mmap" / "vocab.json"
        ),
        # 11,427 vocab; DEFAULT_BATCH_SIZE=20 verified empirically, peak 3.95GB
    },
    "word-bpe-200": {
        "tier_name": "words",
        "tokenizer": lambda: WordBpeTokenizer(
            DATA_DIR / "word-bpe-mmap" / "word-bpe200-vocab.json"
        ),
        # 58,206 vocab; batch_size=3 verified comfortable
        "batch_size": 3,
    },
    "word-bpe-1000": {
        "tier_name": "words",
        "tokenizer": lambda: WordBpeTokenizer(
            DATA_DIR / "word-bpe-mmap" / "word-bpe1000-vocab.json"
        ),
        "batch_size": 3,
    },
    "word-bpe-5000": {
        "tier_name": "words",
        "tokenizer": lambda: WordBpeTokenizer(
            DATA_DIR / "word-bpe-mmap" / "word-bpe5000-vocab.json"
        ),
        "batch_size": 3,
    },
    "wordphn-bpe-200": {
        "tier_name": WORDPHN_TIER,
        "tokenizer": lambda: WordPhoneBpeTokenizer(
            DATA_DIR / "wordphn-bpe-mmap" / "wordphn-bpe200-vocab.json"
        ),
        "batch_size": 3,  # Slightly under word-BPE-200's vocab; same batch size
    },
    "wordphn-bpe-1000": {
        "tier_name": WORDPHN_TIER,
        "tokenizer": lambda: WordPhoneBpeTokenizer(
            DATA_DIR / "wordphn-bpe-mmap" / "wordphn-bpe1000-vocab.json"
        ),
        "batch_size": 3,
    },
    "wordphn-bpe-5000": {
        "tier_name": WORDPHN_TIER,
        "tokenizer": lambda: WordPhoneBpeTokenizer(
            DATA_DIR / "wordphn-bpe-mmap" / "wordphn-bpe5000-vocab.json"
        ),
        "batch_size": 3,
    },
}

# One entry per evaluated checkpoint
UNITS = {
    "phone": {
        **BASE["phone"],
        "checkpoint": CHECKPOINT_DIR
        / "opt125m-phone-librispeech"
        / "opt125m-phone-librispeech.pt",
        # Canonical run: lr_max=1e-3, 8200 steps.
    },
    "subword-bpe": {
        **BASE["subword-bpe"],
        "checkpoint": CHECKPOINT_DIR
        / "opt125m-subword-bpe-librispeech"
        / "opt125m-subword-bpe-librispeech.pt",
        # Canonical run: lr_max=1e-3, 3300 steps.
    },
    # Canonical runs, lr_max=1e-3, at three vocabulary sizes; see results.md,
    # "Subwordphn-BPE".
    **{
        f"subwordphnx-bpe-{v}": {
            **BASE[f"subwordphnx-bpe-{v}"],
            "checkpoint": CHECKPOINT_DIR
            / f"opt125m-subwordphnx-bpe{v}-librispeech"
            / f"opt125m-subwordphnx-bpe{v}-librispeech.pt",
        }
        for v in (200, 2000, 5000)
    },
    "word": {
        **BASE["word"],
        "checkpoint": CHECKPOINT_DIR
        / "opt125m-word-librispeech"
        / "opt125m-word-librispeech.pt",
        # Canonical run: lr_max=1e-3, 2600 steps. No "word-lr2e-4" archival entry
        # exists: that checkpoint was lost to a mis-named sweep run.
    },
    "wordphn": {
        **BASE["wordphn"],
        "checkpoint": CHECKPOINT_DIR
        / "opt125m-wordphn-librispeech"
        / "opt125m-wordphn-librispeech.pt",
        # Canonical run: lr_max=1e-3, 20 epochs, seed 0.
    },
    "wordphn-bpe-200": {
        **BASE["wordphn-bpe-200"],
        "checkpoint": CHECKPOINT_DIR
        / "opt125m-wordphn-bpe200-librispeech"
        / "opt125m-wordphn-bpe200-librispeech.pt",
        # Canonical run: lr_max=1e-3, 20 epochs, seed 0.
    },
    "wordphn-bpe-1000": {
        **BASE["wordphn-bpe-1000"],
        "checkpoint": CHECKPOINT_DIR
        / "opt125m-wordphn-bpe1000-librispeech"
        / "opt125m-wordphn-bpe1000-librispeech.pt",
        # Canonical run: lr_max=1e-3, 20 epochs, seed 0.
    },
    "wordphn-bpe-5000": {
        **BASE["wordphn-bpe-5000"],
        "checkpoint": CHECKPOINT_DIR
        / "opt125m-wordphn-bpe5000-librispeech"
        / "opt125m-wordphn-bpe5000-librispeech.pt",
        # Canonical run: lr_max=1e-3, 20 epochs, seed 0.
    },
    "syllable": {
        **BASE["syllable"],
        "checkpoint": CHECKPOINT_DIR
        / "opt125m-syllable-librispeech"
        / "opt125m-syllable-librispeech.pt",
        # Canonical run: lr_max=1e-3, 3400 steps.
    },
    "word-bpe-200": {
        **BASE["word-bpe-200"],
        "checkpoint": CHECKPOINT_DIR
        / "opt125m-word-bpe200-mincount2-librispeech"
        / "opt125m-word-bpe200-mincount2-librispeech.pt",
        # Canonical run: --min-count 2 (the default), lr_max=1e-3.
    },
    "word-bpe-1000": {
        **BASE["word-bpe-1000"],
        "checkpoint": CHECKPOINT_DIR
        / "opt125m-word-bpe1000-mincount2-librispeech"
        / "opt125m-word-bpe1000-mincount2-librispeech.pt",
        # Canonical run: --min-count 2 (the default), lr_max=1e-3.
    },
    "word-bpe-5000": {
        **BASE["word-bpe-5000"],
        "checkpoint": CHECKPOINT_DIR
        / "opt125m-word-bpe5000-mincount2-librispeech"
        / "opt125m-word-bpe5000-mincount2-librispeech.pt",
        # Canonical run: --min-count 2 (the default), lr_max=1e-3.
    },
    # Seed-variance runs, giving the noise floor that a difference between two
    # independently trained units must clear. Seeds 1-3 were run alongside each unit's
    # canonical seed-0 run, seed 4 on 2026-10-12, so twelve units have five runs each.
    # subwordphnx-BPE at 200 and 5000 were single runs until then; 5000 was brought up
    # to five runs on 2026-10-13, leaving 200 the one unit whose SD rests on two runs.
    # See results.md, "Seed variance". Superseded seed dumps live in
    # llm/per_utterance/superseded/preaug-seeds/ and are excluded from the default
    # utils/bootstrap.py sweep, which globs only the top-level directory.
    **{
        f"{unit}-seed{seed}": {
            **BASE[unit],
            "checkpoint": CHECKPOINT_DIR
            / f"opt125m-{stem}-seed{seed}-librispeech"
            / f"opt125m-{stem}-seed{seed}-librispeech.pt",
        }
        for unit, stem, seeds in (
            ("phone", "phone", (1, 2, 3, 4)),
            ("subword-bpe", "subword-bpe", (1, 2, 3, 4)),
            ("word", "word", (1, 2, 3, 4)),
            ("syllable", "syllable", (1, 2, 3, 4)),
            ("word-bpe-200", "word-bpe200", (1, 2, 3, 4)),
            ("word-bpe-1000", "word-bpe1000", (1, 2, 3, 4)),
            ("word-bpe-5000", "word-bpe5000", (1, 2, 3, 4)),
            ("wordphn", "wordphn", (1, 2, 3, 4)),
            ("wordphn-bpe-200", "wordphn-bpe200", (1, 2, 3, 4)),
            ("wordphn-bpe-1000", "wordphn-bpe1000", (1, 2, 3, 4)),
            ("wordphn-bpe-5000", "wordphn-bpe5000", (1, 2, 3, 4)),
            ("subwordphnx-bpe-2000", "subwordphnx-bpe2000", (1, 2, 3, 4)),
            ("subwordphnx-bpe-200", "subwordphnx-bpe200", (1,)),
            ("subwordphnx-bpe-5000", "subwordphnx-bpe5000", (1, 2, 3, 4)),
        )
        for seed in seeds
    },
}


@lru_cache(maxsize=None)
def dev_duration(tier_name):
    """Total dev-clean speech-only seconds, from the same alignment durations
    ngram/ngram_<unit>_sweep.py uses for its own bit rate. Cached per tier_name: several
    UNITS entries share a tier_name (e.g. six "words" entries), and this would otherwise
    re-walk the same 2703 TextGrids once per entry in main()'s loop.
    """
    dev_utterances = read_tier([DEV_SPLIT], tier_name)
    return sum(duration for _, duration in dev_utterances.values())


def score_dev(model, tokenizer, tier_name, batch_size):
    """Total (nats, non-pad target tokens) over dev-clean, teacher-forced, summed (not
    averaged) so the two can be combined with other units' totals or with duration
    independently. Also returns per-utterance `{key: (bits, duration)}`, which
    `utils/bootstrap.py` resamples; the totals are exactly the sums of those
    per-utterance values, so persisting them cannot drift from the reported bit rate.

    `reduction="none"` rather than `"sum"`: the per-position losses are reshaped back to
    `(batch, seq)` and summed along the sequence, which gives the per-utterance total
    and, summed again, the identical grand total the old `"sum"` call produced. Padding
    positions contribute exactly 0 because `ignore_index` zeroes them, so no masking is
    needed beyond that.
    """
    dataset = TokenizedUnitsUtteranceDataset(
        [DATA_ROOT / DEV_SPLIT], tier_name, tokenizer
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        collate_fn=partial(collate, pad_token_id=tokenizer.pad_token_id),
    )

    model.eval()
    total_nats = 0.0
    total_tokens = 0
    per_utterance = {}
    index = 0  # DataLoader is unshuffled, so this tracks dataset order
    with torch.no_grad():
        for batch in tqdm(loader, desc="Scoring dev-clean"):
            src_tokens = batch.src_tokens.to(DEVICE)
            tgt_tokens = batch.tgt_tokens.to(DEVICE)
            attention_mask = (src_tokens != tokenizer.pad_token_id).long()
            logits = model(src_tokens, attention_mask=attention_mask).logits
            losses = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tgt_tokens.view(-1),
                ignore_index=tokenizer.pad_token_id,
                reduction="none",
            ).view(tgt_tokens.shape)
            utt_nats = losses.sum(dim=1)
            n_real = (tgt_tokens != tokenizer.pad_token_id).sum(dim=1)
            for nats_i, count_i in zip(utt_nats.tolist(), n_real.tolist()):
                key = dataset.keys[index]
                per_utterance[key] = (nats_i / math.log(2), dataset.durations[index])
                total_nats += nats_i
                total_tokens += int(count_i)
                index += 1
    assert index == len(dataset), (index, len(dataset))
    return total_nats, total_tokens, per_utterance


def evaluate_unit(unit_name, spec):
    tokenizer = spec["tokenizer"]()
    ckpt = torch.load(spec["checkpoint"], map_location=DEVICE, weights_only=False)
    assert (
        ckpt["cfg"]["pad_token_id"] == tokenizer.pad_token_id
    ), "checkpoint/tokeniser pad_token_id mismatch"
    # Vocab-size guard. Without it, scoring a checkpoint trained under an older, smaller
    # vocabulary with the current tokeniser indexes past the embedding table and fails
    # as an opaque device-side "index out of bounds" CUDA assert, far from the cause --
    # exactly what happened after the vocab-collision fix. Fail here instead, with the
    # numbers.
    if ckpt["cfg"]["vocab_size"] != tokenizer.vocab_size:
        raise StaleCheckpoint(
            f"{unit_name}: checkpoint vocab_size={ckpt['cfg']['vocab_size']} but "
            f"current tokeniser has vocab_size={tokenizer.vocab_size}. This "
            f"checkpoint was trained under a different vocabulary and cannot be scored."
        )
    model = OPTForCausalLM(OPTConfig.from_dict(ckpt["cfg"]))
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE)

    batch_size = spec.get("batch_size", DEFAULT_BATCH_SIZE)
    total_nats, total_tokens, per_utterance = score_dev(
        model, tokenizer, spec["tier_name"], batch_size
    )
    total_bits = total_nats / math.log(2)
    duration = dev_duration(spec["tier_name"])

    # Per-utterance dump for utils/bootstrap.py. Cross-check first: the per-utterance
    # bits and durations must reproduce the aggregate, or the resampled intervals would
    # be about a different quantity than the reported rate.
    dumped_bits = sum(b for b, _ in per_utterance.values())
    dumped_dur = sum(d for _, d in per_utterance.values())
    assert abs(dumped_bits - total_bits) < 1e-6 * max(
        1.0, total_bits
    ), f"{unit_name}: per-utterance bits {dumped_bits} != total {total_bits}"
    assert (
        abs(dumped_dur - duration) < 1e-6
    ), f"{unit_name}: per-utterance duration {dumped_dur} != total {duration}"
    PER_UTTERANCE_DIR.mkdir(parents=True, exist_ok=True)
    (PER_UTTERANCE_DIR / f"{unit_name}.json").write_text(
        json.dumps({k: [b, d] for k, (b, d) in per_utterance.items()})
    )

    return {
        "bit_rate": total_bits / duration,
        "perplexity": 2 ** (total_bits / total_tokens),
        "total_tokens": total_tokens,
        "duration_s": duration,
    }


def read_existing_report():
    """{unit: (bit rate, perplexity, tokens)} as preformatted strings, parsed back out
    of an existing OUT_PATH table. Used so a partial run (`--units`) updates
    just its own rows instead of dropping every other unit's from the report.
    """
    if not OUT_PATH.exists():
        return {}
    rows = {}
    for line in OUT_PATH.read_text().splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0] != "unit" and not cells[0].startswith("-"):
            rows[cells[0]] = tuple(cells[1:])
    return rows


def write_report(results, merge=False):
    """Write the dev-clean table. With `merge`, rows for units not scored in this run
    are carried over from the existing report unchanged.
    """
    scored = {
        name: (
            f"{r['bit_rate']:.2f}",
            f"{r['perplexity']:.2f}",
            f"{r['total_tokens']:,}",
        )
        for name, r in results.items()
    }
    combined = {**(read_existing_report() if merge else {}), **scored}

    header = "| unit | bit rate (bits/s) | perplexity | dev-clean tokens |"
    sep = "|---|---:|---:|---:|"
    rows = [
        f"| {name} | {bit_rate} | {ppl} | {tokens} |"
        for name, (bit_rate, ppl, tokens) in combined.items()
    ]
    table = "\n".join([header, sep, *rows]) + "\n"
    lines = [
        "# Neural LM bit rate on dev-clean\n",
        "One row per trained unit; see llm/eval_bitrate.py for the exact "
        "definitions and the known mismatch with ngram/'s own numbers.\n",
        table,
    ]
    print("Writing:", OUT_PATH)
    OUT_PATH.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--units",
        nargs="+",
        metavar="UNIT",
        help="Score only these UNITS entries (default: all of them). Rows for "
        "every other unit are carried over from the existing report unchanged, "
        "so a partial run doesn't drop them.",
    )
    args = parser.parse_args()

    if args.units:
        unknown = [u for u in args.units if u not in UNITS]
        if unknown:
            raise SystemExit(
                f"Unknown unit(s): {', '.join(unknown)}. "
                f"Available: {', '.join(UNITS)}"
            )
        selected = {u: UNITS[u] for u in args.units}
    else:
        selected = UNITS

    results = {}
    for unit_name, spec in selected.items():
        if not spec["checkpoint"].exists():
            print(f"Skipping {unit_name}: no checkpoint at {spec['checkpoint']}")
            continue
        print(f"\n{unit_name}: reading checkpoint {spec['checkpoint']}")
        try:
            results[unit_name] = evaluate_unit(unit_name, spec)
        except StaleCheckpoint as exc:
            if args.units:  # explicitly asked for: this is an error, not a skip
                raise
            print(f"  Skipping: {exc}")
            continue
        r = results[unit_name]
        print(
            f"  bit rate {r['bit_rate']:.2f} bits/s, perplexity {r['perplexity']:.2f}, "
            f"{r['total_tokens']:,} dev-clean tokens, {r['duration_s']:.0f}s speech"
        )

    write_report(results, merge=bool(args.units))


if __name__ == "__main__":
    main()
