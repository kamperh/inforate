# Reproduce results

Exact commands to regenerate every result. Result numbers live in `results.md`,
`ngram/*_sweep.md` and `llm/eval_bitrate.md`, and the reasoning behind each
convention in `design.md` -- this file is commands only.

## Environment

```bash
conda create -n inforate python=3.14 -y
conda run -n inforate pip install tgt tqdm numpy
conda activate inforate
```

For the neural stages (`llm/`, `qwen/`), also:

```bash
pip install transformers tokenizers wandb
pip install "torch==2.13.0+cu126" --index-url https://download.pytorch.org/whl/cu126
```

The `torch` install must come from PyTorch's own CUDA-12.6 wheel index: the default
`pip install torch` resolves to a CUDA-13.0 build this machine's driver can't run.
Verify with `python -c "import torch; print(torch.cuda.is_available())"`.

To log training to Weights & Biases (the default, `wandb_mode="online"`, in every
`llm/<unit>_train.py`): `wandb login` once per machine (or set `wandb_mode="offline"`
in a script's `TrainConfig` to skip it and sync later).

## ngram/ stage

```bash
python ngram/librispeech_stats.py       # writes ngram/librispeech_stats.md
python ngram/ngram_lm.py                # toy fit-and-score sanity check
```

`librispeech_stats.py` also reports the realisation-based word vocabulary (the
`wordphn` unit), in its own section of the same file; re-run 2026-10-08 to add it. Its
`word_phone_units` is also what the two wordphn sweeps below build their tier from, and
its `destress` is what the phone sweep uses, so it is the one definition of both rules
inside `ngram/`.

Each sweep below trains n-gram LMs (orders 1-5) on 25/50/75/100% of pooled
LibriSpeech 960h (fixed shuffle seed 0, nested prefixes), evaluates bit rate and
perplexity on dev-clean, and writes its own `ngram/ngram_*_sweep.md`.

### Phone

```bash
python ngram/ngram_phone_sweep.py
```

Tokenised on the `phones` tier with stress stripped (39-symbol ARPABET set).

### Syllable

```bash
python ngram/ngram_syllable_sweep.py
```

Tokenised on the `syllables` tier; out-of-vocabulary syllables spelled out via their
constituent phones.

### Word

```bash
python ngram/ngram_word_sweep.py
```

### Word-BPE

```bash
python ngram/ngram_bpe_sweep.py --n-merges 200
python ngram/ngram_bpe_sweep.py --n-merges 1000
python ngram/ngram_bpe_sweep.py --n-merges 5000
```

Writes `ngram/ngram_bpe<n>_sweep.md` per run. `--n-merges` replaced a module constant
that had to be hand-edited before each run, matching `llm/word_bpe_prepare_mmap.py`'s
own flag.

All three re-run 2026-09-07 to fix an invalid spell-out bound: `spelling_of`
changed from
`token.split()` (constituent words) to `list(token)` (characters), matching
`ngram_word_sweep.py` and `llm/`. The old rule was an invalid bound, not just an
inconsistent one: an OOV word-BPE unit is always a single unseen word, so
splitting on space scored it straight to `<unk>` and the code never transmitted
which word it was.

### Wordphn

```bash
python ngram/ngram_wordphn_sweep.py
```

Tokenised on a *derived* tier: each `words`-tier interval written as its own destressed
phones joined by `-`, built from the `words` and `phones` tiers by
`sweep_common.load_wordphn_utterances`, since no TextGrid carries such a tier.
`WORDPHN_TIER` is accepted by `load_utterances` wherever a real tier name is, so the
sweep looks like every other one. Out-of-vocabulary realisations are spelled out via
their constituent phones, as for syllable. The derived intervals keep the `words`
tier's own times, so the denominator is bit-identical to the word sweep's.

### Wordphn-BPE

```bash
python ngram/ngram_wordphn_bpe_sweep.py --n-merges 200
python ngram/ngram_wordphn_bpe_sweep.py --n-merges 1000
python ngram/ngram_wordphn_bpe_sweep.py --n-merges 5000
```

Writes `ngram/ngram_wordphn_bpe<n>_sweep.md` per run. Whole-unit BPE with `wordphn`
units as the atomic symbols, so a unit is a collocation written out as phones; the same
`BPETokeniser` as word-BPE, given a different stream, and re-fit per fraction the same
way. Out-of-vocabulary units spell out as phones with the word boundary kept as its own
symbol; see `design.md`, "Per-unit spelling", for why the boundary has to survive and
why that branch never actually fires.

### Subwordphnx-BPE

```bash
python ngram/ngram_subwordphnx_bpe_sweep.py --vocab-size 200
python ngram/ngram_subwordphnx_bpe_sweep.py --vocab-size 2000
python ngram/ngram_subwordphnx_bpe_sweep.py --vocab-size 5000
python ngram/ngram_subwordphnx_bpe_sweep.py --vocab-size 50000
```

Writes `ngram/ngram_subwordphnx_bpe<v>_sweep.md`. Subword BPE over phones with every
merge confined to a word, using `bpe.py`'s second tokeniser,
`SubwordPhoneXBPETokeniser`, rather than the collocation one: it merges upwards from
phones where word-BPE and wordphn-BPE merge upwards from words. `--vocab-size` is named
to match `llm/subwordphnx_bpe_train.py`'s own flag, and the merge count is it minus the
39 phones and the 3 special tokens the neural stage reserves, so 2000 means the same
1,958 merges in both stages. Reads the derived `wordphones` tier, for the word grouping
the confinement needs. Nothing can go out of vocabulary, as for `phone`.

All four of the neural stage's vocabularies are run: 2000 on 2026-10-09, to ask whether
the estimator changes the verdict on the unit, then 200 and 5000 on 2026-10-12, to ask
the same of its vocabulary sweep, and 50000 on 2026-10-14, to find where that sweep
ends. About 9 minutes each. Each size is independent, so
they can be run in any order or in parallel; they were run sequentially to keep peak
memory down, vocabulary 200 being the longest piece stream of the four.

## llm/ stage

```bash
python llm/data_common.py     # self-test: tier reading, vocab, mmap round-trip
```

Each unit below: a `<unit>_prepare_mmap.py` step (builds any training-only vocab,
applies the unit's random spell-out augmentation where applicable, writes
`llm/data/<unit>-mmap/`), then `<unit>_train.py` (downsized OPT-125M, d=512/6
layers/8 heads, `lr_max=1e-3`, `ctx_win_size=2048`; `max_steps` and `batch_size` vary
by unit, noted below where they're not the default `10`/`4`). Pass `--tag <name>` to
give a variant run its own checkpoint instead of overwriting the canonical one; without
`--tag` or `--overwrite`, a script refuses to start if its checkpoint already has files.

### Phone

```bash
python llm/phone_prepare_mmap.py
python llm/phone_train.py
```

### Syllable

```bash
python llm/syllable_prepare_mmap.py
python llm/syllable_train.py
```

### Subword BPE

Non-paper sanity-check unit (ordinary subword BPE, vocab 2000) -- not one of the
project's four comparison units.

```bash
python llm/subword_bpe_prepare_mmap.py
python llm/subword_bpe_train.py
python llm/subword_bpe_sample.py   # Optional: sample continuations from a checkpoint
```

### Subwordphnx BPE

The same thing over phones instead of characters (same `BpeTrainer`), with merges
confined to a word so no piece crosses a word boundary, which is what makes it the
faithful counterpart of subword-BPE. Reads the derived `wordphones` tier, since that is
where the word grouping is; same phones, same seconds. Run at four vocabulary sizes,
200/2000/5000/50000.

```bash
python llm/subwordphnx_bpe_prepare_mmap.py --vocab-size 200 2000 5000 50000
for v in 200 2000 5000 50000; do
  python llm/subwordphnx_bpe_train.py --vocab-size $v
done
python llm/eval_bitrate.py --units subwordphnx-bpe-200 subwordphnx-bpe-2000 \
  subwordphnx-bpe-5000 subwordphnx-bpe-50000
```

`--vocab-size` takes several sizes in one prep run on purpose: deriving the
`wordphones` tier over 281,239 TextGrids is a quarter of an hour and the rest is
seconds, so one read serves every size. The train script takes one size at a time, as
`llm/word_bpe_train.py` does with `--n-merges`.

Check each prep's "Longest pieces" line: none may be longer than a single word's
realisation, which is the confinement holding. The comparisons that matter are against
`phone` and `wordphn`, the two units this family interpolates between, which
`utils/bootstrap.py` pairs only against `word`, so those need its `discover`/`report`
called directly. See `results.md`, "Subwordphnx-BPE".

The train script reads `batch_size`/`accumulation_steps` off `--vocab-size` rather than
fixing them, so 50000 trains at 2/20 and the smaller sizes at 10/4; the effective batch
is 81,920 tokens either way. 50000 also needs its own inference batch size, which
`llm/eval_bitrate.py` already carries. Per run on this machine: 50000 takes about 55
minutes against 200's 47, on a 2400-step budget against 5000.

### Word

```bash
python llm/word_prepare_mmap.py
python llm/word_train.py
python llm/word_sample.py   # Optional: sample continuations from a checkpoint
```

`batch_size=2`/`accumulation_steps=20`, not the default `10`/`4`: word's 58,006-token
vocabulary is the dominant memory cost in `cross_entropy`, independent of model size.

### Wordphn

Words identified by their realised phones rather than their spelling ("abbey" is
`AE-B-IY`), derived from the `words` and `phones` tiers by
`data_common.word_phone_intervals`; out-of-vocabulary realisations spelled out via
their constituent phones.

```bash
python llm/wordphn_prepare_mmap.py
python llm/wordphn_train.py
python llm/eval_bitrate.py --units wordphn
python utils/bootstrap.py --units neural:wordphn neural:word
```

`batch_size=2`/`accumulation_steps=20`, as for word: the 56,296-token vocabulary is
the dominant memory cost in `cross_entropy`. `--units wordphn` leaves every other row
in `llm/eval_bitrate.md` merged from the existing report rather than rescoring it.

### Wordphn-BPE

Word-level BPE over the wordphn unit, so a unit is a collocation written out as phones
("of the" is `AH-V DH-AH`). The merges are fitted on the wordphn stream itself, not
inherited from `llm/word_bpe_prepare_mmap.py`. Same three merge counts as word-BPE.

```bash
for n in 200 1000 5000; do
  python llm/wordphn_bpe_prepare_mmap.py --n-merges $n
  python llm/wordphn_bpe_train.py --n-merges $n
done
python llm/eval_bitrate.py --units wordphn-bpe-200 wordphn-bpe-1000 wordphn-bpe-5000
python utils/bootstrap.py --units neural:word \
  neural:wordphn-bpe-200 neural:wordphn-bpe-1000 neural:wordphn-bpe-5000
```

To list what was learned, at any of the three merge counts:

```bash
python utils/inspect_bpe.py                               # first 30 of 200 merges
python utils/inspect_bpe.py --n-merges 5000 --top 50
python utils/inspect_bpe.py --n-merges 1000 --all
python utils/inspect_bpe.py --n-merges 5000 --longest 15  # deepest merge chains
python utils/inspect_bpe.py --compare                     # the three counts together
python utils/inspect_bpe.py --n-merges 200 --verify       # diff against the n-gram path
```

Reads the saved tokeniser only; nothing is re-learned except under `--verify`, which
re-fits along `ngram/`'s path over 960h and takes several minutes. Glosses come from a
dev-clean read (`--no-gloss` skips it, `--gloss-split train-clean-100` widens it).

No `--tag`: word-BPE's `mincount2` suffix is historical, and there is no such
ambiguity for this unit. `batch_size=2`/`accumulation_steps=20` and eval
`batch_size=3`, as for word-BPE. The paired comparisons that matter are against
`wordphn` and the matching `word-bpe-*`, which `utils/bootstrap.py` does not generate
itself (it pairs everything against `word`), so those need its `discover`/`report`
called directly.

### Word-BPE

```bash
for n in 200 1000 5000; do
  python llm/word_bpe_prepare_mmap.py --n-merges $n
  python llm/word_bpe_train.py --n-merges $n --tag mincount2
done
```

The three merge counts match `ngram_bpe_sweep.py`'s own tested values, and
`--n-merges` must agree between the two commands. `max_steps` is computed at runtime
from that run's own token count. `batch_size=2`/`accumulation_steps=20`, same reason
as word. Uses `DTYPE=np.uint32`, not every other unit's `uint16`: word-BPE's
vocabulary exceeded `uint16`'s 65,535 max before `--min-count 2`. It no longer
does (58,206 to 63,002), but the existing mmaps are `uint32` and regenerating
them would mean retraining.

**`--tag mincount2` is required, not optional.** The tag is historical: it dates from
when `--min-count 2` was an experiment run alongside the untagged pre-threshold
checkpoints. It is now the default, so a fresh run needs no
flag to get the right *data* -- but `llm/eval_bitrate.py`'s `word-bpe-{200,1000,5000}`
entries point at `opt125m-word-bpe<n>-mincount2-librispeech`, so omitting the tag
writes a checkpoint the evaluation will not find.

### Evaluation

```bash
python llm/eval_bitrate.py
```

Scores every unit with an existing checkpoint (see `UNITS` at the top of the script)
against the full dev-clean set; writes `llm/eval_bitrate.md`.

## Seed variance

Four extra seeds per unit, in seven batches, oldest first below, so all fifteen units
have five runs each. `--seed` is a permanent flag on every `<unit>_train.py`
(default 0, the canonical value), defined once in `llm/train_args.py` alongside `--tag`,
`--lr-max`, `--epochs` and `--overwrite`; it seeds torch, numpy and python, so it varies
initialisation and the random training-window sampling, but not the augmentation draw
(fixed when the mmap is built) or the learned BPE merges.

```bash
for s in 1 2 3; do
  python llm/word_train.py         --seed $s --tag seed$s
  python llm/word_bpe_train.py --n-merges 200  --seed $s --tag seed$s
  python llm/word_bpe_train.py --n-merges 1000 --seed $s --tag seed$s
  python llm/syllable_train.py     --seed $s --tag seed$s
  python llm/word_bpe_train.py --n-merges 5000 --seed $s --tag seed$s
  python llm/phone_train.py        --seed $s --tag seed$s
  python llm/subword_bpe_train.py  --seed $s --tag seed$s
done
python llm/eval_bitrate.py
python utils/bootstrap.py --seeds
```

Twenty-one runs covering the seven units that existed then, about 19 hours
sequential (2026-09-07/08, all reaching max steps). `--seeds` reports per-unit mean
and SD, then judges every gap against `sqrt(sa^2+sb^2)`. Results and what they settle
are in `results.md`, "Seed variance".

The four wordphn units were seeded the same way on 2026-10-08/09, twelve runs, about
eleven hours, grouped by seed so an interrupted loop still leaves a complete seed:

```bash
for s in 1 2 3; do
  python llm/wordphn_train.py --seed $s --tag seed$s
  for n in 200 1000 5000; do
    python llm/wordphn_bpe_train.py --n-merges $n --seed $s --tag seed$s
  done
done
python llm/eval_bitrate.py --units wordphn-seed1 wordphn-seed2 wordphn-seed3 \
  wordphn-bpe-200-seed1 wordphn-bpe-200-seed2 wordphn-bpe-200-seed3 \
  wordphn-bpe-1000-seed1 wordphn-bpe-1000-seed2 wordphn-bpe-1000-seed3 \
  wordphn-bpe-5000-seed1 wordphn-bpe-5000-seed2 wordphn-bpe-5000-seed3
python utils/bootstrap.py --seeds
```

`--units` keeps every other checkpoint from being rescored; rows for a unit not named
are carried over from the existing report unchanged. These runs matter
more than the others did: two of the four units came back three times noisier than
anything previously measured, which overturned a result written up from seed 0 alone.
See `results.md`, "With collocation merging".

`subwordphnx-bpe-2000` was seeded on 2026-10-09, three runs, about 80 minutes:

```bash
for s in 1 2 3; do
  python llm/subwordphnx_bpe_train.py --vocab-size 2000 --seed $s --tag seed$s
done
python llm/eval_bitrate.py --units subwordphnx-bpe-2000-seed1 \
  subwordphnx-bpe-2000-seed2 subwordphnx-bpe-2000-seed3
python utils/bootstrap.py --seeds
```

The one subwordphnx size seeded then, chosen because it is the one the
vocabulary sweep and the cross-estimator comparison both hinge on.

One further seed per unit was run on 2026-10-12, fourteen runs, about twelve hours:
seed 4 for the twelve units that already had seeds 0-3, and seed 1 for the two
subwordphnx sizes that were still single runs. Cheapest unit first, and
`eval_bitrate.py` called after each training rather than once at the end, so an
interrupted batch still leaves complete rows:

```bash
python llm/subwordphnx_bpe_train.py --vocab-size 200  --seed 1 --tag seed1
python llm/subwordphnx_bpe_train.py --vocab-size 5000 --seed 1 --tag seed1
python llm/subwordphnx_bpe_train.py --vocab-size 2000 --seed 4 --tag seed4
python llm/subword_bpe_train.py --seed 4 --tag seed4
python llm/syllable_train.py    --seed 4 --tag seed4
for n in 5000 1000 200; do
  python llm/word_bpe_train.py --n-merges $n --seed 4 --tag seed4
done
for n in 5000 1000 200; do
  python llm/wordphn_bpe_train.py --n-merges $n --seed 4 --tag seed4
done
python llm/wordphn_train.py --seed 4 --tag seed4
python llm/word_train.py    --seed 4 --tag seed4
python llm/phone_train.py   --seed 4 --tag seed4
python utils/bootstrap.py --seeds
```

`llm/eval_bitrate.py --units <unit>-seed4` follows each training, with
`subwordphnx-bpe-200-seed1` and `subwordphnx-bpe-5000-seed1` for the first two. The
seed column in `results.md`'s headline table is the SD these give.

`subwordphnx-bpe-5000` was taken from two runs to five on 2026-10-13, three runs, about
an hour and a half:

```bash
for s in 2 3 4; do
  python llm/subwordphnx_bpe_train.py --vocab-size 5000 --seed $s --tag seed$s
  python llm/eval_bitrate.py --units subwordphnx-bpe-5000-seed$s
done
python utils/bootstrap.py --seeds
```

`subwordphnx-bpe-200` followed the same day, three runs, about two hours and twenty-five
minutes -- slower per run than the 5,000 size because a 200-piece vocabulary leaves the
longest token stream and so the largest step budget at the fixed 20 epochs. With these
every unit in `results.md`'s headline table has five runs:

```bash
for s in 2 3 4; do
  python llm/subwordphnx_bpe_train.py --vocab-size 200 --seed $s --tag seed$s
  python llm/eval_bitrate.py --units subwordphnx-bpe-200-seed$s
done
python utils/bootstrap.py --seeds
```

`subwordphnx-bpe-50000` was trained and seeded on 2026-10-14, four runs after the
canonical one, about three hours and forty minutes, so it never had a verdict written
against a borrowed or single-run floor. The prepare step is shared with the canonical
run and is not repeated:

```bash
for s in 1 2 3 4; do
  python llm/subwordphnx_bpe_train.py --vocab-size 50000 --seed $s --tag seed$s
done
python llm/eval_bitrate.py --units subwordphnx-bpe-50000-seed1 \
  subwordphnx-bpe-50000-seed2 subwordphnx-bpe-50000-seed3 subwordphnx-bpe-50000-seed4
python utils/bootstrap.py --seeds
```

Each unit's seed tuple is listed in `llm/eval_bitrate.py`'s `UNITS` comprehension, and
has to gain the new seed before `--units <unit>-seed<n>` will resolve.

The superseded dumps that used to sit in `llm/per_utterance/superseded/` -- the
learning-rate variants, the pre-augmentation-fix runs and their seeds -- were deleted on
2026-10-13 as no longer worth keeping. A default `utils/bootstrap.py` run globs only the
top level of `llm/per_utterance/`, so a stale dump put in a subdirectory is still
ignored rather than mixed with current ones. The pre-fix figures quoted in `results.md`,
"Seed variance", can no longer be regenerated without retraining those runs.

## Confidence intervals

Per-utterance bits are dumped by the two scoring paths, then resampled:

```bash
python llm/eval_bitrate.py                 # writes llm/per_utterance/<unit>.json
python ngram/per_utterance.py              # writes ngram/per_utterance/*.json
python utils/bootstrap.py                        # all units, both resampling schemes
python utils/bootstrap.py --list                 # what dumps exist
python utils/bootstrap.py --units neural:word neural:word-bpe-200
```

Order matters: `utils/bootstrap.py` reads the per-utterance dumps, so both scoring steps
must run first. `llm/eval_bitrate.py` writes its dumps as a side effect of the
ordinary evaluation above, so it need not be run twice.

`ngram/per_utterance.py` refits order 3 at 100% only (fourteen fits, ~1.5 h): each
(unit, order) pair is a separate fit, and that is the cell the headline claims use.
`--units` names a subset, which is how the four wordphn dumps were added on 2026-10-08
without recomputing the six already on disk, the two subwordphnx-BPE sizes on
2026-10-12 without recomputing the eleven, and vocabulary 50000 on 2026-10-14 without
recomputing the thirteen; a fit is deterministic given its training
data, so a dump only ever needs regenerating when the unit's tokenisation or spell-out
rule changes. The 50000 fit takes about seven minutes, most of it learning 49,958
merges.

`utils/bootstrap.py` reports marginal and paired intervals under both utterance-level
and speaker-level resampling; the paired speaker-level numbers are the ones to quote.
See `results.md`, "Confidence intervals", and `design.md`, "Uncertainty".

Any pair other than `X` against `word` needs `discover`/`report` called directly, since
the CLI generates comparisons only against `word` within a stage and across stages for
the same unit. The n-gram intervals in `results.md`, "Under n-grams" and "Merging under
n-grams", came from a few lines of this shape, with the whole set of units in one
`datasets` dict so that every contrast sits on the same resample (note the ngram labels
carry the `-order3-100%` suffix, and the `%` needs quoting in a shell):

```python
import sys; sys.path.insert(0, "utils")
from bootstrap import bootstrap, ci, discover

found = discover()
labels = ["ngram:wordphn-order3-100%", "ngram:wordphn-bpe-5000-order3-100%"]
point, rates, names = bootstrap({l: found[l] for l in labels}, by_speaker=True)
d = rates[:, 1] - rates[:, 0]
print(point[labels[1]] - point[labels[0]], ci(d))
```

## qwen/ stage

Reproduces the predecessor project's pretrained-LM "estimator comparison"
(`docs/assets/2026-08-30/paper_2.md` Sec 4.6) on LibriSpeech, at the subword-bpe "level"
(Qwen's own tokenizer, no training). See `qwen/eval_bitrate.py` for the full design.

```bash
python qwen/eval_bitrate.py --model Qwen/Qwen2.5-1.5B
python qwen/eval_bitrate.py --model Qwen/Qwen2.5-3B
```

Writes/updates a row in `qwen/eval_bitrate.md`. Both run 2026-09-04 (3B completed
cleanly despite tight GPU memory, no OOM): see `results.md`, "Qwen pretrained-LM
(side experiment)". Before the first real run, verified without loading the weights:
`python -m py_compile qwen/data_common.py qwen/eval_bitrate.py`; `python
qwen/data_common.py` (self-test); `python qwen/eval_bitrate.py --help`; and a
tiny-random-model smoke test of the scoring code
path (found and fixed one real `DynamicCache` bug).
