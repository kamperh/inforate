# inforate2

Which linguistic unit encodes speech most efficiently, measured in bits per second,
and does the answer depend on the modelling budget available to estimate it? The units
are phones, syllables, words, sub-word pieces and multi-word collocations found by BPE,
each in two flavours above the phone: identified by its spelling, or identified by its
realised phones (the `wordphn` family). All training and evaluation for this project run
on **LibriSpeech**.

## Background

This is a redo of an earlier project (`docs/assets/2026-08-30/`), inspired by Coupé
et al. (2019) reporting ~39 bits/s at the syllable level as roughly constant
across languages.
That predecessor built a careful n-gram estimator (`docs/assets/2026-08-30/inforate.py`,
written up in its `paper_2.md`, with handoff notes alongside) and found that unit
"efficiency" is not a linguistic constant but an artefact of the (unit, model class,
data budget) triple, using FLEURS speech and a Wikipedia/LLM-generated background corpus
for training data. A separate, earlier attempt at the same question was shelved as
inconclusive (referenced there as `projects/inforatex/`).

This project keeps the research question but changes the substrate to LibriSpeech
throughout, and will proceed as a systematic, staged set of experiments rather than one
big pipeline, starting with n-gram word-level LMs and working up to neural LMs. The
detailed experiment plan is drafted separately from this file.

## Directory layout

- `docs/assets/`: reference material, not code to run as-is.
  - `2026-08-30/`: the predecessor project's paper (`paper_2.md`; `paper.md` is an
    earlier superseded draft), reviewer comments, its own handoff notes, research diary
    `googledoc_prjects.md` and its estimator `inforate.py`. Treat as prior art: reuse
    its design lessons (below), not its Wikipedia/FLEURS data pipeline.
  - `opt125m-trainer-example/`: a reusable scaffold for training an OPT-125M causal LM.
    `trainer.py` is meant to be reused as-is; `data.py`/`train.py` are the parts to swap
    in per experiment. Its shipped example already does phones-on-LibriSpeech:
    `prepare_librispeech_mmap.py` flattens ARPABET phone ids from the alignment
    TextGrids into a flat mmap, `train.py` trains OPT-125M on fixed-length chunks of it
    with wandb logging. This was the template for the neural-LM stage; every
    other unit got its own tokeniser and data loader analogous to
    `tokenizer.py`/`data.py`, all of them now in `llm/`.
- `data/`: project data.
  - `librispeech-alignments/`: TextGrid forced alignments for all seven standard
    LibriSpeech splits: train-clean-100 (28539 utterances), train-clean-360 (104014),
    train-other-500 (148686), together the ~960h train set, plus
    dev-clean/dev-other/test-clean/test-other (2703/2864/2620/2939). Each TextGrid has
    three tiers, `words`, `phones` and `syllables`, all pre-segmented with per-interval
    timing (the `syllables` tier is already destressed and hyphen-joined); empty-text
    intervals are silence, already excluded from speech time. This gives transcripts,
    phone sequences, syllable sequences and speech-only durations directly: no raw
    audio, VAD, g2p or a separate syllabifier is needed for LibriSpeech-native
    experiments (contrast the predecessor project, which needed VAD and g2p for
    FLEURS).
- `ngram/`: n-gram experiment scripts, configs and outputs. `ngram_lm.py` (the ported,
  unit-agnostic estimator) and `bpe.py` (two BPE tokenisers: `BPETokeniser` merging
  whole words into collocations, `SubwordPhoneXBPETokeniser` merging phones into
  word-confined subword pieces) are shared by seven sweep scripts,
  `ngram_<unit>_sweep.py` for `word`, `syllable`, `phone`, `bpe`, `wordphn`,
  `wordphn_bpe` and `subwordphnx_bpe`,
  which produce fourteen sweeps between them (the two collocation scripts take
  `--n-merges` and the subword one `--vocab-size`), each covering 25/50/75/100% of
  LibriSpeech 960h times n-gram orders 1-5 and evaluated on dev-clean; shared sweep
  machinery lives in `sweep_common.py`, including the derived `WORDPHN_TIER`.
  `librispeech_stats.py` is the Step 1 data statistics script, and also supplies
  `destress` and `word_phone_units` to the sweeps. `per_utterance.py`
  refits order 3 at 100% for all fourteen units and dumps per-utterance bits to
  `per_utterance/`, for `utils/bootstrap.py`.
- `llm/`: neural-LM training. Adapted from the `docs/assets/opt125m-trainer-example/`
  scaffold, analogous to how `ngram/` was built from
  `docs/assets/2026-08-30/inforate.py`:
  the scaffold stays reference material, the actual code to run lives here. Shared
  modules (`data_common.py`, `tokenisers.py`, `data.py`, `trainer.py`,
  `train_args.py` for the flags every unit shares) plus one
  `<unit>_prepare_mmap.py`/`<unit>_train.py` pair per unit: `phone`, `subword_bpe`,
  `subwordphnx_bpe`, `syllable`, `word`, `word_bpe`, `wordphn` and `wordphn_bpe`, with
  the three BPE-merge counts coming from one `--n-merges` flag, and subwordphnx-BPE's
  four vocabulary sizes from one `--vocab-size` flag, rather than separate
  scripts. `word_sample.py` and `subword_bpe_sample.py` continue a word prompt from a
  trained checkpoint, as a diagnostic that nothing is scored from: word's prints tokens
  rather than decoded text, so a `~`-marked spell-out stays visible.
  `eval_bitrate.py` scores every checkpoint in its `UNITS` and also dumps
  per-utterance bits to `per_utterance/`, for `utils/bootstrap.py`, whose default glob
  reads only the top level of that directory, so a superseded dump can be parked in a
  subdirectory rather than deleted. The ones kept that way -- learning-rate variants and
  pre-augmentation-fix runs -- were pruned on 2026-10-13.
- `qwen/`: pretrained-LM side experiment, porting the
  predecessor project's "estimator comparison"
  (`docs/assets/2026-08-30/paper_2.md` Sec 4.6)
  to LibriSpeech: how far the absolute bit rate moves when the *model class* changes
  drastically, rather than the unit. `eval_bitrate.py` scores dev-clean with a
  pretrained Qwen2.5 (`--model`, run at 1.5B and 3B), Qwen's own tokeniser standing in
  for the subword-bpe level, so nothing is trained and no vocabulary is chosen here; a
  lowercase, unpunctuated primer drawn from train-clean-100 is prepended as shared
  context and never scored, so the model isn't charged for LibriSpeech's transcript
  register. `data_common.py` duplicates `llm/`'s tier reading rather than importing it,
  so `qwen/` doesn't depend on `llm/`'s layout. Writes `eval_bitrate.md`. Deliberately
  outside the controlled OPT-20M comparison: it answers "how much does the estimator
  matter", not "what is the true information rate of English", since Qwen's pretraining
  data is unmeasurable.
- `utils/`: analysis tooling that spans both stages.
  `bootstrap.py`: confidence intervals on bit rate, by resampling dev-clean from
  the per-utterance dumps the two scoring paths write. Reports marginal and **paired**
  differences under both utterance-level and speaker-level (40-cluster) resampling; the
  paired speaker-level numbers are the ones to quote, since marginal intervals are far
  wider than the differences being compared. See `design.md`, "Uncertainty".
  `inspect_bpe.py`: lists the wordphn-BPE merges in the order learned, read from the
  tokeniser `llm/wordphn_bpe_prepare_mmap.py` saved, nothing re-learned. Each unit is
  glossed with the spelling its realisation most often carries on dev-clean, since
  `AH-V DH-AH` is unreadable and "of the" is not, and a gloss marked `*` flags a
  realisation with more than one spelling. `--compare` puts the three merge counts side
  by side; they are nested, so 200's merges are the first 200 of 1000's and those the
  first 1000 of 5000's. `--verify` re-learns along the n-gram path and diffs, which is
  the check `design.md`, "Open issues", records as never having been run for this unit.
  `inspect_example.py`: shows one utterance as all fifteen units tokenise it, reading
  the saved tokenisers and loading no model, with the unit list and each unit's
  tokeniser taken from `llm/eval_bitrate.py`'s own `BASE` so the two cannot diverge.
  `--words N` truncates every unit at the Nth word's end time, so a short prefix stays
  aligned across units, and `--markdown` emits a table. Per-utterance bit rates are
  deliberately not printed: one utterance is far too little to average over, and the
  ordering it gives does not resemble the corpus one.

## Design principles carried over from the predecessor project

Hard-won lessons from `docs/assets/2026-08-30/notes.md` §3; follow these rather than
re-deriving them:

- **One fixed model class across every tokenisation** compared in a given
  experiment. No add-alpha smoothing: with small vocabularies and small contexts
  it makes higher-order models score worse than lower-order ones, which is a
  smoothing artefact, not a finding.
- **Every model assigns probability to whole sentences over an unbounded token
  set** (OOV handled via a spell-out/character fallback), so no utterance is ever
  skipped and comparisons across tokenisations stay paired on the same evaluation
  set.
- **Vocabulary built from training data only**, never train+test: using test data
  to build the vocabulary flatters large-unit tokenisations most, since they have
  the most unseen items.
- **Tokenisers (e.g. BPE merges) are factories re-fit per fold/split**, not fit
  once and reused, to avoid leakage across folds.
- **Held-out cross-entropy only**, never in-sample/plug-in entropy.
- **Shuffle a background corpus before sub-sampling** fractions of it, so corpus
  size isn't confounded with e.g. speaker/chapter order.

## Project documents

Three documents at the project root, each with one job. **Read `design.md` first**;
it is the authoritative technical reference. `paper.md` sits beside them as the
write-up drawn from them, and is the user's own: do not edit it.

- **`design.md`**: the current rules and the reasoning behind them. What is
  measured (bit rate, and why perplexity is not comparable), the data, the n-gram
  estimator with its equations, per-unit tokenisation and out-of-vocabulary
  backoff across both stages, the canonical neural training setup, the evaluation
  protocol, the standing design principles, and the open issues. When a
  convention changes, change it here.
- **`results.md`**: every result table. All of `ngram/`'s sweeps, plus an
  `OPT-20M` column on each bit rate table for the neural stage, a per-unit
  hyperparameters table, and a per-unit table of inventory sizes and token counts
  over 960h and dev-clean. Update by hand as each result lands, rather than
  regenerating.
- **`reproduce.md`**: the exact commands to regenerate every result, one section
  per experiment stage. Commands only, no commentary. Build it up incrementally.

`readme.md` is the user's own and should not be edited.

`docs/archive/` holds superseded material, kept rather than deleted since this is
not a git repository: the completed stage plans, the chronological implementation
log, the dated decisions log and operational traps, analyses that later results
overturned, and the pre-cleanup snapshots and removed scripts from the 2026-10-12
pass. Nothing outside `docs/archive/` should cite anything inside
it: the three documents above carry every convention and result that is still
current, and each must stand on its own. A new experiment stage would get its own
plan at the root, archived once complete.

**Markdown convention**: after editing any of these, check the pandoc export is
clean with `pandoc <file>.md -t latex -o /tmp/c.tex && grep -c 'real{' /tmp/c.tex`,
which must print `0`. Pandoc renders a table whose total header and cell text is
long enough with `\real{...}` column widths, which some LaTeX installs do not
define, so the PDF build then fails; the trigger is text length, not column
count.

## Environment

Primary environment: conda env `inforate`, Python 3.14 (`conda create -n inforate
python=3.14 -y`). Packages are added as each experiment stage needs them, starting
with the n-gram / word-count stage.

The neural stages were folded into `inforate` rather than run from
`docs/assets/opt125m-trainer-example/`'s own `uv` subproject: `transformers`,
`tokenizers`, `wandb` and a CUDA-12.6 `torch` wheel are installed into the same conda
env, so one environment runs everything. The scaffold's `pyproject.toml`/`uv.lock` are
reference material only. See `reproduce.md`, "Environment", for the exact installs.

`conda run` buffers stdout, which hides live training progress; call
`/home/kamperh/miniconda3/envs/inforate/bin/python -u` directly instead.

## Status

`ngram/` is complete: fourteen sweeps run for real on LibriSpeech
960h -- word, syllable, phone, word-BPE at 200/1000/5000 merges, wordphn,
wordphn-BPE at the same three merge counts, and subwordphnx-BPE at vocabulary
200/2000/5000/50000 -- each covering
25/50/75/100% of the data times orders 1-5, evaluated on dev-clean. Results and a
cross-unit summary are in `results.md`. The three word-BPE sweeps were re-run on
2026-09-07 to fix an invalid spell-out bound (their out-of-vocabulary rule never
transmitted *which* unseen word occurred, so it was too cheap, not merely
inconsistent). subwordphnx-BPE at vocabulary 2000 was added on 2026-10-09 for the
cross-estimator comparison, 200 and 5000 on 2026-10-12 to put a deterministic
estimator under the vocabulary sweep the seeds had left unsettled, and 50000 on
2026-10-14 to find where that sweep ends.

`llm/` is complete: all fifteen units trained and scored -- phone, subword-bpe,
subwordphnx-bpe at 200/2000/5000/50000, word, syllable, word-BPE at
200/1000/5000, wordphn, and wordphn-BPE at 200/1000/5000 -- at the canonical setup of
20 epochs and `lr_max=1e-3`, which the learning-rate and epoch sweeps confirmed is
best or tied-best for every unit, so the "one setup for every level" goal was met.
subwordphnx-bpe-50000 is the one unit whose `batch_size`/`accumulation_steps` follow
its `--vocab-size` rather than being fixed, 2/20 rather than 10/4, since the softmax
no longer fits otherwise; the effective batch is unchanged.
Numbers live in `results.md` and `llm/eval_bitrate.md`.

`qwen/` is complete: Qwen2.5-1.5B and 3B scored on dev-clean.

Bit rates carry 95% confidence intervals (`utils/bootstrap.py`) and, for `llm/`, a
seed noise floor from **five seeds for every one of the fifteen units**, the last of
them on 2026-10-14; an n-gram fit is deterministic, so
there the bootstrap interval is the whole error budget. Training variance is about 0.1
bits/s for a difference between independently trained units, and **the floor is not
universal**: SDs run 0.015 to 0.118 over eleven of the fifteen units, across
vocabularies from 41 to 63,002 tokens, but subwordphnx-BPE-5000, wordphn-BPE-1000,
wordphn-BPE-5000 and subwordphnx-BPE-2000 come in at 0.157, 0.235, 0.235 and 0.279. A
new unit's floor has to be measured rather than inherited, and subwordphnx-BPE-2000 is
the project's sharpest demonstration of it: written up from one run against a borrowed
0.109, it measured 0.279 and took most of its own subsection's verdicts down with it.
**Inferring a floor from two runs is no safer.** subwordphnx-BPE at 200 and
5000 read 0.015 and 0.056 over two runs each; taking both to five confirmed 200 at
0.015, the quietest unit in the project, and measured 5000 at 0.157, nearly three times
its two-run spread, which cost two of the four comparisons the second run had bought --
with nothing in the two-run numbers to mark the one that would hold. **And a measured
floor applied to one run is not safe either**: subwordphnx-BPE-50000 reads 0.18 bits/s
below wordphn at seed 0 and 0.03 above it on five-seed means, seed 0 being wordphn's
worst run of five and tied-best for the new unit, so the canonical run alone would have
made it the first unit to beat its own word-level counterpart from below. Quote means
wherever a pair sits within about twice its floor.
**One tokenisation holds the whole range of floors**: the four subwordphnx-BPE
vocabularies measure 0.015, 0.279, 0.157 and 0.062, so the project's quietest unit and
its noisiest differ only in vocabulary size, and the floor is not monotone in it.
Of the 21 pairwise comparisons among the
original seven units, sixteen
are clear; the full set over fifteen units is 105, with the wordphn family's own in
`results.md`, "Wordphn". Over the four paper units the defensible ranking is

    word < word-BPE-200 < word-BPE-1000 < {syllable, word-BPE-5000} < phone

with the braced pair **unordered** (1.1 sigma); every other adjacent step is
clear. subword-BPE cannot be placed among the word-BPE points at all. Reported
numbers remain the `seed = 0` runs; the headline table carries each unit's seed SD
beside them. See `results.md`.

**Spelling a word out is cheaper than pronouncing it.** `wordphn` keeps word's spans
but identifies each by its realised phones, and costs 0.70 bits/s for it at seed 0 and
0.58 on five-seed means. Collocation merging costs more on top, as it does for word,
and **not more sharply**: an earlier write-up claimed a depth-dependent interaction,
which the seeds and the n-gram sweeps independently dissolved (reversal contrast
-0.008 [-0.031, +0.016] under n-grams, +0.084 against a floor of 0.346 on the means).

`subwordphnx-bpe`, subword BPE over phones with merges confined to a word (which is
what `subword-bpe`'s own pre-tokeniser does), lands at **28.82 bits/s** at vocabulary
2000, between wordphn and phone -- but five seeds of it give SD **0.279**, the worst in
the project, which leaves it unseparated from phone (2.8x the floor at seed 0, 2.0x on
means) and clearly worse than wordphn only on the means (3.2x, against 2.3x at seed 0).
Chunking a word into sub-word pieces costs about 0.7 bits/s whether the pieces are
letters or phones at seed 0, though on means the two read +0.612 and +0.915. Swept over
vocabulary at 200/2000/5000/50000 it descends from phone to level with wordphn, 29.04 to
28.82 to 28.52 to 27.98, and the ordering holds in every bootstrap resample. **Every
size but 2000 is clear of phone** -- 9.8x, 6.5x and 19.8x its floor against that unit's
own 2.8x -- while the step from 5000 to 50000 is 3.2x at seed 0 and 3.8x on the means,
and both
steps through vocabulary 2000 stay inside its
0.573 seed range, so the monotonicity through the middle is still a property of one draw
at that size. It descends partly by transmitting more of where the words divide,
since the share of words emitted as one piece grows 47.3% to 98.8% across the sweep, so
it is not a pure unit-size effect either, and the same confound means its gap to phone
is not a clean modelling comparison. **The direction is also independent of the
seeds**: all four sizes were swept under n-grams, on 2026-10-12 and 2026-10-14, where a
fit is deterministic, and the descent is monotone with every step clear (-4.065, -0.543,
-0.296).
End to end that n-gram descent is between four and five times the neural one and
front-loaded where the neural one
is not, so it corroborates the ordering and not the magnitudes -- and the ratio inverts
by the last step, where the neural descent is nearly twice the n-gram one. More seeds at
2000 are
what would settle the inner steps. **The two estimators disagree about where the sweep
ends.** At vocabulary 5000 the n-gram unit is
indistinguishable from wordphn (-0.005 [-0.118, +0.098]) where the neural unit stays
0.60 bits/s above it at 3.5x the floor on means (2.1x at seed 0); at 50000 the n-gram
unit is 0.30 below wordphn and **level with `word`** (+0.130 [-0.015, +0.263], the only
non-significant comparison in that section), while the neural unit is level with wordphn
(0.033 on means, 0.3x) and still +0.550 above word at 7.3x. At that size the unit is
wordphn with a subword backoff rather than an interpolation towards it: 98.8% of words
are one piece and its stream is *shorter* than wordphn's, since wordphn spells an unseen
realisation out phone by phone. So a good part of wordphn's penalty against word is its
spell-out and not realisation-identity, under n-grams at least. See `results.md`,
"Wordphn" and "Subwordphnx-BPE".

**Spell-out pieces are emitted pre-marked**, and must stay that way. A bare piece
encodes to a whole-unit id, because every letter is itself a LibriSpeech word, so
emitting bare pieces produced no `~`-marked fallback tokens at all for word and
word-BPE and train/test-inconsistent hybrids for syllable; that was the state until
2026-09-21 and every augmented unit was retrained after the fix. Nine of the fifteen
units augment; phone, subword-bpe and the four subwordphnx units have nothing to
spell out. See `design.md`, "Random augmentation".

**Collocations buy nothing over plain words**, under either estimator, and both
stages agree more merges is worse. Two conventions keep that honest and must not be
relaxed: word-BPE carries word and syllable's `--min-count 2` threshold (without it
word-BPE kept 31,071 hapax types they discard, worth 3.69 to 4.99 bits/s), and an
n-gram spell-out has to transmit *which* unseen unit occurred (word-BPE's did not
until 2026-09-07, an invalid bound worth 0.24 bits/s the other way). See `design.md`,
"Vocabulary rules" and "Per-unit spelling".

## Writing

- Use British spelling (colour) with -ise (recognise).
- Use the em-dash sparingly.
- Use the Oxford comma sparingly: only in cases where it improves clarity.
- Wrap markdown files at column 88.


## Python

- Don't use hyphens for section headings in Python. E.g. don't have
  `# --- General training control ---`, just use `# General training control`
- Use two spaces before an inline comment.
- If a comment is a short phrase, don't end with a full stop.
- Start all comments (inline or on single lines) with a capital letter.
