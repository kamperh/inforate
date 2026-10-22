# Results

Every result table in the project. The n-gram sweeps (word, syllable, phone, both
collocation families at 200/1000/5000 merges, wordphn, and subwordphnx-BPE at
vocabulary 200/2000/5000/50000) mirror `ngram/ngram_*_sweep.md`; each bit-rate
table carries an `OPT-20M` column for the neural stage, filled in at the 100% row
only, from `llm/eval_bitrate.md`. Confidence intervals and seed spreads are in their
own sections below.

Subword-BPE and the four subwordphnx-BPE sizes are trained and scored too, and are
sanity-check units rather than ones the research question is about; they are reported
separately throughout. All four subwordphnx sizes have an n-gram counterpart;
subword-BPE does not. wordphn, a word-sized unit defined
by its realised phones rather than its spelling, has both stages but is likewise
reported separately, since it was added after the four paper units were settled, as is
its own collocation family wordphn-BPE.

Conventions and their reasoning are in `design.md`, exact commands in
`reproduce.md`, superseded material in `docs/archive/`.

## Headline results

Bit rate on dev-clean, in bits/s, lower is better. Every number below also appears in
its own section later, with the full sweep it came from; this table just puts them in
one place. n-gram cells are the 100%-data fits from `ngram/ngram_*_sweep.md`; the
`OPT-20M` column is `llm/eval_bitrate.md`, at the canonical 20 epochs and
`lr_max=1e-3`, seed 0.

| unit | ngram, order 3 | ngram, order 5 | neural, OPT-20M | seed SD |
|---|---:|---:|---:|---:|
| phone | 43.65 | 34.93 | 29.60 | 0.054 |
| subword-BPE | n/a | n/a | 28.20 | 0.118 |
| subwordphnx-BPE-200 | 35.66 | 32.87 | 29.04 | 0.015 |
| subwordphnx-BPE-2000 | 31.59 | 31.67 | 28.82 | 0.279 |
| subwordphnx-BPE-5000 | 31.05 | 31.24 | 28.52 | 0.157 |
| subwordphnx-BPE-50000 | 30.75 | 31.02 | 27.98 | 0.062 |
| syllable | 31.68 | 31.82 | 28.41 | 0.074 |
| word | 30.62 | 30.91 | **27.46** | 0.043 |
| wordphn | 31.05 | 31.34 | 28.16 | 0.073 |
| wordphn-BPE-200 | 30.97 | 31.23 | 28.35 | 0.115 |
| wordphn-BPE-1000 | 31.05 | 31.26 | 28.48 | 0.235 |
| wordphn-BPE-5000 | 31.32 | 31.49 | 29.58 | 0.235 |
| word-BPE-200 | **30.52** | **30.78** | 27.76 | 0.064 |
| word-BPE-1000 | 30.59 | 30.81 | 28.06 | 0.063 |
| word-BPE-5000 | 30.87 | 31.05 | 28.56 | 0.074 |

`seed SD` is the spread of that `OPT-20M` cell over **five independently trained
seeds**, the canonical seed-0 run plus four retrains at the same setup, run
2026-09-07/08, 2026-10-08/09, 2026-10-12/13 and 2026-10-14. It is the floor a
difference between two units has to clear, and it is not a constant: across all fifteen
units it runs from 0.015 to 0.279, a factor of eighteen, and a unit's own vocabulary
sweep spans the whole of that range. The
n-gram columns have no such column because an n-gram fit is deterministic, which makes
the bootstrap interval its whole error budget. Per-seed
values, pairwise floors and what each comparison then establishes are in "Seed
variance".

Order 3 is the reported n-gram cell, since it is where ten of the fourteen sweeps
bottom out; order 5 is shown alongside to make clear the ordering does not depend on
that choice. **Which order a unit bottoms out at tracks its granularity**, and nothing
else: the ten word-sized units all reach their floor at order 3, syllable and
subwordphnx-BPE-2000 one order further out, and phone and subwordphnx-BPE-200 at order
5. Only phone is still falling steeply there, which is why its two n-gram numbers differ
so much; subwordphnx-BPE-200 is flat from order 4 (32.88 against 32.87). All four
subwordphnx sizes were swept under n-grams, to see whether the estimator changes the
verdict on the unit and on its vocabulary sweep. subword-BPE is the one unit with no
n-gram counterpart, being a sanity-check unit rather than one the research question is
about.

**word is the most efficient unit under the neural estimator**, and the ordering over
the four paper units is

> word < word-BPE-200 < word-BPE-1000 < {syllable, word-BPE-5000} < phone

with the braced pair statistically unordered. The n-gram stage agrees on everything
except word itself, which it places third behind word-BPE at 200 and 1000 merges, by
0.10 and 0.03 bits/s. Both stages agree that within word-BPE, more merges is
monotonically worse. Confidence intervals are in "Confidence intervals", the seed
noise floor in "Seed variance", and what the comparisons do and do not establish in
both.

**subwordphnx-BPE sits between wordphn and phone**, at 28.82 bits/s at vocabulary
2,000, but five seeds of it make it the noisiest unit in the project (SD 0.279), which
leaves it clearly worse than word and **unseparated from both of its neighbours**.
Swept over vocabulary it descends from phone towards wordphn, 29.04 at 200 merges to
28.52 at 5,000, and each end's gap to phone is clear. The end-to-end descent is 3.3x
its floor at seed 0 but 2.5x on the means, and no step through vocabulary 2,000 clears
anything at all, that unit's own spread of 0.573 covering the whole descent.
It also descends partly by transmitting more and more of where the words divide, so it
is not a pure unit-size effect either. **Under n-grams the same unit beats phone by
12.06 bits/s rather than 0.78**, which is where the two estimators disagree most
sharply anywhere in the project: chunking buys a weak estimator the context it cannot
afford and buys a neural LM almost nothing. See "Subwordphnx-BPE".

wordphn keeps word's unit boundaries but identifies each unit by its realised phones,
and costs 0.70 bits/s for it at seed 0, 0.58 on five-seed means, so **spelling out a
word is cheaper than pronouncing it**. Merging collocations into it costs a further
0.19, 0.33 and 1.42 bits/s at 200, 1000 and 5000 merges, so "more merges is worse"
holds at the realisation level too.

**Whether it holds more sharply there than for word is not established, and an earlier
write-up of these numbers claimed it did.** Two things independently undo that claim.
wordphn-BPE-1000 and wordphn-BPE-5000 turn out to be three times noisier than a typical
unit here (seed SDs 0.235 each), and seed 0 happens to sit at the bottom of one
distribution and the top of the other, which is most of the apparent effect. And the
n-gram sweeps, which have no training noise at all, put the realisation penalty at a
flat +0.43 to +0.46 bits/s at every merge depth with a reversal contrast of -0.008
[-0.031, +0.016]. See "Wordphn".

## Neural LM hyperparameters

Per-unit training configuration for `llm/<unit>_train.py`, since these diverge across
units (word needed a smaller `batch_size`, see design.md, "Neural training setup") and
are otherwise easy to lose track of across eight separate scripts. Every unit shares
`d=512`, `6` layers, `8` heads, `ffn_dim=2048`, an effective batch size (`batch_size x
accumulation_steps`) of `40`, a context window of `2048` tokens and `lr_max=1e-3`, per
`CLAUDE.md`'s "one fixed model class" principle; only what actually varies is broken
out below. `vocab_size` (and so `params`) is data-dependent per unit, not a free
choice. Word-BPE carries `--min-count 2`, matching word and syllable.

Two narrower tables rather than one wide one: a single 8-column table pushes pandoc's
PDF export into a percentage-width column layout that isn't reliably supported across
LaTeX installs.

`params` counts tied input/output embeddings **once**: every unit uses
`tie_word_embeddings=True`, so a raw `state_dict` sum double-counts the embedding
matrix.

| unit | vocab_size | params | status |
|---|---:|---:|---|
| phone | 41 | 19,985,920 | trained |
| subword-bpe | 2,000 | 20,988,928 | trained |
| subwordphnx-bpe-200 | 200 | 20,067,328 | trained |
| subwordphnx-bpe-2000 | 2,000 | 20,988,928 | trained |
| subwordphnx-bpe-5000 | 5,000 | 22,524,928 | trained |
| subwordphnx-bpe-50000 | 50,000 | 45,564,928 | trained |
| word | 58,006 | 49,664,000 | trained |
| wordphn | 56,296 | 48,788,480 | trained |
| syllable | 11,427 | 25,815,552 | trained |
| word-BPE (200 merges) | 58,206 | 49,766,400 | trained |
| word-BPE (1000 merges) | 59,006 | 50,176,000 | trained |
| word-BPE (5000 merges) | 63,002 | 52,221,952 | trained |
| wordphn-BPE (200 merges) | 56,496 | 48,890,880 | trained |
| wordphn-BPE (1000 merges) | 57,296 | 49,300,480 | trained |
| wordphn-BPE (5000 merges) | 61,294 | 51,347,456 | trained |

`params` grows almost entirely from the embedding layer, not the model body: every
unit shares the identical **19,964,928-parameter body** (the 6 transformer layers,
attention, layer norms -- everything except the embedding), verified directly via
`sum(p.numel() for p in model.parameters())` at each `vocab_size` with the embedding
excluded. `lm_head` is tied to the input embedding (`OPTForCausalLM`'s default), so
it's counted once, not twice; that one embedding matrix is `vocab_size x 512`. So:

```
params = 19,964,928 + vocab_size x 512
```

exactly, for every row above -- e.g. phone's 41-token vocab adds a negligible ~21K
params, word's 58,006-token vocab adds ~29.7M. This is also why the word-BPE units
(the largest vocabularies of the fifteen) have the most parameters despite using the
same architecture as phone. subword-bpe and subwordphnx-bpe-2000 land on the same
count because they are the same size by design, 2,000 pieces each.

| unit | batch_size | accumulation_steps | max_steps | warmup_steps |
|---|---:|---:|---:|---:|
| phone | 10 | 4 | 8200 | 656 |
| subword-bpe | 10 | 4 | 3300 | 264 |
| subwordphnx-bpe-200 | 10 | 4 | 5000 | 400 |
| subwordphnx-bpe-2000 | 10 | 4 | 3200 | 256 |
| subwordphnx-bpe-5000 | 10 | 4 | 2800 | 224 |
| subwordphnx-bpe-50000 | 2 | 20 | 2400 | 192 |
| word | 2 | 20 | 2600 | 208 |
| wordphn | 2 | 20 | 2500 | 200 |
| syllable | 10 | 4 | 3400 | 272 |
| word-BPE (200 merges) | 2 | 20 | 2300 | 184 |
| word-BPE (1000 merges) | 2 | 20 | 2200 | 176 |
| word-BPE (5000 merges) | 2 | 20 | 2000 | 160 |
| wordphn-BPE (200 merges) | 2 | 20 | 2300 | 184 |
| wordphn-BPE (1000 merges) | 2 | 20 | 2200 | 176 |
| wordphn-BPE (5000 merges) | 2 | 20 | 2000 | 160 |

Word-BPE's merge count is a `--n-merges` flag on `llm/word_bpe_prepare_mmap.py`/
`llm/word_bpe_train.py`, not a fixed choice (matching `ngram/ngram_bpe_sweep.py`'s own
200/1000/5000); every output filename embeds it, so multiple merge counts' outputs
coexist without colliding. `vocab_size`/`params`/`max_steps`/`warmup_steps` all vary
by merge count (more merges -> larger vocabulary but *fewer* training tokens, since
word-BPE merges reduce token count); all three tested merge counts are now trained.
`subwordphnx-bpe` is parametrised the same way, by a `--vocab-size` flag rather than a
merge count, and its four sizes are under "Vocabulary size". It is also the one unit
whose `batch_size`/`accumulation_steps` are not fixed by the script but chosen from
`--vocab-size`: at 50,000 the softmax no longer fits at `10`/`4`, so that size takes
word's `2`/`20`. The effective batch is `40` either way, so the canonical setup is
unchanged and only the memory profile differs.

`llm/eval_bitrate.py` needed its own, separate per-unit `batch_size` too (inference,
not training, but the same vocab-driven memory issue applies): `20` for phone,
subword-bpe and the three smaller subwordphnx units, `4` for word, wordphn and
subwordphnx-bpe-50000, `3` for word-BPE (all three merge counts) and wordphn-BPE.

## Uncertainty

Every number in this document is one point estimate from one run. Two independent
sources of error bound what the comparisons can claim: resampling the evaluation set,
and retraining at a different seed. Both are below; `design.md`, "Uncertainty", gives
the reasoning and the verdict thresholds (ratio to the floor: 3x clear, 2-3x marginal,
under 2x within noise). Exploratory `--tag` variants are not reported anywhere in this
document: learning-rate and epoch sweeps settled the canonical setup (`lr_max=1e-3`,
20 epochs) before these results were produced, and `design.md`, "Neural training
setup", records that outcome.

### Confidence intervals

`utils/bootstrap.py` attaches 95% intervals by resampling dev-clean, reading the
per-utterance bits that `llm/eval_bitrate.py` and `ngram/per_utterance.py` dump. B =
10,000, seed 0. n-gram coverage is order 3 at 100% data, the cell the headline claims
use. This section covers the four paper units plus syllable and phone; the wordphn
family's own intervals, both stages, are under "Wordphn", keeping that family's tables
together, and subwordphnx-BPE's four sizes are under "Subwordphnx-BPE".

**Marginal intervals are wide, and comparing them would be badly misleading.**
Resampling the 40 dev-clean speakers gives roughly +/-0.8 bits/s on any single
unit:

| unit | neural | 95% CI | ngram (order 3) | 95% CI |
|---|---:|---|---:|---|
| word | 27.46 | [26.64, 28.27] | 30.62 | [29.69, 31.56] |
| word-BPE-200 | 27.76 | [26.94, 28.58] | 30.52 | [29.59, 31.45] |
| word-BPE-1000 | 28.06 | [27.24, 28.88] | 30.59 | [29.66, 31.51] |
| word-BPE-5000 | 28.56 | [27.72, 29.37] | 30.87 | [29.96, 31.78] |
| syllable | 28.41 | [27.61, 29.21] | 31.68 | [30.77, 32.58] |
| phone | 29.60 | [28.76, 30.43] | 43.65 | [42.50, 44.77] |

Those intervals overlap almost completely, yet the comparisons are decisive.
Every unit scores the *identical* 2,703 utterances, so per-utterance difficulty is
shared and cancels: the right statistic is the **paired** difference, recomputed on
the same resample. Speaker-level (cluster) intervals, the conservative ones:

| comparison | difference | 95% CI | significant |
|---|---:|---|---|
| neural: word - word-BPE-200 | -0.30 | [-0.36, -0.24] | yes |
| neural: word - word-BPE-1000 | -0.60 | [-0.67, -0.54] | yes |
| neural: word - word-BPE-5000 | -1.10 | [-1.17, -1.03] | yes |
| neural: word - subword-bpe | -0.74 | [-0.88, -0.60] | yes |
| neural: word - syllable | -0.95 | [-1.06, -0.84] | yes |
| neural: word - phone | -2.14 | [-2.33, -1.96] | yes |
| ngram: word - word-BPE-200 | +0.10 | [0.06, 0.14] | yes |
| ngram: word - word-BPE-1000 | +0.03 | [-0.02, 0.09] | **no** |
| ngram: word - word-BPE-5000 | -0.24 | [-0.32, -0.16] | yes |
| ngram: word - syllable | -1.05 | [-1.19, -0.91] | yes |

Cross-stage, same unit, n-gram minus neural: word 3.16 [2.94, 3.39], word-BPE-200
2.76 [2.53, 2.99], word-BPE-1000 2.52 [2.30, 2.75], word-BPE-5000 2.31 [2.08,
2.54], syllable 3.26 [3.10, 3.43], phone 14.05 [13.54, 14.56]. All significant, so
the neural stage genuinely beats the n-gram stage at every unit.

**What this settles.** Word's neural lead over every other unit is real, including
the 0.30 bits/s over word-BPE-200 that motivated this. Under n-grams the picture is
finer: word-BPE-200 is *significantly* better than word, but by 0.10 bits/s, about
0.3% -- significant and negligible at the same time, which is what 2,703 paired
utterances buys you. word-BPE-1000 is a genuine tie.

**What it does not settle.** The bootstrap treats each checkpoint as fixed, so it
says nothing about neural training variance: whether a retrain at another seed
would move word's 0.30 bits/s lead. For `ngram/` there is nothing further to
measure, since those fits are deterministic given their training data. Utterance
level intervals are also reported by the script and are roughly four times
narrower; the speaker-level ones above are the honest numbers, since utterances
within a speaker are not independent.

### Seed variance

The bootstrap above covers evaluation uncertainty but holds each checkpoint fixed. Four
extra seeds were run for every unit, everything else canonical: 20 epochs,
`lr_max=1e-3`, same mmaps. Five seeds per unit for all fifteen, once the existing
seed-0 run is included. The seven original units were seeded on 2026-09-07/08;
the four wordphn units on 2026-10-08/09, after their seed-0 results had already been
written up against an *assumed* floor of 0.07, what that assumption cost being under
"With collocation merging"; subwordphnx-BPE-2000 on 2026-10-09, which cost rather more;
and subwordphnx-BPE at 200 and 5,000 from two runs each to five on 2026-10-13, which is
also when the fifth seed of every other unit was run. subwordphnx-BPE-50000 was trained
and seeded on 2026-10-14, five runs from the start, so no verdict about it was ever
written against a borrowed floor.

| unit | s0 | s1 | s2 | s3 | s4 | mean | SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| phone | 29.60 | 29.61 | 29.59 | 29.49 | 29.52 | 29.562 | 0.054 |
| subword | 28.20 | 27.96 | 27.99 | 28.11 | 28.22 | 28.096 | 0.118 |
| sphnx-200 | 29.04 | 29.06 | 29.07 | 29.08 | 29.06 | 29.065 | 0.015 |
| sphnx-2000 | 28.82 | 28.76 | 29.24 | 29.33 | 28.76 | 28.982 | 0.279 |
| sphnx-5000 | 28.52 | 28.60 | 28.82 | 28.86 | 28.56 | 28.670 | 0.157 |
| sphnx-50000 | 27.98 | 27.98 | 28.01 | 28.09 | 28.11 | 28.034 | 0.062 |
| syllable | 28.41 | 28.33 | 28.40 | 28.38 | 28.53 | 28.410 | 0.074 |
| word | 27.46 | 27.50 | 27.55 | 27.45 | 27.45 | 27.484 | 0.043 |
| wBPE-200 | 27.76 | 27.94 | 27.83 | 27.87 | 27.84 | 27.848 | 0.064 |
| wBPE-1000 | 28.06 | 28.12 | 28.06 | 28.22 | 28.11 | 28.115 | 0.063 |
| wBPE-5000 | 28.56 | 28.54 | 28.56 | 28.39 | 28.57 | 28.523 | 0.074 |
| wordphn | 28.16 | 28.07 | 28.07 | 27.95 | 28.08 | 28.066 | 0.073 |
| wphn-200 | 28.35 | 28.21 | 28.11 | 28.27 | 28.40 | 28.268 | 0.115 |
| wphn-1000 | 28.48 | 28.82 | 28.99 | 29.01 | 28.59 | 28.781 | 0.235 |
| wphn-5000 | 29.58 | 29.40 | 29.01 | 29.31 | 29.07 | 29.274 | 0.235 |

Row labels are abbreviated: `subword` is subword-BPE, `sphnx-k` subwordphnx-BPE at
vocabulary k, `wBPE-k` word-BPE at k merges and `wphn-k` wordphn-BPE at k merges, as
in the comparison tables below.

**The floor is architecture-independent over eleven of the fifteen units, and not
beyond them.** Those eleven run from 0.015 to 0.118 across vocabularies from 41 to
63,002 tokens and both batch configurations, so about **0.1 bits/s is the floor for a
difference** between two of them, `sqrt(sa^2 + sb^2)`. That range is eightfold rather
than the threefold it read before subwordphnx-BPE-200 was taken to five runs, but the
new extreme is at the quiet end: 0.015 over a range of 0.042, the tightest unit in the
project, which only makes differences involving it cheaper to establish, so 0.1 stays a
conservative floor. The other four are worse by a factor of two to six:
subwordphnx-BPE-5000 at 0.157, wordphn-BPE-1000 and wordphn-BPE-5000 at 0.235 each and
subwordphnx-BPE-2000 at **0.279**, with ranges of 0.340, 0.525, 0.569 and 0.573 bits/s.
A difference involving one of them must clear about 0.16 to 0.28, and a difference
between two of them about 0.22 to 0.39. Nothing in the vocabulary size or the batch
configuration marks those four out: wordphn-BPE-200 shares both with its noisy siblings
and sits at 0.115, and **one tokenisation holds the whole range on its own**. The four
subwordphnx-BPE sizes measure 0.015, 0.279, 0.157 and 0.062, so the quietest unit in
the project and the noisiest differ only in vocabulary size, the floor is not monotone
in that size, and the largest vocabulary is back among the quiet units. So the right
lesson is not a revised constant but that **a new unit's floor has to be measured, not
inherited** -- the reason the four wordphn units were seeded at all, then
subwordphnx-BPE-2000 after them, and the reason subwordphnx-BPE-50000 was run at five
seeds from the start rather than one.

**Two runs told the truth at one end of the subwordphnx sweep and not at the other**,
which is the case against reading a floor off two of anything. subwordphnx-BPE at 200
and 5,000 came back 0.021 and 0.080 apart on their second runs, both far inside the
0.279 their sibling at vocabulary 2,000 measures over five, and that was read here as
weak evidence that the ends of the sweep are quieter than its middle. Both were then
taken to five runs on 2026-10-13. 5,000 measured **0.157**, nearly three times its
two-run spread, which cost two of the four comparisons its second run had bought; 200
measured **0.015**, within 5% of what its two runs showed, and is now the tightest unit
in the project. The ends are indeed the quiet ones, so the reading was right in
direction and wrong by a factor of three in magnitude at one of them -- and nothing
visible in the two-run numbers distinguished the case that held from the case that did
not. Borrowing 0.279 from vocabulary 2,000 remains the worse of the two assumptions:
both ends came in well below it.

**And five runs changed a verdict the canonical run had already given.**
subwordphnx-BPE-50000 was seeded from the start, so nothing had to be retracted, but
its seed-0 number sits 0.18 bits/s below wordphn and its five-seed mean sits 0.03
above: seed 0 is wordphn's *worst* run of five and tied-best for the new unit, and the
two units are indistinguishable once both are averaged. Written up from the canonical
run alone it would have read as the first unit in the project to beat wordphn from
below. This is the same failure as subwordphnx-BPE-2000's, caught at the other end --
not a borrowed floor but a measured one applied to a single draw -- and it is why the
verdicts in "Subwordphnx-BPE" are stated on means wherever a pair is close.

All 21 pairwise gaps among the original seven units against the 0.1 floor (the 34
further pairs involving a wordphn unit are under "Wordphn", and subwordphnx-BPE's are
under "Subwordphnx-BPE"):

| comparison | gap | sd_diff | ratio | verdict |
|---|---:|---:|---:|---|
| phone vs word | 2.078 | 0.069 | 29.94 | clear |
| phone vs wBPE-200 | 1.714 | 0.084 | 20.42 | clear |
| phone vs subword | 1.466 | 0.130 | 11.26 | clear |
| phone vs wBPE-1000 | 1.447 | 0.083 | 17.36 | clear |
| phone vs syllable | 1.152 | 0.092 | 12.53 | clear |
| word vs wBPE-5000 | 1.040 | 0.086 | 12.09 | clear |
| phone vs wBPE-5000 | 1.039 | 0.092 | 11.27 | clear |
| syllable vs word | 0.927 | 0.086 | 10.81 | clear |
| wBPE-200 vs wBPE-5000 | 0.675 | 0.098 | 6.89 | clear |
| word vs wBPE-1000 | 0.631 | 0.076 | 8.26 | clear |
| subword vs word | 0.612 | 0.126 | 4.87 | clear |
| syllable vs wBPE-200 | 0.562 | 0.098 | 5.75 | clear |
| subword vs wBPE-5000 | 0.427 | 0.140 | 3.06 | clear |
| wBPE-1000 vs wBPE-5000 | 0.408 | 0.098 | 4.19 | clear |
| word vs wBPE-200 | 0.364 | 0.077 | 4.72 | clear |
| subword vs syllable | 0.314 | 0.140 | **2.25** | **marginal** |
| syllable vs wBPE-1000 | 0.296 | 0.097 | 3.04 | clear |
| wBPE-1000 vs wBPE-200 | 0.267 | 0.090 | **2.97** | **marginal** |
| subword vs wBPE-200 | 0.248 | 0.134 | **1.85** | **noise** |
| syllable vs wBPE-5000 | 0.113 | 0.105 | **1.08** | **noise** |
| subword vs wBPE-1000 | 0.019 | 0.134 | **0.14** | **noise** |
**Sixteen of twenty-one are clear**, two marginal, three within noise, exactly as the
four-seed read had it and with the same five exceptions. Three of the ratios sit within
0.07 of the 3x bar, which is why this table carries two decimals.

**The linguistic ranking, over the four paper units**, is

> word < word-BPE-200 < word-BPE-1000 < {syllable, word-BPE-5000} < phone

with the braced pair **unordered** (1.1). Every other adjacent step is clear,
including syllable versus word-BPE-1000 (3.04, by a margin of 0.04) and word versus
word-BPE-200 (4.7).

**subword-BPE cannot be placed at all** among the word-BPE points: it is within
noise of both word-BPE-1000 (0.1) and word-BPE-200 (1.9), and only marginal
against syllable (2.3). It is the noisiest of the nine tight units (SD 0.118), and its
canonical seed-0 value of 28.20 is second-highest of five, against a mean of
28.096. Any statement placing subword-BPE relative to the word-BPE family is
therefore unsupported. It is clearly worse than word (4.9) and clearly better than
word-BPE-5000 (3.1) and phone (11.3).

**Note on the canonical numbers.** Seed 0 sits low in its unit's distribution for
word (27.46 against a mean of 27.484) and lowest of five for word-BPE-200 (27.76
against 27.848), and high for subword-BPE (28.20 against 28.096). The reported
word-versus-word-BPE-200 gap of 0.30 is accordingly a little *smaller* than the 0.36
the means give, so the canonical figure is the conservative one here. Seed 0 remains
the reported run throughout, so every other table and cross-reference is unchanged.

For the two noisy units seed 0 is not merely off-centre but **unrepresentative in
opposite directions**: lowest of five for wordphn-BPE-1000 (28.48 against 28.781)
and highest of five for wordphn-BPE-5000 (29.58 against 29.274). Its 1.10 bits/s
step between them is 0.49 on the means. That one coincidence is what produced the
"interaction reverses with depth" reading written up before the seeds existed, and
it is the clearest example in the project of why a single run is not a result.

**The augmentation fix tightened word and left syllable about where it was.** word's
SD fell from 0.085 over four pre-fix runs to 0.043 over five post-fix ones. syllable's
read 0.038 over four post-fix runs against 0.091 before, which looked like the same
effect, but the fifth run came in at 28.53, 0.12 above the previous maximum, and took
it to 0.074. The three word-BPE points went the other way, 0.048, 0.048 and 0.085
before against 0.064, 0.063 and 0.074 after. phone and subword-bpe have nothing to
spell out, so they were never retrained for the fix and have no before-figure. The
honest summary is that only word's change survived the fifth run, and with four degrees
of freedom even that is a direction rather than a measurement. The pre-fix figures were
computed from that generation's own per-utterance dumps, which were pruned on
2026-10-13, so they stand as reported here rather than being re-derivable.

**The noisiest unit in the project was written up from a single run.**
subwordphnx-BPE-2000's five seeds run 28.76 to 29.33, SD 0.279, which displaces
wordphn-BPE-1000 as the worst in the project. Its seed 0 of 28.82 is third of the
five against a mean of 28.982, so the canonical run flatters it by 0.17 bits/s. Final
validation loss orders exactly with bit rate across all five runs (4.100, 4.105, 4.111,
4.172, 4.183 against 28.76, 28.76, 28.82, 29.24, 29.33), so this is optimisation
variance reaching the metric, not an evaluation artefact -- and it looks bimodal rather
than Gaussian, three runs landing near 4.105 and two near 4.178. **The two neighbouring
sizes do the same thing, and the split survives even where the unit is quiet.** At
vocabulary 5,000 three runs land near 4.635 and two near 4.680; at 200, three near
2.6688 and two near 2.6717. All three order exactly with bit rate, so the ranking of
the five runs is the same whichever of the two quantities is read. Its scale does not
travel with it: vocabulary 200 spreads its five validation losses over 0.0032 against
5,000's 0.052, a factor of sixteen, which is why the same shape goes with SDs of 0.015
and 0.157.

**Vocabulary 50,000 then broke the pattern, which is what five runs are worth.** Three
sizes showing the same 3-2 split had read here as evidence that the shape belongs to
the family rather than to one size. The fourth does not reproduce it. Its validation
losses are 5.3088, 5.3173, 5.3188, 5.3321, 5.3425, so the gap between the would-be
groups (0.0133) is no wider than the spread inside the first one (0.0100) -- a
continuum, not two modes. And it is the one size whose validation loss does *not* order
with bit rate: seed 0 is best on bits (27.9764) and only third on loss, behind seeds 1
and 2. Both quantities separate the five runs by very little here, which is the same
fact as the unit's quiet 0.062, so a near-tie reordering is what should be expected
rather than a puzzle. Five runs cannot establish a shape, which is now the measured
conclusion rather than the cautious one.
What it costs is set out under "Subwordphnx-BPE", and it is not small: the two steps
inside the vocabulary sweep that pass through 2,000 stop clearing the floor. Those
subsections were written against a floor borrowed as 0.109 and 0.154, each of which
turns out to be about two fifths of the right one.

**A caution about short-run reads, learned repeatedly.** Units have looked far
tighter after three seeds than they proved to be after four, and syllable has now done
the same thing again at the fifth run, doubling from 0.038 to 0.074. Five runs give
four degrees of freedom and that is still not many; treat every SD above as an order of
magnitude rather than a measurement. subwordphnx-BPE-5000 is the cleanest case: 0.056
over two runs, 0.157 over five.

## Wordphn

A word-sized unit defined by its **realised phones** rather than its spelling, so
`abbey` becomes `AE-B-IY`, grouped from the `words` and `phones` tiers by their shared
time boundaries (`design.md`, "The wordphn unit"). Run on the neural stage at the
canonical setup of 20 epochs and `lr_max=1e-3`, seed 0, and on the n-gram stage over
orders 1 to 5 and four data fractions, both 2026-10-08. The full n-gram tables are
under "Wordphn (n-gram sweep)" below; the cross-estimator comparison is "Under
n-grams".

The unit is matched to word on everything except unit identity: one unit per word
interval, so dev-clean is the same 54,402 units over the same 15,845.69 s of speech,
with per-utterance durations bit-identical to word's. Any difference is therefore about
*what* the unit is, not about how many there are or how long they last.

### Bit rate

| unit | bits/s | ppl | bits/word | tokens |
|---|---:|---:|---:|---:|
| word | **27.46** | 147.27 | 7.998 | 60,412 |
| wordphn | 28.16 | 177.84 | 8.202 | 59,695 |

`ppl` is perplexity per *scored token* and `tokens` is how many of those there were.
The two differ, since the phone spell-out expands an out-of-vocabulary unit into fewer
pieces than the character spell-out does, so perplexity is not comparable across the
two rows. `bits/word` is, being total bits over the shared 54,402 word intervals:
wordphn costs **0.204 bits more per word**.

### Against every other unit

Paired speaker-level bootstrap, B = 10,000, over the same 2,703 dev-clean utterances,
from `llm/per_utterance/` as in "Confidence intervals":

| comparison | difference | 95% CI | significant |
|---|---:|---|---|
| wordphn - word | +0.70 | [0.62, 0.77] | yes |
| wordphn - word-BPE-200 | +0.40 | [0.32, 0.47] | yes |
| wordphn - word-BPE-1000 | +0.09 | [0.01, 0.17] | yes |
| wordphn - subword-BPE | -0.04 | [-0.17, 0.10] | **no** |
| wordphn - syllable | -0.25 | [-0.34, -0.17] | yes |
| wordphn - phone | -1.44 | [-1.61, -1.27] | yes |

Those intervals hold the checkpoint fixed. Seeds 1 to 3 were run on 2026-10-08/09 and
seed 4 on 2026-10-13, so the training axis is measured here rather than assumed.
Five-seed means, each gap against its own `sqrt(sa^2 + sb^2)` floor from "Seed
variance":

| comparison | gap (means) | floor | ratio | verdict |
|---|---:|---:|---:|---|
| wordphn - word | +0.583 | 0.085 | 6.9 | clear |
| wordphn - word-BPE-200 | +0.219 | 0.097 | 2.3 | **marginal** |
| wordphn - word-BPE-1000 | -0.048 | 0.096 | 0.5 | **noise** |
| wordphn - subword-BPE | -0.030 | 0.139 | 0.2 | **noise** |
| wordphn - syllable | -0.344 | 0.104 | 3.3 | clear |
| wordphn - word-BPE-5000 | -0.457 | 0.104 | 4.4 | clear |
| wordphn - phone | -1.495 | 0.091 | 16.4 | clear |

**Spelling beats realisation**, and that survives the seeds: +0.583 on the means at
6.9x the floor, against the +0.70 seed 0 reports. Two verdicts move from the single-run
read, in opposite directions. wordphn against word-BPE-200 was called clear at 4x and
is **marginal** (2.3x), the seed-0 gap of 0.40 being 0.22 on the means. wordphn against
syllable was called a lean at 2.5x and is **clear** at 3.3x, with wordphn the better
unit. wordphn stays unplaced against word-BPE-1000 (0.5x) and indistinguishable from
subword-BPE (a gap of 0.030 bits/s), so the "unplaced among {word-BPE-1000,
subword-BPE}" reading stands, as does the ranking over the four paper units, which
wordphn is not one of.

### Why it loses

Two effects pull in opposite directions and the measurement is their sum.

- **Homophones collapse**, which should help. Over 960h this cuts the vocabulary from
  89,048 orthographic types to 84,163 realisations, and at min-count 2 from 57,977 to
  56,255; on dev-clean 203 realisations cover more than one word type, `T-UW` for
  to/too/two, `ER` for are/or, `DH-IY` for the/thee.
- **Pronunciation variants split**, which should hurt. 248 dev-clean word types have
  more than one realisation, 3,674 over 960h: `the` is `DH-AH` 2,929 times and `DH-IY`
  519, `and` is `AE-N-D` 1,326 and `AH-N-D` 589. The model is charged for predicting
  *which* variant the speaker produced, a cost word never pays.

The second dominates, and that is the honest reading of the penalty: not evidence
that lexical identity is better carried by spelling, but that **this unit asks the
model to transmit the pronunciation as well as the word**, and the extra pronunciation
entropy outweighs what homophone merging saves. Separating the two would need a variant
that collapses homophones *without* splitting variants, mapping each word to one
canonical realisation; that was not run.

The deficit is not an out-of-vocabulary artefact. Spell-out fires on 441 of the 54,402
dev-clean units, *fewer* than word's 486, despite the smaller vocabulary, because
collapsing homophones lifts rare words over the min-count threshold.

### Under n-grams

The sweep was added after the neural result, to ask whether the 0.70 bits/s is about
the unit or about OPT-20M. It is about the unit, but it is smaller, and it grows with
the strength of the estimator:

| cell | word | wordphn | penalty |
|---|---:|---:|---:|
| n-gram, order 3, 25% | 32.30 | 32.66 | +0.36 |
| n-gram, order 3, 50% | 31.39 | 31.79 | +0.40 |
| n-gram, order 3, 75% | 30.96 | 31.38 | +0.42 |
| n-gram, order 3, 100% | 30.62 | 31.05 | +0.43 |
| n-gram, order 5, 100% | 30.91 | 31.34 | +0.43 |
| OPT-20M, seed 0 | 27.46 | 28.16 | +0.70 |
| OPT-20M, 4-seed mean | 27.49 | 28.06 | +0.57 |

Perplexity is comparable between the two n-gram sweeps, for once, because both score
the same 54,402 dev-clean units and the n-gram stage's spell-out adds no tokens: 484.11
for word against 528.22 for wordphn, so **+0.126 bits per word interval** against the
neural stage's +0.204. Both units bottom out at order 3 and their top-order discounts
agree to 0.01 (0.84 against 0.85), so this is not a smoothing artefact. The n-gram
stage also applies no minimum-count threshold, so the comparison here runs over the
full type inventories, 84,163 realisations against 89,048 words, rather than the
thresholded vocabularies the neural stage uses; the penalty has the same sign and is
smaller either way.

Paired speaker-level bootstrap on the order-3 100% dumps, B = 10,000, gives
**+0.432 [0.350, 0.507]**. An n-gram fit is deterministic given its training data, so
unlike the neural rows there is no seed floor to clear and the interval is the whole
error budget (`design.md`, "Uncertainty").

The rest of the family's n-gram intervals, all on that one resample. The marginals are
wide because dev-clean's 40 speakers are the resampling unit; the paired columns are
the ones to read.

| unit | bits/s | 95% CI | vs phone | vs syll |
|---|---:|---:|---:|---:|
| wordphn | 31.05 | [30.15, 31.95] | -12.59 | -0.62 |
| wphn-BPE-200 | 30.97 | [30.08, 31.86] | -12.67 | -0.70 |
| wphn-BPE-1000 | 31.05 | [30.15, 31.93] | -12.60 | -0.63 |
| wphn-BPE-5000 | 31.32 | [30.43, 32.20] | -12.33 | -0.36 |

Every paired difference there is clear. The syllable intervals run from
[-0.808, -0.589] at 200 merges to [-0.465, -0.248] at 5000, so the whole realisation
family beats syllable under n-grams, by 0.36 to 0.70 bits/s, where under OPT-20M only
wordphn itself manages it (see "Against every other unit"). Against phone the intervals
span [-13.16, -11.87] and the question is not close.

Swapping the n-gram for OPT-20M buys the four units 2.895 [2.697, 3.102],
2.622 [2.401, 2.850], 2.562 [2.344, 2.786] and 1.738 [1.540, 1.951] bits/s, shrinking
with merge depth exactly as the word family's own cross-estimator gaps do (3.16 down to
2.31 bits/s in "Confidence intervals").

The monotone trend in that table, +0.36 to +0.43 as the n-gram training data grows and
then +0.57 under a neural LM, has an obvious reading: the stronger the estimator, the
more of the word-level structure the two units share it already captures, so the
irreducible pronunciation-variant entropy is a larger share of whatever is left. Four
fractions of one estimator and one checkpoint of another are thin evidence for a
mechanism, so take the direction and not the explanation.

### With collocation merging (wordphn-BPE)

The collocation variant of the unit, standing to wordphn as word-BPE stands to word:
k merges of the most frequent adjacent pair, fitted on the wordphn stream itself rather
than inherited from word-BPE (`design.md`, "The wordphn-BPE unit"). Run at the same
three merge counts as word-BPE, 200/1000/5000, canonical setup, 2026-10-08, with
seeds 1 to 3 the following day, seed 4 on 2026-10-13 and the n-gram sweeps alongside
(see "Merging under n-grams" below). Reported bit rates are seed 0, as everywhere in
this file. The first merges are the expected function-word collocations,
`AH-V DH-AH` ("of the"), `IH-N DH-AH`, `T-AH DH-AH`, `T-AH B-IY`.

The vocabulary is **essentially nested**. At 200 and 1000 merges the singletons are a
bit-identical set to wordphn's own min-count-2 vocabulary of 56,255, so those two
comparisons vary the merged units and nothing else; at 5000 merges two singletons slip
below the threshold once merging absorbs their occurrences, as four do for
word-BPE-5000.

| unit | bits/s | ppl | bits/word | tokens |
|---|---:|---:|---:|---:|
| wordphn | 28.16 | 177.84 | 8.202 | 59,695 |
| wordphn-BPE-200 | 28.35 | 297.85 | 8.258 | 54,661 |
| wordphn-BPE-1000 | 28.48 | 448.66 | 8.297 | 51,236 |
| wordphn-BPE-5000 | 29.58 | 952.70 | 8.616 | 47,364 |

**The series is monotone, as word-BPE's is, but its shape is a seed artefact.**
wordphn-BPE-1000 and wordphn-BPE-5000 both have an SD of 0.235, and seed 0 is the
lowest of five for the first and the highest of five for the second, which inflates the
step between them. On five-seed means the series reads 28.066, 28.268, 28.781, 29.274
and the steps are, each ratio against its own two-checkpoint floor:

| step | wordphn-BPE | ratio | word-BPE | ratio |
|---|---:|---:|---:|---:|
| none to 200 | +0.202 | 1.48 | +0.364 | 4.72 |
| 200 to 1000 | +0.513 | 1.96 | +0.267 | 2.97 |
| 1000 to 5000 | +0.493 | 1.48 | +0.408 | 4.19 |
| none to 5000 | +1.207 | 4.90 | +1.040 | 12.09 |
**No single wordphn-BPE step is established**, and the 1000-to-5000 step that seed 0
puts at +1.095 is +0.493 on the means, indistinguishable from word-BPE's +0.408 at the
same depth. What survives is the span: merging 5000 collocations into the realisation
unit costs +1.207 bits/s at 4.9x the floor, and that is clear. On the word side only
the 200-to-1000 step fails to clear 3x, and it fails it by 0.03.

**There is no interaction with depth.** Merging cost measured against the unmerged
unit. Each row spans four independently trained units, so its floor is
`sqrt` of the four SDs summed in quadrature:

| merges | seed 0 | 5-seed mean | floor | ratio | verdict |
|---|---:|---:|---:|---:|---|
| 200 | -0.111 | -0.162 | 0.157 | 1.0 | noise |
| 1000 | -0.278 | +0.083 | 0.258 | 0.3 | noise |
| 5000 | +0.324 | +0.168 | 0.261 | 0.6 | noise |
| reversal, 1000 to 5000 | +0.602 | +0.084 | 0.346 | 0.2 | noise |

The seed-0 bootstrap intervals exclude zero on all four rows, and **none of it
survives the training axis.** The last row is the one that matters: the word and
wordphn baselines cancel, leaving the four-checkpoint contrast `wordphn-BPE-5000 -
word-BPE-5000 - wordphn-BPE-1000 + word-BPE-1000`, which seed 0 puts at +0.602 against
a measured floor of 0.346, and which the five-seed means put at +0.084, a ratio of
**0.2x**. The reversal was one unlucky pair of runs.

Stated as the realisation cost at fixed segmentation, the same cells read +0.583
unmerged, +0.421 at 200 merges, +0.666 at 1000 and +0.751 at 5000 on the means, against
seed 0's +0.699, +0.588, +0.421 and +1.023. Neither ordering is monotone, the two
disagree about where the minimum is, and every pairwise difference between them is
inside the floor. **The honest statement is that the realisation penalty does not
measurably depend on merge depth.**

The n-gram sweeps say the same thing independently, and say it much more precisely,
since an n-gram fit has no training noise at all: the interaction is +0.02 to +0.03 at
every depth and the reversal contrast is -0.008 [-0.031, +0.016]. See "Merging under
n-grams" below.

**Where the three points sit.** On five-seed means, against each pair's own floor.
wordphn-BPE-200, at 28.268, is unplaced against word-BPE-1000 (1.2x), syllable (1.0x),
subword-BPE (1.0x) and word-BPE-5000 (1.9x), and clear of word
(6.4x), word-BPE-200 (3.2x) and phone (10.2x). wordphn-BPE-1000, at 28.781, is unplaced
against word-BPE-5000 (1.0x) and syllable (1.5x) and clear of word (5.4x) and
word-BPE-200 (3.8x). wordphn-BPE-5000, at 29.274, is clear of every unit outside its
own family except phone.

The two deep points are so noisy that they are **not even ordered against each other**
(+0.493, 1.5x), and wordphn-BPE-1000 is only marginal against word-BPE-1000 (2.7x) and
subword-BPE (2.6x), though the fifth seed moves it clear of phone (3.2x). The one
placement that survives intact from the single-run read is the striking one:
**wordphn-BPE-5000 is indistinguishable from
phone** (-0.288, 1.2x), the costliest unit in the project, so merging 5000 collocations
into the realisation unit costs about as much as discarding the word altogether. None
of this changes the ranking over the four paper units, which no wordphn variant is part
of.

### Merging under n-grams

The n-gram sweeps were run at the same three merge counts, orders 1 to 5 over four
data fractions. They are the independent check on everything above, and they were in
hand before the seeds were. The order-3 100% cells are

| merges | word | wordphn | penalty |
|---|---:|---:|---:|
| none | 30.62 | 31.05 | +0.43 |
| 200 | 30.52 | 30.97 | +0.45 |
| 1000 | 30.59 | 31.05 | +0.46 |
| 5000 | 30.87 | 31.32 | +0.45 |

The realisation penalty is flat to within 0.03 bits/s across a factor of 25 in merge
depth, where OPT-20M gives +0.699, +0.588, +0.421 and +1.023 at seed 0 and +0.572,
+0.421, +0.666 and +0.751 on five-seed means, two orderings that disagree with each
other. Equivalently the two merge curves are parallel: step by step word reads -0.10,
+0.07 and +0.28, wordphn -0.08, +0.08 and +0.27. Order 5 says the same thing, +0.43,
+0.45, +0.45 and +0.44.

Paired speaker-level bootstrap on the eight order-3 100% dumps, B = 10,000, all on one
resample, so every row below is exactly comparable:

| contrast | difference | 95% CI | significant |
|---|---:|---|---|
| penalty, unmerged | +0.432 | [0.350, 0.507] | yes |
| penalty, 200 merges | +0.453 | [0.376, 0.523] | yes |
| penalty, 1000 merges | +0.459 | [0.377, 0.536] | yes |
| penalty, 5000 merges | +0.451 | [0.370, 0.529] | yes |
| interaction, 200 merges | +0.021 | [0.006, 0.036] | yes |
| interaction, 1000 merges | +0.028 | [0.006, 0.049] | yes |
| interaction, 5000 merges | +0.020 | [-0.006, 0.046] | **no** |
| reversal, 1000 to 5000 | -0.008 | [-0.031, 0.016] | **no** |

The four penalties are mutually indistinguishable. The interaction is a constant
+0.02 to +0.03 that two of three intervals do separate from zero, which is credible
precisely because an n-gram fit is deterministic and the interval is the whole error
budget, but it is 0.07% of the unit's bit rate and does not move with depth. The
reversal contrast, which seed 0 of the neural stage puts at +0.602 [0.519, 0.689], is
**zero here**.

**This is what settled the question**, and it settled it before the seeds did. These
sweeps ran the day the seed-0 reversal was written up, and they disagreed with it at a
precision the neural stage cannot reach, over a factor of 25 in merge depth, with no
training noise to argue about. The seeds then found the neural contrast to be
+0.084 against a floor of 0.346 and agreed. Two independent estimators, one of them
exact, now say there is **no depth-dependent interaction between collocation merging
and unit identity**. Under n-grams merging charges the two families the same amount at
every depth, and the 0.6 bits/s the earlier write-up could not explain turns out never
to have needed explaining.

The same run reproduces the word family's own published n-gram comparisons, which is
the check that this is the established machinery rather than a new path: word-BPE-200
beats word by 0.102 [0.060, 0.142] and word-BPE-1000's 0.034 lead does not clear zero
([-0.088, +0.020]), exactly as "Confidence intervals" has it.

The family's own merge curve, every step of it on that same resample:

| contrast | difference | 95% CI | significant |
|---|---:|---|---|
| 200 vs unmerged | -0.081 | [-0.122, -0.038] | yes |
| 1000 vs unmerged | -0.007 | [-0.061, 0.052] | **no** |
| 5000 vs unmerged | +0.264 | [0.186, 0.341] | yes |
| 1000 vs 200 | +0.075 | [0.045, 0.104] | yes |
| 5000 vs 200 | +0.345 | [0.286, 0.402] | yes |
| 5000 vs 1000 | +0.270 | [0.226, 0.315] | yes |

The curve is a shallow dip and then a climb: 200 merges buy 0.08 bits/s, the gain is
spent by 1000, and 5000 costs 0.26 against the unmerged unit and 0.35 against the best
point. Five of the six steps are clear, the exception being the one that straddles the
minimum. Word's curve has the same shape, -0.102, -0.034 and +0.244 against its own
unmerged point, which is the merge-depth half of the no-interaction finding restated
without reference to the word family at all.

## Subwordphnx-BPE

Standard subword BPE over **phones** instead of characters: the same HuggingFace
`BpeTrainer` as subword-BPE with only the alphabet swapped, 39 phones for 27 letters,
and merges confined to a word so that no piece crosses a word boundary
(`design.md`, "The subwordphnx-BPE unit"). That confinement makes it the faithful
phonetic analogue of subword-BPE, which is word-confined too by virtue of its
`WhitespaceSplit` pre-tokeniser. Run at vocabulary 200, 2,000, 5,000 and 50,000;
canonical setup, 2026-10-09 for the first three and 2026-10-14 for the fourth, with
five seeds at every size.

It completes the realisation ladder. Each side of the project now has a sub-word unit,
a word-sized unit and a collocation unit: subword-BPE, word, word-BPE-k by spelling,
and subwordphnx-BPE, wordphn, wordphn-BPE-k by realisation, with phone underneath both.

### Bit rate

At vocabulary 2,000: **28.82 bits/s at seed 0**, perplexity 59.87 over 77,344 dev-clean
tokens; 28.982 on the five-seed mean, SD **0.279**, which is the worst seed floor in
the project. It lands between wordphn and phone.

| vs | difference | 95% CI | x floor |
|---|---:|---|---:|
| phone | -0.783 | [-0.884, -0.691] | **2.8** |
| wordphn | +0.658 | [0.540, 0.766] | **2.3** |
| subword-BPE | +0.621 | [0.537, 0.699] | **2.0** |
| wordphn-BPE-200 | +0.466 | [0.353, 0.573] | **1.5** |
| syllable | +0.405 | [0.345, 0.461] | **1.4** |
| word | +1.357 | [1.222, 1.488] | 4.8 |

Paired speaker-level bootstrap, B = 10,000, every unit on one resample; the `x floor`
column divides each seed-0 difference by that pair's own floor from "Seed variance".

**Only the gap to word survives the measured floor at seed 0.** Every interval above
excludes zero, but the floor is 0.279, so the gap to phone is marginal at 2.8x and the
three comparisons that would place this unit inside the middle of the ranking --
against wordphn, wordphn-BPE-200 and syllable -- do not clear it. Seed 0 is not
flattering the unit here. On five-seed means the gap to phone *widens* to 0.580 and
2.0x, and the gap to wordphn widens to 0.915, which is 3.2x and the one comparison in
this table that the means and the canonical run disagree about: seed 0 happens to be
below this unit's mean and above wordphn's. So the unit's position between wordphn and
phone is where the point estimates put it, it is **clearly worse than wordphn on the
means and not at seed 0**, and it is unseparated from phone either way: wordphn 28.16,
subwordphnx-BPE-2000 28.82 (28.98 on means), phone 29.60.

**The parallel with the spelling side is the most useful thing here.** Sub-word
chunking costs almost exactly the same relative to its own word-level unit on both
sides:

| chunking | against its word-level unit | 95% CI |
|---|---:|---|
| subword-BPE vs word | +0.736 | [0.597, 0.876] |
| subwordphnx-BPE-2000 vs wordphn | +0.658 | [0.540, 0.766] |

The intervals overlap. Chunking a word into sub-word pieces costs about 0.7 bits/s
whether the pieces are letters or phones, which is a tidier result than either unit
alone. Two caveats the seeds add. The realisation-side figure is 2.3x its floor, so the
agreement is between two point estimates of which only the spelling-side one is
individually established. And on five-seed means the two read +0.612 and +0.915, which
is a factor of 1.5 rather than a match, so how close the agreement looks depends on
reading seed 0. The parallel is suggestive rather than demonstrated.

**The segmentation smuggles word boundaries in, so the comparison with phone is not a
clean one.** This has to be stated carefully, because it is the one caveat that
affects every number above. The piece sequence depended on where the words divided, so
it carries that information back: `DH-AH K-AE-T` ("the cat") and a hypothetical
one-word `DH-AH-K AE-T` tokenise differently from the same phones. That is why this
unit beats **phone** on content bits per phone, 2.289 against 2.343, while describing
the identical phone string: it is handed information phone does not have, for free, and
is charged nothing for it. So the -0.78 against phone is **not** a statement that
chunked phones model better than bare phones. The honest version is that word-confined
chunking transmits more than the phone stream does, and the bit rate reflects that.
Three quarters of word tokens (75.8%) come out as a single piece at this vocabulary, so
the unit is closer to a 2,000-entry lexicon with a phone-piece fallback than to a
sub-word model. The seeds add a second reason not to lean on the comparison: it does
not clear the measured floor either way, so it is both confounded and unestablished.

### Vocabulary size

2,000 was inherited from subword-BPE, where it is a sensible size for an alphabet of
26 letters; there is no reason it should also be right for one of 39 phones. The
confined unit was therefore rerun at 200 and 5,000, which brackets it and spans the
interesting range: at 200 the longest pieces are morphemes (`EY-SH-AH-N`, "-ation";
`AH-DH-ER`, "-other"), at 5,000 they are whole polysyllabic words
(`IH-K-S-T-R-AO-R-D-AH-N-EH-R-IY`, "extraordinary"). 5,000 left the curve still
falling, so 50,000 was added on 2026-10-14 to find where it ends; there its longest
pieces run to seventeen phones (`K-AA-N-T-R-AE-D-IH-S-T-IH-NG-G-W-IH-SH-T`,
"contradistinguished"). Canonical setup, 2026-10-09 for the first three sizes and
2026-10-14 for the fourth; five runs at each of the four, the extra ones on
2026-10-12/13 and 2026-10-14.
Merges are the vocabulary minus the 39 phones and 3 special tokens.

| unit | merges | dev pieces | bits/s | perplexity |
|---|---:|---:|---:|---:|
| phone | n/a | 194,827 | 29.60 | 5.19 |
| subwordphnx-BPE-200 | 158 | 117,563 | 29.04 | 14.19 |
| subwordphnx-BPE-2000 | 1,958 | 74,641 | 28.82 | 59.87 |
| subwordphnx-BPE-5000 | 4,958 | 65,155 | 28.52 | 101.07 |
| subwordphnx-BPE-50000 | 49,958 | 55,214 | 27.98 | 201.43 |
| wordphn | n/a | 56,992 | 28.16 | 177.84 |

Bit rates are seed 0 throughout, as everywhere else in this document. On means the four
sizes read 29.065, 28.982, 28.670 and 28.034, so the first step of the sweep all but
vanishes while the later two grow; what that does to them is in the bootstrap
discussion below.

Dev pieces excludes the per-utterance boundary token, so it is directly comparable to
dev-clean's 194,827 phones and 54,402 words; the perplexities are per piece and not
comparable across rows at all (`design.md`, "Scope and metrics"); the boundary column
below is bits for the one end-of-utterance token. Training tokens and step budgets are
in `design.md`, "Neural training setup", and in the per-unit hyperparameters above.

**The bit rate falls with vocabulary, monotonically over four sizes, and the sweep runs
out the far side of wordphn rather than stopping at it.** Each of the three outer steps
clears this family's floors; the two that pass through vocabulary 2,000 clear nothing,
that unit's own seed spread being wider than the whole descent. See the bootstrap table
below, which is where that gets measured. The two anchors were read here as the sweep's
endpoints -- one piece per phone, and one piece per word -- with the confined unit the
one-parameter family between them. **That framing survives the bottom anchor and not
the top one.** Neither is exact: phone emits a one-phone word such as "a" or "I" as a
single piece (3.6% of word tokens), and wordphn spells an out-of-vocabulary word out
into several, which is why its stream is 56,992 pieces rather than 54,402. The second
of those is the whole story at vocabulary 50,000.

| unit | phones/piece | pieces/word | one-piece words |
|---|---:|---:|---:|
| phone | 1.000 | 3.581 | 3.6% |
| subwordphnx-BPE-200 | 1.657 | 2.161 | 47.3% |
| subwordphnx-BPE-2000 | 2.610 | 1.372 | 75.8% |
| subwordphnx-BPE-5000 | 2.990 | 1.198 | 85.8% |
| subwordphnx-BPE-50000 | 3.529 | 1.015 | 98.8% |
| wordphn | 3.418 | 1.048 | 99.2% |

**At 50,000 the unit overtakes wordphn structurally.** Its pieces are longer than
wordphn's units (3.529 phones against 3.418) and its stream is shorter (55,214 against
56,992), even though slightly fewer of its word tokens come out whole. The two facts
are the same fact: wordphn has one unit per word and spells an unseen realisation out
phone by phone, eight or nine tokens for one word, while this unit splits a rare
realisation into two pieces. Of dev-clean's 54,402 words, 53,762 are a single piece
here, 493 take two, 124 take three and 23 take more. So at this vocabulary the unit is
not an interpolation towards wordphn at all -- it *is* wordphn, with a subword backoff
in place of a phone spell-out. Any comparison between the two at this size is a
comparison between backoffs, and the sweep's interpolation reading applies to the three
smaller sizes.

Paired speaker-level bootstrap over all six, B = 10,000, same resample:

Rows name the four sizes by their vocabulary alone, so `200` is subwordphnx-BPE-200.

| comparison | difference | 95% CI | x floor |
|---|---:|---|---:|
| 200 vs phone | -0.556 | [-0.619, -0.493] | 9.8 |
| 2000 vs 200 | -0.227 | [-0.304, -0.155] | 0.8 |
| 5000 vs 2000 | -0.299 | [-0.357, -0.239] | 0.9 |
| 50000 vs 5000 | -0.541 | [-0.639, -0.443] | 3.2 |
| 5000 vs 200 | -0.525 | [-0.619, -0.435] | 3.3 |
| 50000 vs 200 | -1.067 | [-1.205, -0.923] | 16.8 |
| 5000 vs phone | -1.081 | [-1.201, -0.967] | 6.5 |
| 50000 vs phone | -1.623 | [-1.780, -1.469] | 19.8 |
| wordphn vs 5000 | -0.359 | [-0.461, -0.253] | 2.1 |
| 50000 vs wordphn | -0.182 | [-0.244, -0.119] | 1.9 |

Six of the ten rows clear 3x at seed 0; wordphn against 5,000 and the new unit against
wordphn are both marginal, and the two that pass through vocabulary 2,000 are within
noise.

**Every size is separated from phone; the descent through the middle is not settled;
and the overtake of wordphn is a property of seed 0 alone.** Every interval excludes
zero, and the ordering phone > 200 > 2,000 > 5,000 > 50,000 holds in 100% of the 10,000
speaker-level resamples. But a bootstrap interval only covers which utterances were
drawn, so the yardstick is the seed floor, and all four sizes carry a measured one of
their own over five runs: **0.015** at 200, **0.279** at 2,000, **0.157** at 5,000 and
**0.062** at 50,000, which is the widest spread of floors within any one tokenisation in
the project. Against those, 200 beats phone at 9.8x, 5,000 at 6.5x and 50,000 at 19.8x;
the step from 5,000 to 50,000 is 3.2x at seed 0 and 3.8x on the means, so it holds
either way; the end-to-end step from 200 is 16.8x. The two steps through vocabulary
2,000 sit at 0.8x and 0.9x, as before.

**The exception is the comparison against wordphn, and it inverts.** At seed 0 the new
size is 0.182 below wordphn, 1.9x the pair's 0.095 floor, which is marginal. On
five-seed means it is 0.033 *above* it, 0.3x, which is nothing at all: 28.034 against
28.066. Seed 0 is wordphn's worst run of five and tied-best for this unit, so the two
ends of a 0.18 gap are both drawn from the wrong tail of their own spreads. **The honest
statement is that the neural sweep arrives at wordphn and does not pass it.** The point
estimates descend through it, the measured floor does not let that be claimed, and this
is the one place in the subsection where reading the canonical run alone would have
produced a result in the opposite direction -- a phone-built unit beating its
word-level counterpart from below. See "Under n-grams", where a deterministic estimator
*does* separate them, and in the same direction.

That reading has moved three times, which is worth recording. Until 2026-10-12 both
endpoints were single runs read against 2,000's 0.289 borrowed, which put every row
under 2x and left the sweep with one surviving comparison. A second run at each end gave
spreads of 0.021 and 0.080 and moved four rows to clear. Three further runs at each end
then settled it: 5,000 measured **0.157**, nearly three times its own two-run spread,
and took two of those four rows back, so what the two-run read gave as 9.0x and 13.8x
are 3.3x and 6.5x; 200 measured **0.015**, which is what its two runs had shown, so the
rows that rest on it stand. Adding 50,000 then extended the sweep by a step that holds
on both readings while dissolving the one claim its own canonical run supported. The
only comparison here that has never moved under seeding is the one against phone, which
is the subsection's firmest finding and now holds at all four sizes, every one of them
measured against five runs at either side.

What remains unmeasurable is the shape of the middle. subwordphnx-BPE-2000's five seeds
run 28.76 to 29.33, and that range still **covers the 29.05-to-28.56 span of the three
smaller sizes**. Seed 0's 28.82 is third of its five runs; had seed 2 or seed 3 been the
canonical run, those three sizes would read 29.04, 29.24, 28.52 and would not be
monotone at all. So the monotonicity of the *reported* numbers is, between 200 and
5,000, a property of one draw at the middle size. Settling the inner steps needs more
seeds at vocabulary 2,000 rather than at the ends, all of which are now at five. The
outer step to 50,000 is not affected: 0.637 on the means against a 0.168 floor is clear
of 2,000's spread as well.

**The direction, though, no longer rests on these runs.** All four vocabularies were
swept under n-grams ("Under n-grams"), where a fit is deterministic and there is no
seed floor to clear, and the descent is monotone there with every step excluding zero:
-4.065 from 200 to 2,000, -0.543 to 5,000 and -0.296 to 50,000. So a second estimator,
with a different error budget entirely, puts the sizes in the same order. That settles
the sign of each step and nothing more: the n-gram descent is nine times the neural one
and front-loaded where the neural one is not, so it cannot stand in for the magnitudes
here, and the seeds are still what would establish them.

**Two confounds, and they both grow along the sweep.** Neither is a reason to doubt the
overall descent from phone, the one comparison the floor allows at every size, but both
mean it cannot be read as the effect of unit size alone.

The first is the one from "Bit rate" above, and it is the serious one. A confined
segmentation transmits where the words divide, and it transmits *more* of that as the
vocabulary grows: at 200 merges 47.3% of word tokens come out as a single piece, at
5,000 it is 85.8% and at 50,000 **98.8%**. The piece boundaries increasingly just are
the word boundaries, so the unit is drifting towards wordphn not only in granularity
but in how much free side information it is handed. That the curve reaches wordphn at
the top end is therefore close to a tautology: at 98.8% it has been handed almost
everything wordphn has. Splitting the bits accordingly:

| unit | content bits/phone | content bits/s | boundary bits |
|---|---:|---:|---:|
| phone | 2.3429 | 28.806 | 4.648 |
| subwordphnx-BPE-200 | 2.3084 | 28.382 | 3.875 |
| subwordphnx-BPE-2000 | 2.2893 | 28.147 | 3.922 |
| subwordphnx-BPE-5000 | 2.2666 | 27.868 | 3.809 |
| subwordphnx-BPE-50000 | 2.2279 | 27.393 | 3.420 |
| wordphn | 2.2414 | 27.559 | 3.514 |

Content bits per phone fall at every step, which is the whole effect: of the -1.623
from phone to 50,000, -1.413 is content and -0.209 is the cheaper end-of-utterance
token, and the same split reproduces each neighbouring step to within 0.001 bits/s.
Describing the identical 194,827 phones gets steadily cheaper as more of the word
structure is given away for free. The seed-0 gap to wordphn divides the same way, -0.166
content and -0.016 boundary, so what crosses wordphn at that one seed is the content
term and not a cheaper boundary token: 2.2279 bits per phone against 2.2414.

The second is that **the step budget is not constant across the sweep**: 5,000 steps at
vocabulary 200 against 2,400 at 50,000, because the project fixes 20 epochs over the
unit's own stream and a smaller vocabulary leaves a longer stream (`design.md`,
"Neural training setup"). That is the standing protocol and the same confound applies
to every other BPE sweep here, so it was followed rather than departed from; but it
does mean optimisation steps vary inversely with vocabulary, and they vary in the
direction that would flatter the small vocabularies, which are the ones that lose.
50,000 is also the one size that trains at a different `batch_size`, 2 rather than 10,
since the softmax no longer fits otherwise; the effective batch is 40 either way, so
what differs is the memory profile and not the optimisation.

### Under n-grams

All four vocabularies were swept: 2,000 on 2026-10-09 to ask whether the estimator
changes the verdict on the unit, then 200 and 5,000 on 2026-10-12 to ask the same of its
vocabulary sweep, which the seeds had left unsettled in the neural stage, and 50,000 on
2026-10-14 alongside the neural run at that size. The merge
counts are the neural stage's own (158, 1,958, 4,958 and 49,958), re-fit per fraction as
every BPE sweep here is, and the dev-clean streams come out at 74,644 pieces against the
neural unit's 74,641 at vocabulary 2,000, 65,150 against 65,155 at 5,000 and 55,223
against 55,214 at 50,000, so the two
stages really are scoring the same unit at each size. Each order-3 100% refit reproduces
its sweep cell to four decimals (35.6564, 31.5911, 31.0482, 30.7518), which is the check
that the per-utterance dumps describe the tables' quantity.

**The vocabulary sweep is monotone here too, and every step of it is clear.** This is
what the three extra sweeps were run for, and it is the one place where the n-gram stage
settles something the neural stage could not.

| step | n-gram, order 3 | 95% CI | OPT-20M |
|---|---:|---|---:|
| 2000 vs 200 | -4.065 | [-4.26, -3.87] | -0.227 |
| 5000 vs 2000 | -0.543 | [-0.60, -0.49] | -0.299 |
| 50000 vs 5000 | -0.296 | [-0.34, -0.25] | -0.541 |
| 5000 vs 200 | -4.608 | [-4.83, -4.39] | -0.525 |
| 50000 vs 200 | -4.905 | [-5.13, -4.68] | -1.067 |

An n-gram fit is deterministic given its training data, so these intervals are the whole
error budget and there is no floor to clear: the descent with vocabulary is real. On the
neural side the two steps through vocabulary 2,000 are both smaller than the 0.279 seed
SD measured there, so they stay unestablished; the other two clear their own floors,
200 to 5,000 at 3.3x at seed 0 though 2.5x on the means, and 5,000 to 50,000 at 3.2x and
3.8x ("Vocabulary size"). So the
*direction* has independent support from an estimator with no training noise at all,
which is what was missing, and both neural steps that can be tested agree.

**What it does not do is corroborate the neural sweep's magnitude, and the mismatch
reverses along the way.** End to end the n-gram descent is 4.91 bits/s against the
neural stage's 1.07, between four and five times larger, and it is front-loaded: 200 to
2,000 is worth -4.07 and the two later steps only -0.54 and -0.30, where the neural
steps are -0.23, -0.30 and -0.54. So the ratio between the estimators falls from
eighteenfold on the first step to **about one half on the last**: by vocabulary 50,000
the neural descent is nearly twice the n-gram one. The n-gram estimator takes its gains
as soon as chunking buys it any context at all and then has little left to take, while
the neural one keeps gaining as the pieces approach whole words. The two therefore agree
on the sign of every step and on nothing else about them, and reading the neural sweep
as a scaled-down copy of this one would be wrong in both magnitude and shape; what
transfers is the ordering.

**At vocabulary 5,000 the unit is no longer distinguishable from wordphn, under
n-grams, and at 50,000 it is clearly below it.** The paired difference is -0.005
[-0.118, +0.098] at 5,000, against +0.537 [0.43, 0.64] at vocabulary 2,000 and
+4.603 [4.37, 4.83] at 200, and then **-0.302 [-0.384, -0.225]** at 50,000. That is the
interpolation claim of "Vocabulary size"
measured directly rather than inferred, and then measured past its endpoint: a unit
built from 39 phones upwards arrives, at
4,958 merges, at a bit rate indistinguishable from the unit built from a 56,255-word
lexicon downwards, and at 49,958 merges it beats it. The mechanism is visible in the
segmentation -- 85.8% of dev-clean
word tokens are a single piece at 5,000 and 98.8% at 50,000, where the unit's whole
stream is shorter than wordphn's -- and it is also the clearest
statement of the boundary confound: what converges on wordphn is a unit that has been
handed where the words divide.

**Under n-grams the unit also catches `word`, which nothing else in the project does.**
At vocabulary 50,000 the difference is **+0.130 [-0.015, +0.263]**, the only comparison
anywhere in this section that fails to exclude zero, against +0.969 [0.81, 1.12] at
2,000. Plain wordphn is +0.432 worse than word under this estimator, so the subword
backoff recovers essentially all of wordphn's penalty for identifying a word by its
realisation. That puts a qualification under "Spelling a word out is cheaper than
pronouncing it" (see "Wordphn"): a good part of that cost is the phone-by-phone
spell-out wordphn uses for an unseen realisation, not realisation-identity itself. The
qualification is specific to this estimator, and the boundary confound does not reach
it, since word knows where the words divide too.

**The neural stage does not go that far at either size.** There the 5,000 size stays
0.60 bits/s above
wordphn on five-seed means, 3.5x the pair's floor, though 0.36 and only 2.1x at seed 0;
and 50,000 lands 0.033 above wordphn on the means, 0.3x, which is level rather than
below. Against word, 50,000 stays +0.550 and 7.3x clear. So
what the n-gram stage reads as convergence and then an overtake the neural stage reads
as a gap that closes and stops. The sweep
interpolates *towards* wordphn in both stages and passes it only in the weaker one,
which
is the same pattern as everything else here: chunking closes most of the distance for an
estimator that cannot carry context and rather less for one that can.

**The unit's standing against the word family barely moves. Its standing against phone
collapses.**

| comparison | n-gram, order 3 | 95% CI | OPT-20M | 95% CI |
|---|---:|---|---:|---|
| vs phone | -12.056 | [-12.53, -11.57] | -0.783 | [-0.88, -0.69] |
| vs wordphn | +0.537 | [0.43, 0.64] | +0.658 | [0.54, 0.77] |
| vs wordphn-BPE-200 | +0.619 | [0.51, 0.72] | +0.466 | [0.35, 0.57] |
| vs word | +0.969 | [0.81, 1.12] | +1.357 | [1.22, 1.49] |
| vs syllable | -0.084 | [-0.15, -0.02] | +0.405 | [0.35, 0.46] |

Paired speaker-level bootstrap, B = 10,000, every unit on one resample; the n-gram rows
are the order-3 100% dumps. An n-gram fit is deterministic given its training data, so
there the interval is the whole error budget, while the neural rows also have to clear
a seed floor (`design.md`, "Uncertainty"). Every difference in the table excludes zero.

Three things come out of that.

**Against phone the two estimators disagree by an order of magnitude**, -12.06 against
-0.78, and at order 5 phone recovers to 34.93 so the gap is -3.26 there (no interval:
only order 3 is dumped). The seeds sharpen this rather than soften it: the neural gap is
twice this unit's floor and no more, so one estimator puts the two units 12 bits/s
apart and the other cannot separate them. These two units describe nearly the same
object, so this is the project's own thesis in miniature rather than a fact about
phones: chunking is
worth a great deal to an estimator that cannot afford context and almost nothing to one
that can. The boundary caveat from "Bit rate" applies here too -- the confined
segmentation hands over where the words divide, so neither number is a clean modelling
comparison -- but the confound is identical in both stages and so cannot explain why
they differ.

**The mechanism is visible in the sweep's own shape.** The order at which a unit
bottoms out tracks its granularity and nothing else: order 3 for the ten word-sized
sweeps, order 4 for syllable and subwordphnx-BPE-2000, order 5 for phone and
subwordphnx-BPE-200. A piece averages 2.61 phones at vocabulary 2000, so four pieces of
context is about ten phones' worth: the chunking buys the n-gram model a span it cannot
otherwise reach, which is exactly the budget a neural LM does not need help with.

**Switching estimator buys this unit less than it buys any other unit measured, and
the four vocabularies trace out the whole trend.**

| unit | n-gram, order 3 | OPT-20M | gain | 95% CI |
|---|---:|---:|---:|---|
| phone | 43.647 | 29.599 | 14.048 | [13.54, 14.56] |
| subwordphnx-BPE-200 | 35.656 | 29.043 | 6.614 | [6.36, 6.87] |
| syllable | 31.676 | 28.411 | 3.264 | [3.10, 3.43] |
| word | 30.622 | 27.459 | 3.163 | [2.94, 3.39] |
| wordphn | 31.054 | 28.158 | 2.895 | [2.70, 3.10] |
| subwordphnx-BPE-2000 | 31.591 | 28.816 | 2.775 | [2.61, 2.94] |
| subwordphnx-BPE-50000 | 30.752 | 27.976 | 2.775 | [2.61, 2.95] |
| subwordphnx-BPE-5000 | 31.048 | 28.518 | 2.531 | [2.37, 2.69] |

Read down that column and the ordering is by how much context the unit already carries
in its own tokens: phone carries least and gains most, and the coarse subwordphnx sizes
carry enough that a trigram over them is already most of the way to what OPT-20M
extracts. The word family's own gains shrink the same way with merge depth (3.16 down to
2.31 bits/s under "Confidence intervals"), so this is one more point on a trend the
project already had.

**Sweeping the vocabulary is what turns that trend into a curve, and the curve has a
floor.** The four sizes land at 6.61, 2.78, 2.53 and 2.78, so one unit, varied only in
its vocabulary, spans most of the
range the whole project covers: vocabulary 200 sits between phone and the word-sized
units, where no other unit does, and the three larger sizes sit below all of them. The
gain more than
halves between 200 and 2,000 and then stops moving, which is the same front-loading the
bit rates show, and for the same reason: by 2,000 merges a piece is most of a word, so
little context is left for extra vocabulary to hand the n-gram model. It is **not
monotone** past that point -- 2.53 at 5,000 and back to 2.78 at 50,000 -- but the 0.24
between them is 1.4x the floor those two neural runs carry between them, so the three
coarse sizes are best read as flat rather than as a dip and a recovery. The n-gram
terms are deterministic but the neural ones carry each unit's own seed floor, so "less
than any other unit" holds comfortably against phone, word and the 200 size. Among the
bottom four places -- wordphn 2.895, 2000 and 50000 at 2.775 and 5000 at 2.531 -- the
5,000 size
is separated from wordphn on the means (3.5x) but not at seed 0 (2.1x), and not from
2,000 either way, whose floor swallows a gap of that size.

**One ranking flips, and only one half of it is established.** Under n-grams the unit
beats syllable by a small but clear 0.084 [0.020, 0.149], an n-gram fit being
deterministic; under OPT-20M it loses to syllable by 0.405, which is 1.4x its seed
floor and so within noise. Nothing else in either stage's ordering changes, and the
flip is between two units whose n-gram cells are 0.08 apart, so it is a reminder of how
little separates the middle of the ranking rather than a result in its own right --
which the seeds have since made the stronger reading of the two.

## Phone

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes
of that one shuffle. Vocabulary is built from each cell's training subset only.
Evaluated on dev-clean. Phones are unstressed (standard 39-symbol ARPABET set).

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 | OPT-20M |
|---|---:|---:|---:|---:|---:|---:|
| 25% | 60.63 | 50.46 | 43.68 | 38.76 | 36.06 | - |
| 50% | 60.63 | 50.46 | 43.66 | 38.57 | 35.40 | - |
| 75% | 60.63 | 50.46 | 43.65 | 38.49 | 35.11 | - |
| 100% | 60.63 | 50.46 | 43.65 | 38.45 | 34.93 | 29.60 |

## Syllable

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes
of that one shuffle. Vocabulary is built from each cell's training subset only.
Evaluated on dev-clean. Out-of-vocabulary syllables are spelled out via their
constituent phones (`token.split("-")`), not characters.

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 | OPT-20M |
|---|---:|---:|---:|---:|---:|---:|
| 25% | 46.22 | 35.24 | 32.98 | 33.06 | 33.25 | - |
| 50% | 46.21 | 34.91 | 32.28 | 32.30 | 32.51 | - |
| 75% | 46.20 | 34.78 | 31.94 | 31.91 | 32.13 | - |
| 100% | 46.20 | 34.68 | 31.68 | 31.61 | 31.82 | 28.41 |

## Word

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes
of that one shuffle. Vocabulary is built from each cell's training subset only.
Evaluated on dev-clean.

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 | OPT-20M |
|---|---:|---:|---:|---:|---:|---:|
| 25% | 37.12 | 32.66 | 32.30 | 32.46 | 32.62 | - |
| 50% | 36.79 | 31.90 | 31.39 | 31.54 | 31.70 | - |
| 75% | 36.68 | 31.58 | 30.96 | 31.09 | 31.26 | - |
| 100% | 36.59 | 31.32 | 30.62 | 30.74 | 30.91 | 27.46 |

## Word-BPE (200 merges)

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes
of that one shuffle. 200 word-level BPE merges are learned fresh from each fraction's
own training subset only (never from dev-clean, never reused across fractions).
Vocabulary is built from each cell's training subset only. Evaluated on dev-clean.

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 | OPT-20M |
|---|---:|---:|---:|---:|---:|---:|
| 25% | 36.51 | 32.60 | 32.32 | 32.44 | 32.57 | - |
| 50% | 36.17 | 31.78 | 31.34 | 31.48 | 31.61 | - |
| 75% | 36.07 | 31.42 | 30.88 | 31.02 | 31.15 | - |
| 100% | 35.97 | 31.13 | 30.52 | 30.65 | 30.78 | 27.76 |

## Word-BPE (1000 merges)

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes
of that one shuffle. 1000 word-level BPE merges are learned fresh from each fraction's
own training subset only (never from dev-clean, never reused across fractions).
Vocabulary is built from each cell's training subset only. Evaluated on dev-clean.

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 | OPT-20M |
|---|---:|---:|---:|---:|---:|---:|
| 25% | 36.04 | 32.61 | 32.48 | 32.59 | 32.69 | - |
| 50% | 35.70 | 31.71 | 31.46 | 31.57 | 31.68 | - |
| 75% | 35.59 | 31.32 | 30.97 | 31.08 | 31.19 | - |
| 100% | 35.49 | 31.01 | 30.59 | 30.70 | 30.81 | 28.06 |

## Word-BPE (5000 merges)

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes
of that one shuffle. 5000 word-level BPE merges are learned fresh from each fraction's
own training subset only (never from dev-clean, never reused across fractions).
Vocabulary is built from each cell's training subset only. Evaluated on dev-clean.

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 | OPT-20M |
|---|---:|---:|---:|---:|---:|---:|
| 25% | 35.39 | 32.79 | 32.80 | 32.88 | 32.96 | - |
| 50% | 35.05 | 31.85 | 31.78 | 31.86 | 31.95 | - |
| 75% | 34.93 | 31.40 | 31.26 | 31.35 | 31.44 | - |
| 100% | 34.83 | 31.05 | 30.87 | 30.96 | 31.05 | 28.56 |

## Wordphn (n-gram sweep)

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes
of that one shuffle. Vocabulary is built from each cell's training subset only.
Evaluated on dev-clean. Run 2026-10-08, after the neural result. A unit is one word
interval written as its own destressed phones joined by `-`; the tier is derived, not
read (`design.md`, "The wordphn unit"). The analysis of what this unit measures is
under "Wordphn" above.

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 | OPT-20M |
|---|---:|---:|---:|---:|---:|---:|
| 25% | 37.37 | 32.99 | 32.66 | 32.82 | 32.97 | - |
| 50% | 37.07 | 32.27 | 31.79 | 31.94 | 32.10 | - |
| 75% | 36.99 | 31.97 | 31.38 | 31.52 | 31.68 | - |
| 100% | 36.90 | 31.72 | 31.05 | 31.17 | 31.34 | 28.16 |

## Wordphn-BPE (200 merges)

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes
of that one shuffle. 200 whole-unit BPE merges are learned fresh from each fraction's
own training subset only (never from dev-clean, never reused across fractions), with
`wordphn` units as the atomic symbols, so a unit is a collocation written as phones
("of the" is `AH-V DH-AH`). Vocabulary is built from each cell's training subset only.
Evaluated on dev-clean. Run 2026-10-08, after the neural result. The analysis is under
"Wordphn", "With collocation merging".

At order 3 / 100% this is 30.97 against word-BPE-200's 30.52, so the realisation
penalty here is +0.45 bits/s, against +0.43 for the unmerged pair.

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 | OPT-20M |
|---|---:|---:|---:|---:|---:|---:|
| 25% | 36.77 | 32.95 | 32.70 | 32.83 | 32.95 | - |
| 50% | 36.48 | 32.16 | 31.76 | 31.90 | 32.02 | - |
| 75% | 36.39 | 31.82 | 31.32 | 31.46 | 31.58 | - |
| 100% | 36.31 | 31.55 | 30.97 | 31.10 | 31.23 | 28.35 |

## Wordphn-BPE (1000 merges)

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes
of that one shuffle. 1000 whole-unit BPE merges are learned fresh from each fraction's
own training subset only (never from dev-clean, never reused across fractions), with
`wordphn` units as the atomic symbols. Vocabulary is built from each cell's training
subset only. Evaluated on dev-clean. Run 2026-10-08, after the neural result. The
analysis is under "Wordphn", "With collocation merging".

At order 3 / 100% this is 31.05 against word-BPE-1000's 30.59, a realisation penalty
of +0.46 bits/s. Merging 1000 collocations leaves wordphn exactly where it started,
31.05 either way, as it nearly does for word.

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 | OPT-20M |
|---|---:|---:|---:|---:|---:|---:|
| 25% | 36.29 | 32.98 | 32.87 | 32.97 | 33.07 | - |
| 50% | 35.99 | 32.12 | 31.89 | 32.00 | 32.10 | - |
| 75% | 35.91 | 31.74 | 31.42 | 31.53 | 31.63 | - |
| 100% | 35.82 | 31.45 | 31.05 | 31.16 | 31.26 | 28.48 |

## Wordphn-BPE (5000 merges)

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes
of that one shuffle. 5000 whole-unit BPE merges are learned fresh from each fraction's
own training subset only (never from dev-clean, never reused across fractions), with
`wordphn` units as the atomic symbols. Vocabulary is built from each cell's training
subset only. Evaluated on dev-clean. Run 2026-10-08, after the neural result. The
analysis is under "Wordphn", "With collocation merging".

At order 3 / 100% this is 31.32 against word-BPE-5000's 30.87, a realisation penalty
of +0.45 bits/s. This is the cell that decides the question the neural stage left open:
the penalty does **not** grow with merge depth under n-grams, so the blow-up OPT-20M
shows at 5000 merges is a property of that estimator, not of the unit.

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 | OPT-20M |
|---|---:|---:|---:|---:|---:|---:|
| 25% | 35.68 | 33.15 | 33.16 | 33.24 | 33.32 | - |
| 50% | 35.37 | 32.23 | 32.18 | 32.26 | 32.35 | - |
| 75% | 35.28 | 31.82 | 31.70 | 31.78 | 31.87 | - |
| 100% | 35.19 | 31.49 | 31.32 | 31.41 | 31.49 | 29.58 |

## Subwordphnx-BPE (n-gram sweep, 200)

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes
of that one shuffle. 158 subword BPE merges over *phones* are learned fresh from each
fraction's own training subset only, with every merge confined to a single word. That is
the merge count the neural unit has at vocabulary 200. Vocabulary is built from each
cell's training subset only. Evaluated on dev-clean. Run 2026-10-12. Nothing can go out
of vocabulary, as for phone. The analysis is under "Subwordphnx-BPE", "Under n-grams".

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 | OPT-20M |
|---|---:|---:|---:|---:|---:|---:|
| 25% | 52.98 | 43.23 | 36.26 | 34.37 | 34.57 | - |
| 50% | 53.13 | 43.30 | 35.94 | 33.59 | 33.72 | - |
| 75% | 53.12 | 43.30 | 35.77 | 33.17 | 33.21 | - |
| 100% | 53.12 | 43.30 | 35.66 | 32.88 | **32.87** | 29.04 |

**At this vocabulary the unit behaves like phone, not like a word-sized unit.** Its
floor is at order 5 rather than order 3, and the order-3 cell overstates it by 2.8
bits/s, which is why the headline table's two n-gram columns differ so much for this row
alone among the subwordphnx sizes. A piece averages only 1.66 phones here, so a trigram
over pieces is barely five phones of context. Unlike phone it has stopped falling by
order 4 (32.88 against 32.87), so the floor is real rather than an artefact of the
order cap.

## Subwordphnx-BPE (n-gram sweep, 2000)

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes
of that one shuffle. 1,958 subword BPE merges over *phones* are learned fresh from each
fraction's own training subset only (never from dev-clean, never reused across
fractions), with every merge confined to a single word. That is the merge count the
neural unit has at vocabulary 2,000, which is what makes the two stages the same unit.
Vocabulary is built from each cell's training subset only. Evaluated on dev-clean. Run
2026-10-09, after the neural result. Nothing can go out of vocabulary, as for phone.
The analysis is under "Subwordphnx-BPE", "Under n-grams".

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 | OPT-20M |
|---|---:|---:|---:|---:|---:|---:|
| 25% | 44.80 | 35.48 | 32.85 | 32.89 | 33.04 | - |
| 50% | 44.81 | 35.21 | 32.18 | 32.15 | 32.33 | - |
| 75% | 44.78 | 35.10 | 31.84 | 31.77 | 31.95 | - |
| 100% | 44.78 | 35.03 | 31.59 | **31.48** | 31.67 | 28.82 |

**Its minimum is at order 4**, 31.48 against 31.59 at order 3, which it shares with
syllable (31.61 against 31.68) and with no other sweep. Every word-sized unit bottoms
out at order 3, and phone and subwordphnx-BPE-200 only at order 5; this unit and
syllable sit between the two groups in granularity and reach their floor one order
further out than the coarse units. That is what the granularity buys: a piece averages
2.61 phones, so four pieces of context is about ten phones' worth, which is a span no
phone-level n-gram in this project can afford.

## Subwordphnx-BPE (n-gram sweep, 5000)

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes
of that one shuffle. 4,958 subword BPE merges over *phones* are learned fresh from each
fraction's own training subset only, with every merge confined to a single word. That is
the merge count the neural unit has at vocabulary 5,000. Vocabulary is built from each
cell's training subset only. Evaluated on dev-clean. Run 2026-10-12. Nothing can go out
of vocabulary, as for phone. The analysis is under "Subwordphnx-BPE", "Under n-grams".

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 | OPT-20M |
|---|---:|---:|---:|---:|---:|---:|
| 25% | 41.46 | 33.71 | 32.36 | 32.49 | 32.62 | - |
| 50% | 41.49 | 33.37 | 31.67 | 31.76 | 31.91 | - |
| 75% | 41.47 | 33.20 | 31.31 | 31.38 | 31.54 | - |
| 100% | 41.46 | 33.08 | **31.05** | 31.09 | 31.24 | 28.52 |

**At this vocabulary it has become a word-sized unit, by every n-gram signature it
has.** Its floor moves back to order 3, where the ten word-sized sweeps bottom out and
the two finer subwordphnx sizes do not, and the floor itself, 31.0482, lands on
wordphn's 31.0536 to within 0.005 bits/s. 85.8% of dev-clean word tokens are a single
piece at this vocabulary, so there is little left for the unit to be other than wordphn.

## Subwordphnx-BPE (n-gram sweep, 50000)

Pooled and shuffled (seed=0) 281,239 training utterances; fractions are nested prefixes
of that one shuffle. 49,958 subword BPE merges over *phones* are learned fresh from each
fraction's own training subset only, with every merge confined to a single word. That is
the merge count the neural unit has at vocabulary 50,000. Vocabulary is built from each
cell's training subset only. Evaluated on dev-clean. Run 2026-10-14. Nothing can go out
of vocabulary, as for phone. The analysis is under "Subwordphnx-BPE", "Under n-grams".

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 | OPT-20M |
|---|---:|---:|---:|---:|---:|---:|
| 25% | 37.19 | 32.56 | 32.21 | 32.37 | 32.52 | - |
| 50% | 36.99 | 31.93 | 31.44 | 31.58 | 31.74 | - |
| 75% | 36.93 | 31.66 | 31.04 | 31.18 | 31.33 | - |
| 100% | 36.90 | 31.45 | **30.75** | 30.87 | 31.02 | 27.98 |

**This is the one n-gram cell in the project that is not distinguishable from `word`.**
The order-3 100% floor, 30.7518, sits +0.130 [-0.015, +0.263] from word's 30.6217 and
-0.302 [-0.384, -0.225] from wordphn's 31.0536, so going from 4,958 merges to 49,958
takes the unit from level with wordphn to clear of it and level with word. It keeps the
word-sized signature of the 5,000 size -- a floor at order 3, still falling from 25% to
100% at every order -- and the data-size trend is the shallowest of the four sizes,
32.21 to 30.75 against the 200 size's much steeper climb, which is what a unit with
little left to learn from more data looks like.

## OPT-20M notes

The `OPT-20M` column is the neural stage's bit rate (`llm/eval_bitrate.py`): one
downsized OPT-125M-derived model per unit (~20M parameters, d=512, 6 layers, 8
heads), trained on 100% of LibriSpeech 960h, so it only ever populates the 100% row.

**Interpretive caveat.** The neural LM conditions on a full 2048-token context,
against `ngram/`'s much shorter fixed orders, so a lower `OPT-20M` value is not
"this unit is more efficient" in the same sense an order-to-order n-gram comparison
would be.

**Every number here is post-fix.** The conventions the project's corrections
established are in `design.md`, "Evaluation protocol" and "Tokenisation and backoff
in the neural stage".

## Qwen pretrained-LM (side experiment)

Not part of the `OPT-20M` comparison above: a single pretrained LM (no training),
scored at the subword-bpe "level" (Qwen's own tokenizer stands in for the unit).
Reproduces the predecessor project's own "estimator comparison"
(`docs/assets/2026-08-30/paper_2.md` Sec 4.6) on LibriSpeech; see
`qwen/eval_bitrate.py` for the full design (primer construction, scoring
convention, KV-cache reuse).

| model | bit rate (bits/s) | perplexity | dev-clean tokens |
|---|---:|---:|---:|
| Qwen2.5-1.5B (primed) | 25.90 | 94.37 | 62,551 |
| Qwen2.5-3B (primed) | 25.26 | 84.34 | 62,551 |

Both lower than every unit trained in this project, including phone's 29.60 and
word's 27.46 -- consistent with the project's own thesis (unit "efficiency" depends
on the (unit, model class, data budget) triple, not just the unit): a model
pretrained on an unmeasurable quantity of text moves the number well past what any
controlled, single-fixed-budget comparison here can reach. 3B beats 1.5B (25.26 <
25.90, a 0.64 bits/s gap), the same direction the predecessor found on FLEURS
(1.5B/3B: 18.02/17.93, a much smaller 0.09 bits/s gap -- not directly comparable
across corpora/languages, but the same qualitative ordering). Single run each, not a
controlled comparison.
