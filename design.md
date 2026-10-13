# Design reference

The current rules of this project and the reasoning behind them: what is measured,
how each unit is tokenised, how out-of-vocabulary items are paid for, and how the
two model classes are trained and scored. This is the authoritative technical
reference. Result numbers live in `results.md` and exact commands in
`reproduce.md`; superseded material is in `docs/archive/`.

## Scope and metrics

The research question is which linguistic unit encodes speech most efficiently in
bits per second, and whether the answer depends on the modelling budget used to
estimate it. Everything trains and evaluates on LibriSpeech.

**Bit rate** is the only quantity compared across units:

$$
\text{bit rate} = \frac{\sum_u \text{bits}(u)}{\sum_u \text{duration}(u)}
$$

summed over all dev-clean utterances, where $\text{duration}(u)$ is speech-only
seconds from the alignment and $\text{bits}(u)$ is the code length the model
implies for that utterance:

$$
\text{bits}(u) = -\log_2 P(u)
$$

$P(u)$ is the probability the model assigns to the *whole* utterance, factorised
autoregressively over its tokenisation $t_1 \ldots t_m$ with an explicit
end-of-utterance symbol $t_{m+1}$, so that the model also pays for the sentence
stopping where it does:

$$
P(u) = \prod_{i=1}^{m+1} P\!\left(t_i \mid t_1, \ldots, t_{i-1}\right),
\qquad
\text{bits}(u) = -\sum_{i=1}^{m+1}
  \log_2 P\!\left(t_i \mid t_1, \ldots, t_{i-1}\right)
$$

**The start symbol is context, never a term in this sum.** The product runs from
$i = 1$, the first real token, so at $i = 1$ the conditioning set is a
start-of-utterance symbol rather than the empty sequence. That symbol is what the
first token is predicted *from*; it is never itself predicted, so it contributes no
bits. Only the $m$ content tokens and the single end-of-utterance symbol are paid
for. Scoring it as well would charge every unit a constant per utterance and, worse,
a *different* constant per unit, since each stage's start symbol has its own
probability under its own model.

The two stages instantiate this differently, but both compute a code length for
the same underlying text:

- **`llm/`**: exactly the sum above, teacher-forced, one term per token. The
  conditioning is truncated to the 2048-token context window, and an
  out-of-vocabulary item has already been expanded by the tokeniser into its
  `~`-marked fallback pieces, so those pieces are ordinary tokens contributing
  ordinary terms. There is no separate spell-out model and no `<unk>` event.
- **`ngram/`**: the conditioning is truncated to the previous $k-1$ tokens, and an
  out-of-vocabulary token contributes *two* terms instead of one,
  $-\log_2 P(\texttt{unk} \mid h)$ for the unknown event plus the spelling model's
  own bits for its identity. See "Backing off to the spelled-out model".

Both are **upper bounds** on the true code length, never underestimates. Each text
has exactly one tokenisation under the deterministic encode rule, so scoring that
one sequence lower-bounds the text's probability; and `ngram/`'s composite is
additionally a sub-probability (see "Known deficiency").

This is unit-agnostic by construction: two tokenisations of the same audio differ in
token count but not in duration, so the denominator is shared. Verified directly:
the `words`, `phones` and `syllables` tiers give an identical dev-clean total of
15,845.690 s, to the millisecond. The numerator is comparable for the same reason:
$\text{bits}(u)$ is the cost of transmitting the same utterance either way, whatever
the tokenisation used to get there.

**Perplexity is reported but is not comparable**, in two separate ways:

- *Across units*, because the token count it normalises by is a property of the
  tokenisation.
- *Across stages*, because the two stages count differently. `ngram/` divides
  end-of-sentence-inclusive bits by a token count that excludes `</s>`, and counts
  an unspelled unit once; `llm/` includes the boundary token in both, and counts
  each spelled-out character as a token. For word this is 54,402 against 60,412.

Treat perplexity as a within-stage, within-unit diagnostic only.

### Uncertainty

Bit rate is reported with 95% confidence intervals from `utils/bootstrap.py`, which
resamples dev-clean using the per-utterance bits that `llm/eval_bitrate.py` and
`ngram/per_utterance.py` dump.

**How speakers enter it.** A dev-clean utterance key is
`speaker-chapter-utterance`, so the speaker is the first field: 2,703 utterances
across **40 speakers**, between 36 and 101 utterances each. Two resampling schemes
are computed:

- *Utterance-level*: draw 2,703 utterances with replacement, independently. Every
  resample has exactly 2,703 utterances.
- *Speaker-level (cluster)*: draw 40 **speakers** with replacement, and take every
  utterance belonging to each drawn speaker. A speaker therefore enters 0, 1, 2 or
  more times, always as a whole block, so within-speaker correlation survives into
  the resample. Because speakers differ in size the resample varies between about
  2,374 and 3,096 utterances, which is harmless for a ratio of sums.

The cluster scheme is the honest one. A speaker's rate, voice and vocabulary make
their utterances correlated, so the effective sample size is nearer 40 independent
units than 2,703, and treating utterances as independent overstates precision. The
effect is large on a single unit and almost absent on a comparison:

- Marginal interval on word: utterance-level [27.26, 27.66], width 0.39;
  speaker-level [26.64, 28.27], width **1.63**.
- Paired difference, word minus word-BPE-200: utterance-level [-0.353, -0.255],
  width 0.098; speaker-level [-0.362, -0.245], width **0.117**.

Clustering widens the marginal interval **4.2x** but the paired difference only
**1.2x**, because the speaker effect is shared by both models and cancels in the
difference. That is also why comparisons must use the paired interval: marginal
intervals of +/-0.8 bits/s overlap almost completely while the differences of
interest are 0.1 to 1 bits/s, so asking whether marginals overlap would hide
effects that are in fact decisive. The difference is recomputed on the same
resample instead.

**What "significant" means in the tables.** The 95% percentile interval on the
paired difference excludes zero: a two-sided test at `alpha = 0.05`, with
`B = 10000` resamples at a fixed seed. Five caveats:

- **Multiple comparisons are not corrected for.** Well over 20 comparisons are now
  reported, so a few false positives would be expected at `alpha = 0.05`. The
  borderline cases were checked against a Bonferroni threshold (n = 20, a 99.75%
  interval): the closest, n-gram word versus word-BPE-200 at +0.10 bits/s, still
  excludes zero at [0.034, 0.161], two-sided `p < 2e-4`. That `n` is now too small,
  but every borderline case clears it by so much that a larger one changes no verdict,
  and the verdicts that are close are reported as close rather than as findings.
- **Significant is not important.** That same 0.10 bits/s is about 0.3%. With
  2,703 paired utterances very small effects are detectable, so effect size is
  always quoted next to the interval.
- **Percentile intervals, not BCa**: no bias correction or acceleration, which can
  be slightly off under skew. Immaterial at the distances involved here.
- **Only 40 clusters.** Cluster bootstraps with few clusters can under-cover, so
  the speaker-level intervals are the best available rather than exact.
- **Selection on dev-clean is invisible to this.** dev-clean is also the neural
  model-selection set (validation and `-best.pt`), so those numbers are mildly
  optimistic in a way no bootstrap over the same set can detect.

**Training variance is a separate axis.** The bootstrap treats each model as
fixed, so it measures evaluation uncertainty only. For `ngram/` that is the whole
story: a fit is deterministic given its training data, with no initialisation and
no SGD, so re-running at a different seed changes nothing. For `llm/` there is a
second source, measured on 2026-09-07/08 with three extra seeds per unit, on
2026-10-08/09 for the four wordphn units, on 2026-10-09 for `subwordphnx-BPE-2000`,
and on 2026-10-12/13 with one further seed for every unit in `results.md`'s headline
table:

- Per-unit SD across **five seeds** is **0.043 to 0.118 bits/s** for the seven original
  units, so the noise floor for a *difference* between two independently trained
  ones, `sqrt(sa^2 + sb^2)`, is roughly **0.1 bits/s**. That is the figure a neural
  gap between them must clear.
- **The floor is not universal.** Three of the twelve five-seed units are far worse:
  `wordphn-BPE-1000` and `wordphn-BPE-5000` at **0.235** each and
  `subwordphnx-BPE-2000` at **0.279**, the worst in the project. A difference
  involving one of them must clear about **0.24 to 0.28 bits/s** and a difference
  between two of them **0.33 to 0.39**. Every other unit, wordphn and
  wordphn-BPE-200 included, sits in the original 0.04 to 0.12 band.
- **Within that band the floor is roughly architecture-independent**: under 3x
  across vocabularies from 41 to 63,002 tokens and two batch configurations. The
  three outliers show that it cannot simply be assumed for a new unit, which is why
  the four wordphn units were seeded rather than read against an assumed 0.07; see
  `results.md`, "With collocation merging", for what that changed.
- **subwordphnx-BPE at 200 and 5,000 have two runs each**, not five, and read **0.015
  and 0.056**: tight, but on one degree of freedom. Until 2026-10-12 they were single
  runs read against **0.289** borrowed from `subwordphnx-BPE-2000`, and seeding that
  sibling is the clearest demonstration in the project of why borrowing is unsafe --
  it had been written up against 0.109 and 0.154, so every ratio in `results.md`,
  "Subwordphnx-BPE", fell by half or more. The second run at each endpoint replaces the
  borrowing with a weak measurement, which is enough to establish the sweep's two ends
  against each other and against phone, and not enough for anything that passes through
  vocabulary 2,000. More seeds there, rather than at the ends, are what the rest needs.
- The seed varies initialisation and the random training-window sampling. It does
  **not** vary the augmentation draw, fixed when the mmap is built at
  `AUGMENT_SEED = 0`, nor the learned BPE merges, so this is retrain-on-fixed-data
  variance rather than full pipeline variance.
- Five runs give four degrees of freedom, so treat each SD as an order of
  magnitude. Several units looked far tighter after three seeds than they proved to
  be, and syllable did it again at the fifth run, doubling from 0.038 to 0.074, so
  short-run reads are not safe here.
- Of the 21 pairwise comparisons among the original seven units, **16 are clear, 2
  marginal and 3 within noise**; the full set over all fourteen units is 91, of which
  the 34 further pairs within the wordphn family and the original seven are in
  `results.md`, "Wordphn", and the subwordphnx family's own in "Subwordphnx-BPE".
  The defensible ranking over the four paper units is
  `word < word-BPE-200 < word-BPE-1000 < {syllable, word-BPE-5000} < phone` with
  the braced pair unordered (1.1); every other adjacent step is clear.
  **subword-BPE cannot be placed** among the word-BPE points at all, being
  within noise of both word-BPE-1000 (0.1) and word-BPE-200 (1.9).
- Seed 0 sits low in its distribution for word and word-BPE-200 and high for
  subword-BPE, so the reported word-versus-word-BPE-200 gap is slightly
  conservative. It is **badly unrepresentative** for the two noisy units: lowest of
  five for wordphn-BPE-1000 (28.48 against a mean of 28.781) and highest of five for
  wordphn-BPE-5000 (29.58 against 29.274), which between them manufactured a
  depth-dependent interaction that the seeds and the n-gram sweeps both dissolve.
  Reported numbers remain the `seed = 0` run throughout; see `results.md`,
  "Seed variance".

## Data

Each LibriSpeech alignment TextGrid under `data/librispeech-alignments/` carries
three tiers:

- `words`: word-level intervals, lowercase, no punctuation. Matches `^[a-z']+$`.
- `phones`: ARPABET phones with stress digits (e.g. `AH1`, `ER0`).
- `syllables`: already syllabified and destressed, hyphen-joined (e.g. `D-ER-T`).

A fourth unit, `wordphn`, is **derived** rather than read: word spans whose text is
replaced by their own realised phones (see "The wordphn unit", below).

All three share timing, and empty-text intervals mark silence and are excluded from
speech time. So **no g2p and no separate syllabifier are needed** for any unit,
unlike the predecessor project which needed both for FLEURS. Verified: zero empty or
whitespace-only tokens across all three dev-clean tiers and across the full 960h
`words` tier (9,403,427 tokens), so silence never leaks in as a token.

| set | utterances | speech duration | word tokens | word vocab |
|---|---:|---:|---:|---:|
| LibriSpeech 960h | 281,239 | 769.8h (2771365s) | 9,403,427 | 89,048 |
| dev-clean | 2,703 | 4.4h (15846s) | 54,402 | 8,332 |

| set | phones | phone types* | syllables | syllable types |
|---|---:|---:|---:|---:|
| LibriSpeech 960h | 33,464,416 | 39 | 13,025,947 | 13,656 |
| dev-clean | 194,827 | 39 | 75,910 | 4,082 |

\* unstressed.

**Splits.** Training is the pooled 960h set (train-clean-100, train-clean-360,
train-other-500), shuffled once with seed 0, with data fractions taken as nested
prefixes by utterance count. Evaluation is dev-clean. **dev-other, test-clean and
test-other stay untouched** for a final reporting pass.

## The n-gram estimator

Interpolated absolute discounting, implemented in `ngram/ngram_lm.py`
(`AbsoluteDiscountingLM`, `SpellingLM`), ported from the predecessor project's
`inforate.py` and matching `paper_2.md` §2.2-2.3 and Appendix D.

### Interpolation across orders

For an order-$k$ model, history $h$ of length $k-1$ and per-order discount $D_k$:

$$
P_k(t \mid h) = \frac{\max(c(ht) - D_k,\ 0)
  + D_k\, N_{1+}(h\cdot)\, P_{k-1}(t \mid h')}{c(h)}
$$

where $c(\cdot)$ is a training count, $h'$ drops the first symbol of $h$, and
$N_{1+}(h\cdot) = |\{t : c(ht) > 0\}|$ is the number of distinct continuations
observed after $h$. This recurses down to the unigram base case:

$$
P_1(t) = \frac{\max(c(t) - D_1,\ 0) + D_1\, N_{1+}(\cdot)\, u(t)}{N}, \qquad
u(t) = \frac{1}{|V| + 1}
$$

with $N$ the training token count and $V$ the training vocabulary. The $+1$ in the
floor $u(t)$ reserves mass for `<unk>`. Every conditional sums to one over
$V \cup \{\texttt{unk}\}$ (checked by `assert_normalised`).

**What $N_{1+}(\cdot)$ means here.** The history is empty at the unigram level, so
"distinct continuations of $h$" becomes distinct continuations of nothing: it is
simply **the number of token types seen at least once in training**, which is the
vocabulary size $|V|$. It counts *types*, not occurrences, so a word appearing
90,000 times and a word appearing once each contribute exactly 1. `</s>` is
included, as it is predicted like any other token; `<s>` is not, as it only ever
serves as context. In the code this is `ctx_types[1][()]`, and it does equal
`len(self.vocab)`.

Its role is bookkeeping: discounting subtracts $D_1$ from each of those $|V|$ type
counts, so $D_1 N_{1+}(\cdot)$ is the total count mass removed, and dividing by $N$
turns it into the probability mass handed to the uniform floor $u(t)$. Taking
exactly as much as was given away is what makes the unigram distribution sum to
one. The same reading holds at higher orders, where $N_{1+}(h\cdot)$ is the number
of distinct types seen following that particular history.

Implementation detail: when a context $h$ at level $k$ was never observed
($c(h) = 0$), the code skips that level (`if total == 0: continue`) and passes the
lower-order estimate straight through, so $P_k(t \mid h) := P_{k-1}(t \mid h')$.

**No add-alpha smoothing anywhere.** With small vocabularies and small contexts it
makes higher-order models score worse than lower-order ones, which is a smoothing
artefact rather than a finding.

### Discount estimation

$D_k$ is estimated per order from that order's own training counts
(`_estimate_discount`), rather than fixed across every order and unit as in the
predecessor project. Chen and Goodman's data-driven estimate:

$$
D_k = \frac{n_1^{(k)}}{n_1^{(k)} + 2n_2^{(k)}}
$$

where $n_1^{(k)}$ and $n_2^{(k)}$ are the number of distinct order-$k$ $n$-grams
occurring exactly once and exactly twice in training. Two practical guards, neither
part of the formula itself:

- **Fallback**: if $n_1^{(k)} + 2n_2^{(k)} = 0$, fall back to a fixed default
  ($D = 0.75$). A message tagged with the class name (e.g. `[SpellingLM]`) prints
  whenever this fires.
- **Clamp**: otherwise clamp $D_k$ to $[0.1, 0.9]$, guarding against near-0 or
  near-1 estimates from small samples.

Rationale: the project compares units with very different vocabulary sizes, so a
single shared $D$ risks fitting some units' count statistics better than others',
exactly the kind of hidden unit-dependent artefact this project exists to rule out.
Estimating $D_k$ per fit keeps the *procedure* fixed across units without fixing the
*numeric value*.

**Expect the fallback to fire routinely for the spelling model, not the token
model.** With tens of thousands of word types spelled into ~27 characters, every
character is seen far more than twice, so the spelling model's order-1 level almost
always has no singleton or doubleton counts and falls back to 0.75. This is expected
and harmless. The `[SpellingLM]` tag distinguishes it from a genuine token-level
fallback.

### Sentence probability

The end-of-sequence symbol is scored, so the model accounts for the sentence
stopping:

$$
P_M(w) = \prod_{i=1}^{m} P\!\left(t_i \mid t_{i-k+1}^{\,i-1}\right)
\cdot P\!\left(\texttt{</s>} \mid t_{m-k+2}^{\,m}\right)
$$

Where that context runs off the front of the sentence it is filled with $k-1$
copies of `<s>`, so the first token is predicted from a genuine start-of-sentence
context rather than from nothing. `<s>` is padding only: `fit` adds it to the
context n-grams but never to the vocabulary, and `sentence_bits` starts its loop at
the first real token, so `<s>` is neither scored nor given probability mass. `</s>`
is in the vocabulary, because it is predicted.

### Backing off to the spelled-out model

Any token outside the training vocabulary is mapped to `<unk>`, then a second model
pays to spell it out:

$$
-\log_2 P_M(w) = \underbrace{-\sum_i \log_2 P(\tilde t_i \mid \tilde h_i)}_{\text{token
level}} + \underbrace{\sum_{i:\ \tilde t_i = \texttt{unk}} -\log_2
P_{\mathrm{spell}}(\mathrm{spelling}(t_i))}_{\text{spell-out}}
$$

where $\tilde t_i$ is $t_i$ mapped through the vocabulary and
$\mathrm{spelling}(\cdot)$ decomposes a token into finer subunits (see "Per-unit
spelling"). Because `<unk>` has zero count at every order, $\max(c - D, 0) = 0$ for
it everywhere, so $P(\texttt{unk} \mid h)$ is built entirely from discount mass
passed down the same recursion; at the unigram base:

$$
P(\texttt{unk}) = \frac{D\, N_{1+}(\cdot)\, \frac{1}{|V|+1}}{N}
$$

$P_{\mathrm{spell}}$ is itself an `AbsoluteDiscountingLM` (order 3, via
`build_spelling_lm`), fit over subunits: its training "sentences" are the spellings
of every vocabulary token type. It has its own uniform floor over the training
subunit set plus `<unk>`, so it bottoms out and assigns non-zero probability to any
string with no further backoff. Together with the end-of-word symbol below, this is
what makes the composite a proper distribution over all sentences: **no utterance
is ever skipped**, whatever the tokenisation.

**The spelling model scores its own end-of-word symbol.** `SpellingLM` brackets a
spelling exactly as the token level brackets a sentence: `<s>` padding for context,
a final `</s>` that *is* scored. One level down, that symbol means end-of-*word*
rather than end-of-sentence, and it is what pays for the spelling stopping where it
does, so a short spelling is not simply a free prefix of a longer one.

The floor and the end symbol do different jobs, and both are load-bearing: the
floor makes every *symbol* affordable, the end symbol makes every *length*
affordable. Without the latter, $P_{\mathrm{spell}}$ would be a separate
distribution over strings of each fixed length, with no mass spent on stopping, and
its total over all strings would exceed one, which is not a valid code and would
under-charge for spelled-out tokens. Verified by brute-force enumeration over every
string up to length 13 on a toy alphabet: the mass sums to $0.99999995$ with the
end-of-word symbol and to $3.12$ without it.

The `ngram/` vocabulary is every type seen in training, with **no minimum-count
threshold** at any unit. This differs deliberately from `llm/`; see
"Vocabulary rules" below.

### Per-unit spelling

`spelling(token)` is supplied by whichever sweep is active, so the same
`SpellingLM` mechanism stays correct across units without hardcoding "character"
into the class:

- **word**: `list(token)` -- characters.
- **syllable**: `token.split("-")` -- phones; free, since the `syllables` tier is
  already hyphen-joined phone labels, no lookup needed.
- **phone**: `list(token)` -- own letters. Never fires in practice, given the
  closed inventory.
- **word-BPE**: `list(token)` -- characters, matching word.
- **wordphn**: `token.split("-")` -- phones; free for the same reason as syllable,
  since the unit is already a hyphen-joined phone sequence.
- **subwordphnx-BPE**: `token.split("-")` -- phones, for the same reason again. A piece
  *is* its phone sequence, so this identifies it uniquely. Unreachable rather than
  merely rare: every phone is an atomic symbol and every merged piece was learned from
  the fraction's own training data, so a dev-clean piece is in vocabulary by
  construction, exactly as for `phone`.
- **wordphn-BPE**: phones, with the space between two realisations kept as its own
  symbol. This is the analogue of word-BPE's `list(token)`, which likewise keeps the
  space between merged words as just another character. Flattening to phones and
  *dropping* the boundary would let `AH-V DH-AH` and a hypothetical single word
  `AH-V-DH-AH` spell out identically, which is the invalid-bound trap below in another
  guise: a spell-out has to transmit enough to identify the unit uniquely. The boundary
  branch never fires in practice, for exactly the reason word-BPE's space never appears
  -- an out-of-vocabulary unit is always a single never-merged unseen realisation --
  verified at the hardest cell, 25% of the data with 5000 merges, where none of the 583
  out-of-vocabulary dev-clean units is multi-word. One of those 5,000 merged units is
  never emitted by the encoder at all, since encoding applies merges in rank order
  rather than the order they were learned in, so it cannot reach dev-clean either.

**A spell-out must transmit which unit occurred.** word-BPE spelt out to its
constituent *words* until 2026-09-07, which was an invalid bound rather than merely an
inconsistent one: an out-of-vocabulary word-BPE unit is always a single never-merged
unseen word, because merges fire only on pairs of known training words, so splitting on
space returned a one-element list that scored straight to `<unk>`. Every unseen word
then cost the same and the code never said which word it was. Measured at order 3, 200
merges, 100% data: 30.28 bits/s under the old rule against 30.52 once the word's
identity is paid for. A two-level unit-to-words-to-characters cascade was also tried
and is valid but looser, 30.92 bits/s, because it charges a second unknown event for
information the unit-level `<unk>` already conveyed; since every rule here is an upper
bound, the tightest valid one wins, so plain characters it is.

### Known deficiency, inherited from the predecessor project

$P_{\mathrm{spell}}$ was trained on in-vocabulary spellings and readily regenerates
them, so part of the mass reserved for `<unk>` leaks back onto strings already in
$V$. The composite is therefore a sub-probability rather than a probability, and
reported bits are an upper bound on the true code length, never an underestimate.

**Not double-counting, a shortfall.** Each token is scored via exactly one path: an
in-vocabulary token costs $-\log_2 P(w \mid h)$ from the ordinary recursion alone,
and an out-of-vocabulary one costs
$-\log_2 P(\texttt{unk} \mid h) - \log_2 P_{\mathrm{spell}}(\mathrm{spelling}(w))$,
the standard two-stage decomposition. Nothing is scored twice. The deficiency is
that $P_{\mathrm{spell}}$'s support is not restricted to OOV strings. Summing every
token's scored probability over $V \cup \{\texttt{unk}\}$ gives

$$
\sum_{w \in V} P(w \mid h)
  \;+\; P(\texttt{unk} \mid h) \sum_{w \notin V} P_{\mathrm{spell}}(w)
\;=\; 1 \;-\; P(\texttt{unk} \mid h)\, L \;<\; 1,
$$

confirming the composite is under-normalised, not over. Appendix D of `paper_2.md`
measures this at no more than 0.03 bits/s for the FLEURS comparison. We do not
renormalise, matching the predecessor, and report the un-renormalised bound.

## Tokenisation and backoff in the neural stage

`llm/` takes a different approach to the same problem. Rather than an `<unk>` event
plus a separate spelling model, **rare and unseen units are represented as their
constituent finer units inside the same token stream the transformer already
models**, at both train and test time. Everything stays in one homogeneous model:
no external component, no train/test mismatch, and no sub-probability leakage to
bound.

**The fallback marker.** Every fallback subunit gets its own vocabulary entry,
prefixed with `FALLBACK_MARKER = "~"`. So `~a`, the letter "a" used inside a
spelled-out word, is a *different id* from `a`, the word. Without this the fallback
role silently aliases onto an unrelated whole-unit entry: confirmed live to affect
all 26 letters for word and word-BPE, and 19 single-phone syllables. Marking is
uniform rather than applied only to currently-colliding entries, so it stays
correct if a vocabulary changes. `_MarkedFallbackTokenizer._decode` strips the
marker, so decoded text is unaffected.

**The encode rule is deterministic and identical at train and test time**: a unit
in vocabulary becomes its own id; otherwise the sequence of `~`-marked subunit ids
is emitted. There is no separate `<unk>` case to design for.

How often the fallback actually fires on dev-clean, as a share of emitted pieces:
word 6.57%, word-BPE-200 7.27%, wordphn 5.32%, wordphn-BPE 5.83% to 6.79% over
200 to 5000 merges, syllable 0.15%, and phone, subword-BPE and every subwordphnx
size 0%.
Word-BPE fires on exactly the same 3,793 characters as word, over a smaller
denominator because merging shortens the sequence. Its rate rose with
`--min-count 2`, which discarded the hapax types that previously covered those
words: at 1000 merges it was 5.47% before the threshold and is 7.78% after.
Wordphn-BPE stands in the same relation to wordphn, firing on the same 3,031
phones at every merge count, over 51,958, 48,533 and 44,661 pieces at 200, 1000
and 5000 merges rather than wordphn's 56,992.
Subword-BPE's `<unk>` exists but never fires (0 of 79,234 tokens), and
subwordphnx-BPE's does not either, so neither has an unpaid-unknown loophole: both
alphabets are closed over their input, 27 characters for one and 39 phones for the
other, so anything not covered as a piece is still covered symbol by symbol.

| unit | vocabulary | fallback |
|---|---|---|
| phone | closed ARPABET, 39 | none |
| subword-BPE | 2,000 BPE | `<unk>`, never fires |
| subwordphnx-BPE | 2,000 BPE | `<unk>`, never fires |
| word | count >= 2, plus `~`chars | to characters |
| wordphn | count >= 2, plus `~`phones | to phones |
| syllable | count >= 2, plus `~`phones | to phones |
| word-BPE | units, plus `~`chars | to words, then characters |
| wordphn-BPE | units, plus `~`phones | to wordphn, then phones |

**Every `vocab_size` in this project is the tokeniser vocabulary, two larger than
the linguistic inventory**, because `<bos>` and `<pad>` occupy ids 0 and 1 in every
unit. Phone is the clearest case: 39 ARPABET phones but `vocab_size = 41`. Likewise
word's 58,006 is 57,977 word types plus 27 `~`-marked characters plus the two
specials. `<bos>` doubles as the end-of-utterance marker (see "Evaluation
protocol"); `<pad>` only ever appears as batch padding and is excluded from scoring
via `ignore_index`, so it is never a prediction target. The `params` column in
`results.md` is computed from these tokeniser totals, so the two are consistent.

**Word-BPE uses a two-level backoff** in this stage, unlike `ngram/`: an
out-of-vocabulary unit falls back to its constituent *words* first, and only a word
that is itself out of vocabulary is spelled as characters. This is a guard rather
than a behaviour: an OOV unit is in practice always a single unseen word, verified
at 200 and 5000 merges, so the two routes coincide. It matters if a *merged* unit
ever goes out of vocabulary, which `min_count = 2` makes possible in principle,
because spelling one out directly would charge character cost for words the model
knows and would emit the unit's internal space, which has no `CHARACTERS` entry and
fails with "WordLevel error: Missing [UNK] token".

### The wordphn unit

`wordphn` keeps word-sized spans but identifies each one by **its realised phones
rather than its spelling**: "abbey" is `AE-B-IY`. No TextGrid carries such a tier, so
`data_common.word_phone_intervals` derives it, and `WORDPHN_TIER` is accepted wherever
a real tier name is (`read_tier`, validation, evaluation), so nothing downstream needs
to know it is derived.

**The grouping rule.** A word's phones are the run of `phones`-tier intervals ending at
or before the word's own end time. The two tiers share their boundaries exactly, and
the derived interval **keeps the word interval's own start and end times**, which is
what makes the denominator comparable: summing durations over this tier gives
bit-identical speech time to the `words` tier (dev-clean 15,845.690 s, to the last
digit), so `wordphn` and `word` divide by exactly the same seconds. Phones are
destressed, matching the `syllables` tier and `ArpabetTokenizer`, so the backoff
inventory is the same 39 symbols. Verified over all 281,239 training and 2,703
dev-clean TextGrids with zero mismatches; the raw type count it produces for the words
themselves, 89,048, reproduces "Data" above exactly. `word_phone_intervals` asserts
both directions of coverage rather than trusting them. The longest unit is a genuine
64-phone LibriSpeech compound word, not a broken alignment.

**The tokeniser is `SyllableTokenizer`, subclassed and not reimplemented.** The token
format is the one that class already handles, and so is the phone backoff, so
`WordPhoneTokenizer` adds no logic. The fallback marker matters here if anything more
than for syllables: every single-phone word ("a" as `AH`, "I" as `AY`) is both a
whole-unit entry and a phone that a longer word's spell-out emits.

**Vocabulary.** 84,163 raw types over 960h, fewer than word's 89,048, and 56,255 at
min-count 2 against word's 57,977, giving a tokeniser vocabulary of 56,296. On
dev-clean it is also the *less* out-of-vocabulary unit of the two: 441 units back off
(0.811%) against word's 486 (0.893%), and because a phone spell-out is shorter than a
character one those produce 3,031 pieces against word's 3,793.

**What the comparison with word does and does not isolate.** Sequence length is
identical: one unit per word interval, so both units see the same 54,402 dev-clean
units, and any bit-rate difference is about unit identity rather than how many units
there are. But the mapping is many-to-many in both directions, and the two directions
pull opposite ways:

- **Homophones collapse**, which should help: on dev-clean 203 realisations cover more
  than one word (`T-UW` is to/too/two, `ER` is are/or, `DH-IY` is the/thee).
- **Pronunciation variants split**, which should hurt: 248 dev-clean word types have
  more than one realisation ("the" as `DH-AH` 2,929 times and `DH-IY` 519 times, "and"
  as `AE-N-D` 1,326 and `AH-N-D` 589). The model is charged for predicting *which*
  variant the speaker realised, which `word` is never charged for.

So part of the measured `wordphn` deficit, 0.70 bits/s against word at seed 0 and 0.58
on five-seed means (`results.md`, "Wordphn"), is the cost of pronunciation realisation
rather than of lexical content,
and the unit is not a drop-in "is a word a good unit" control.
`ngram/librispeech_stats.py` reports both directions for the full 960h set as well.

### The wordphn-BPE unit

`wordphn-BPE-k` is the collocation variant of `wordphn`, standing to it exactly as
`word-BPE-k` stands to `word`: `ngram/bpe.py`'s `BPETokeniser` merges the most frequent
adjacent pair k times, with wordphn units as the atomic symbols. A unit is therefore a
collocation written out as phones, "of the" being `AH-V DH-AH`. The merges are fitted
on the wordphn stream itself, not inherited from the word-level run, so each unit's
merges are learned on that unit's own stream.

**The two separators do not collide**: a space joins the words inside a merged unit and
"-" joins the phones inside a word, and a realisation never contains a space, so
`BPETokeniser` needs no modification. `WordPhoneBpeTokenizer` likewise only overrides
`WordBpeTokenizer`'s `SUBUNITS`/`split_subunits`; the three-level cascade is shared,
and the middle level is the reason it must be: the cascade runs one notch finer here,
unit -> wordphn -> phones.

Three merge counts are run, 200/1000/5000, matching word-BPE's own tested values.

**The vocabulary is essentially nested.** At `min_count = 2` it is the k merged units
plus `wordphn`'s own min-count-2 vocabulary, which at 200 and 1000 merges is
*bit-identical* as a set, verified by set comparison: merging neither gains nor loses a
singleton type, so the comparison against `wordphn` varies only the merged units. At
5000 merges two singletons slip below the threshold (61,253 rather than 61,255), the
same effect word-BPE-5000 shows at four types.

| merges | vocabulary | tokeniser | training tokens |
|---|---:|---:|---:|
| none (`wordphn`) | 56,255 | 56,296 | 10,334,163 |
| 200 | 56,455 | 56,496 | 9,489,139 |
| 1000 | 57,255 | 57,296 | 8,902,852 |
| 5000 | 61,253 | 61,294 | 8,227,336 |

**Merge slots are spent on pronunciation variants.** This is the realisation-specific
cost, and it is measurable. Mapping each realisation back to its most frequent word
over 960h and collapsing the merges to word-level collocations, roughly one slot in ten
is a variant of a collocation already merged, and the share **grows with merge depth**:

| merges | distinct collocations | duplicate slots | wasted |
|---|---:|---:|---:|
| 200 | 183 | 17 | 8.5% |
| 1000 | 907 | 93 | 9.3% |
| 5000 | 4,498 | 502 | 10.0% |

The same statistic is 0% at every merge count for word-BPE, necessarily, since a
word-level merge cannot have a pronunciation variant; that is the control confirming
the measure means what it says. At 200 merges "to the" takes three slots (`T-AH DH-AH`
rank 2, `T-AH DH-IY` rank 60, `T-UW DH-AH` rank 148) and "and the" three, and of the 38
merges ending in a realisation of "the", six are `DH-IY` forms whose `DH-AH` sibling is
also merged. The counterweight is that pair counts pool across homophones, so a merge
candidate can be more frequent than any single spelling of it is. The ranking shifts
visibly against the word level: "and the" is word-BPE-200's rank-3 merge but only rank
8 here, its count split across `AE-N-D` and `AH-N-D`.

The word-to-realisation mapping used for those counts is approximate, since it resolves
each realisation to one word and homophones therefore collapse; it is a descriptive
statistic, not an input to any model. It is also not the whole story: `results.md`,
"With collocation merging", finds this duplication the right shape but too small to
account for how sharply the unit degrades at 5000 merges.

### The subwordphnx-BPE unit

`subwordphnx-BPE-v` is `subword-BPE` with the alphabet swapped: standard subword BPE
run over **phones** instead of characters, with the same HuggingFace `BpeTrainer`, at
vocabulary size `v`, run at 200, 2000 and 5000. So it is the phonetic counterpart of
the project's one non-lexical spelling unit, and it completes the realisation ladder:
phone, subwordphnx-BPE, wordphn, wordphn-BPE-k, against subword-BPE, word, word-BPE-k
on the spelling side.

**A phone has to be made atomic first.** A BPE alphabet is single characters, and
ARPABET symbols are one to three characters long, so training on the bare strings would
spend its early merges turning `D` and `H` into `DH` and would never merge two phones
with each other. `llm/tokenisers.py` therefore maps each of the 39 phones to its own
private-use code point, a bijection applied inside `SubwordPhoneXBpeTokenizer` so that
nothing upstream or downstream sees it; `as_phones` writes any piece back out as
"-"-joined phones for inspection. At vocabulary 2000 the saved vocabulary is 3 specials
plus 39 single phones plus 1,958 merged pieces.

**Merges are confined to a word**, which is what the `x` in the name marks. A
`WhitespaceSplit` pre-tokeniser over one BPE word per spoken word does it, so no piece
ever straddles a boundary. That makes this the faithful counterpart of `subword-BPE`,
which is word-confined too by virtue of the same pre-tokeniser. It reads `WORDPHN_TIER`
rather than the `phones` tier, since that is where the word grouping lives; the phones
are the same phones and the seconds are bit-identical, so bit rates stay paired across
every unit.

**Not to be confused with `wordphn-BPE`.** The two differ in scale: `wordphn-BPE`
merges whole realised words into collocations, starting from a 56,255-unit lexicon,
while this unit merges individual phones into pieces that are usually shorter than a
word, starting from 39 symbols. One asks whether multi-word chunks pay, the other
whether sub-word chunks do.

**Stress digits are stripped inside the tokeniser**, exactly as `ArpabetTokenizer` does
it, rather than through `read_tier`'s `token_transform`. Validation reads interval text
verbatim, so this is the only place that cannot let training and dev disagree. For this
unit it is a no-op in practice, since `data_common.word_phone_intervals` has already
destressed the tier it reads.

**No fallback and no augmentation.** Every phone is in the alphabet by construction, so
nothing can be out of vocabulary, `<unk>` is unreachable, and there is no spell-out rule
to mark, exactly as for `phone` and `subword-BPE`. That also makes it one of the two
units whose vocabulary carries no minimum-count threshold, since it has no types to
threshold.

**It is not information-equivalent to `phone`.** Decoding recovers the phone string,
but the segmentation *depended* on where the words divided, so the piece sequence
carries that information back: `DH-AH K-AE-T` ("the cat") and a hypothetical one-word
`DH-AH-K AE-T` tokenise differently from the same phones. How much it carries grows
with the vocabulary, since the share of dev-clean word tokens emitted as a *single*
piece does: 47.3% at 200, 75.8% at 2000 and 85.8% at 5000. Any comparison of this unit
against `phone` therefore mixes better modelling with more information, and the mixture
is not constant along the sweep; `results.md`, "Subwordphnx-BPE", is where that is
unpicked.

**Vocabulary size makes it an interpolation between `phone` and `wordphn`.** At the
bottom its pieces are single phones and at the top they are whole words, so the unit
slides continuously between two units the project already has. That is the reason for
sweeping it, and the reason it is the only unit here whose vocabulary size is swept at
all. One consequence to keep in view: a smaller vocabulary merges less and so leaves a
longer token stream, which at the fixed 20-epoch target means a larger step budget
(5000 steps at vocab 200 against 2800 at 5000), so the sweep varies optimisation steps
alongside vocabulary. That is the standing protocol, and word-BPE's own merge counts
vary in the same way. A second consequence, found once the middle size was seeded: the
whole sweep spans 0.52 bits/s while its middle point has a seed SD of 0.279, so no step
that passes through vocabulary 2000 can be separated from training noise, although a
second seed at each end (2026-10-12/13) does separate the two ends from each other and
from phone. See "Uncertainty". The n-gram sweeps at the same three vocabularies, run
2026-10-12, recover the same ordering with every step clear of zero and no seed floor
to clear, so the direction is established even though the neural magnitudes are not.
At vocabulary 5000 the n-gram unit is indistinguishable from wordphn, the
interpolation's upper endpoint measured directly; the neural unit is not, staying 0.49
bits/s above wordphn at 5.3x the floor, so the two estimators agree on the direction of
this sweep and not on where it ends.

**All three vocabularies have an n-gram counterpart**,
`ngram/ngram_subwordphnx_bpe_sweep.py --vocab-size <v>`, built on `bpe.py`'s
`SubwordPhoneXBPETokeniser`
rather than on `llm/`'s HuggingFace path, so the two stages stay independent in the way
`ngram/` and `llm/` are everywhere else. The two implementations were checked against
each other on identical data before the first sweep was run: at vocabulary 2000, 1,957
of 1,958 merged pieces
agree, one tie-break differs at the last merge, and the dev-clean segmentation is
identical on 99.98% of word tokens, giving 74,644 pieces against 74,641. Vocabulary
5000 gives 65,150 against 65,155. That is close
enough for the two stages to be scoring the same unit, which is what the
cross-estimator comparison needs. 2000 was swept first, on 2026-10-09, for that
comparison; 200 and 5000 followed on 2026-10-12, to put an estimator with no training
noise under the vocabulary sweep.

### Vocabulary rules

Built from LibriSpeech 960h **training data only**, never train plus test. Using
test data to build a vocabulary flatters large-unit tokenisations most, since they
have the most unseen items.

A minimum-count threshold of 2 (cutting hapax legomena) is used for word, wordphn and
syllable, rather than a type-percentage cutoff: word frequency is heavily Zipfian,
so keeping the top 95% of word types still covers 99.953% of training tokens. For
word, the threshold cuts 31,071 types (34.9% of the vocabulary) but only 0.33% of
training tokens; for wordphn, 27,908 types (33.2%); for syllable, 2,270 types
(16.6%) and 0.017% of tokens.

**The threshold was not applied to word-BPE until 2026-09-07.** It is now the
default there too, so all three units that need a vocabulary rule use the same
one; see "Open issues" for what that changed.

`ngram/` applies no threshold at any unit. Each stage is internally consistent,
which is what a within-stage cross-unit comparison needs; the cross-stage
difference is documented rather than eliminated, since threading a threshold into
`AbsoluteDiscountingLM.fit` would be invasive surgery on the shared estimator.

### Random augmentation

Independently for 2% of all occurrences in the *training* data only (seed 0),
regardless of whether the item is in vocabulary, the token is replaced by its
spelled-out pieces, **already marked with `FALLBACK_MARKER`** so they encode to the
same fallback ids the deterministic rule produces at test time. Purely
frequency-based spelling gives very little exposure to the fallback pathway, so the
model would rarely practise it; syllable's natural OOV
rate (28 of 75,910 dev-clean tokens, 0.037%) is otherwise far too low to expect the
model to have learned anything about it.

**The marking is load-bearing, and was missing until 2026-09-21.** `_encode_plus`
keeps anything already in the unit vocabulary whole, and every single letter is
itself a LibriSpeech word, so bare augmented characters resolved to whole-word ids:
the augmentation produced no fallback tokens at all for word and word-BPE, and
train/test-inconsistent hybrids for syllable, where 19 of 39 phones are also whole
syllables. Each tokeniser's `_encode_plus` now passes an already-marked piece
straight through; without that it would be spelled out again and double-marked.
`WordBpeTokenizer` additionally segments *around* marked pieces so they bypass BPE
entirely, which also matches test time, where an unseen word never merges with a
neighbour. The `~`-marked share of the word training stream went from 2.40% to
10.08% as a result.

Fixed once when the mmap is built, not re-rolled per epoch. Never applied at
validation or evaluation time, where only the deterministic rule runs.

## Neural training setup

**What OPT is.** OPT (Open Pre-trained Transformer) is Meta AI's openly released
suite of decoder-only autoregressive language models, spanning 125M to 175B
parameters, built to reproduce GPT-3-class models under an open licence with a
public training logbook (Zhang et al., 2022, arXiv:2205.01068). Each is a standard
causal transformer: stacked blocks of masked multi-head self-attention and a
feed-forward layer, with pre-layer-norm, ReLU activations, learned position
embeddings and tied input and output embeddings. Its smallest member, OPT-125M, has
12 layers, hidden size 768, 12 heads and a 2048-token context.

**This project uses the architecture, not the released weights.** `llm/trainer.py`
builds `OPTForCausalLM` from an `OPTConfig` (HuggingFace `transformers`), randomly
initialised, so every unit is trained from scratch on LibriSpeech alone. That is
what makes it a controlled estimator, and what separates it from `qwen/`, whose
pretraining data is unmeasurable. The configuration shrinks OPT-125M in every
dimension -- 6 layers instead of 12, hidden size 512 instead of 768, 8 heads
instead of 12, an FFN of 2048 instead of 3072 -- while keeping OPT's 2048-token
context, giving a 19,964,928-parameter body, hence **OPT-20M** in `results.md`. It
is otherwise reused as-is from the scaffold, identical across every unit per the
"one fixed model class" principle. Only `vocab_size`, and therefore the total
parameter count, varies, and that is data-dependent rather than a free choice.

| setting | value |
|---|---|
| hidden size / layers / heads | 512 / 6 / 8 |
| ffn dim | 2048 |
| context window | 2048 |
| effective batch (tokens) | 81,920 |
| effective batch (sequences) | 40 |
| epochs | 20 |
| `lr_max` | 1e-3 |
| schedule | linear warmup 8% of steps, then decay to 2e-5 |
| optimiser | AdamW, betas (0.9, 0.98), weight decay 0.01 |

**20 epochs at `lr_max=1e-3` is the canonical setup for every unit**, settled by
sweep rather than assumption. A learning-rate sweep found 1e-3 better than 5e-4 on
phone, syllable, subword-bpe and word, losing only marginally on two word-BPE
variants. An epoch sweep at 30 and 40 epochs found 20 best or tied-best at every
merge count, with larger vocabularies degrading as training lengthened. Numbers in
`results.md`.

**"Epoch" here means a step budget, not a pass over the data.**
`TokenizedUnitsChunkedMMapDataset.__getitem__` ignores its index and samples a
random window **with replacement**, and `steps_per_epoch` is defined as
`num_tokens / 81920`. `max_steps` and `warmup_steps` are derived at runtime from
each unit's own `train.json` token count and the `--epochs` flag, so changing the
mmap or the epoch count cannot silently desynchronise them.

**No unit trains for exactly 20 epochs.** `max_steps` is rounded to the nearest
100, so what each unit actually receives is `max_steps / steps_per_epoch`:

| unit | training tokens | steps/epoch | max_steps | effective epochs |
|---|---:|---:|---:|---:|
| phone | 33,745,655 | 411.9 | 8200 | 19.91 |
| subword-bpe | 13,362,472 | 163.1 | 3300 | 20.23 |
| subwordphnx-bpe-200 | 20,427,217 | 249.4 | 5000 | 20.05 |
| subwordphnx-bpe-2000 | 13,070,150 | 159.5 | 3200 | 20.06 |
| subwordphnx-bpe-5000 | 11,457,890 | 139.9 | 2800 | 20.02 |
| word | 10,527,856 | 128.5 | 2600 | 20.23 |
| syllable | 13,722,601 | 167.5 | 3400 | 20.30 |
| word-BPE-200 | 9,593,621 | 117.1 | 2300 | 19.64 |
| word-BPE-1000 | 8,991,501 | 109.8 | 2200 | 20.04 |
| word-BPE-5000 | 8,310,620 | 101.4 | 2000 | 19.71 |
| wordphn | 10,334,163 | 126.1 | 2500 | 19.82 |
| wordphn-BPE-200 | 9,489,139 | 115.8 | 2300 | 19.86 |
| wordphn-BPE-1000 | 8,902,852 | 108.7 | 2200 | 20.24 |
| wordphn-BPE-5000 | 8,227,336 | 100.4 | 2000 | 19.91 |

The spread is **19.64 to 20.30** over the fourteen units, about 3% end to end, and it
is an artefact of the rounding rather than a choice: the units with the fewest tokens
have the coarsest grid, since 100 steps is a larger fraction of an epoch for them.
Token counts differ by unit because a tokenisation changes how many tokens the
same speech becomes, and they shift slightly whenever the training stream is
regenerated, so these are the current figures rather than fixed constants.

Whether 3% matters is answerable from the epoch sweep: 20 against 30 epochs, a 50%
increase, moved word-BPE by at most 0.60 bits/s and often less, so a 3% difference
is far below the seed noise floor and cannot account for any gap in `results.md`.
Equalising it would mean dropping the round-number `max_steps`, which costs more in
legibility than it buys in precision.

**GPU feasibility.** On this machine's RTX 3070 (8GB), `batch_size=10` with
`accumulation_steps=4` fits for phone, syllable, and the four units whose vocabulary is
a few thousand at most: subword-bpe and the three subwordphnx units. Word, wordphn and
both BPE families above them need `batch_size=2` with `accumulation_steps=20` for the
same effective batch: the
dominant memory cost is the softmax in `cross_entropy` over a ~58k-94k vocabulary,
independent of model size. Verified empirically at 200 merges, where
`batch_size=3` still OOMs and `batch_size=2` peaks at 5.29GB.

**Storage.** The flat training stream is a `uint16` mmap for phone, subword-bpe, all
three subwordphnx units, word, wordphn and syllable, and `uint32` for the two BPE-merge
families, word-BPE and wordphn-BPE. That was necessary when word-BPE
carried ~89,248 units, over `uint16`'s 65,535 maximum; `--min-count 2` brought it
to 58,206-63,002, so `uint16` would now fit, but the existing mmaps and
checkpoints are `uint32` and changing it would mean regenerating and retraining
for no gain; wordphn-BPE was given `uint32` to match its family rather than for
need. The guard is asserted both in each `*_prepare_mmap.py` and again
inside `write_train_mmap`.

## Evaluation protocol

Every model is scored on the full dev-clean set, teacher-forced, with padding
excluded via `ignore_index`.

**Every utterance carries a real end-of-utterance token.** `bos_token_id` is reused
as the boundary marker, matching Qwen2.5's own bos-equals-eos convention. Each
utterance is encoded as `[bos] + content + [bos]`, and `collate` splits
`src = tokens[:-1]`, `tgt = tokens[1:]`, so the trailing boundary token is a scored
prediction target exactly once per utterance while the **leading one falls in
`src` only** and is never scored, playing the same role `<s>` plays in `ngram/`.
Every tokeniser prepends it (`add_bos_token=True`), so this needs no handling
in `TokenizedUnitsUtteranceDataset`, which only appends the trailing one.
Verified: encoded content tokens plus 2,703 boundaries reconciles exactly with
every reported token count, for every unit.

The flat training stream keeps each utterance's leading `<bos>` too, so the model is
*trained* with genuine, densely occurring utterance boundaries rather than meeting
them first at evaluation. Before this was fixed the model was never trained to
expect a boundary at a genuine utterance end at all, and word-BPE turned out to be
by far the most sensitive to the correction.

**Known asymmetry**: training sees cross-utterance windows of a flat stream, while
evaluation resets context per utterance. Bit rate is robust to this, being a
duration ratio.

**Model selection caveat**: validation during training, `-best.pt` selection, and
the reported bit rate all use dev-clean. For a final write-up, numbers would be
better reported on test-clean, which remains untouched.

`llm/eval_bitrate.py` asserts that a checkpoint's `pad_token_id` and `vocab_size`
match the current tokeniser, so scoring a checkpoint trained under a different
vocabulary fails with an explicit message rather than an opaque device-side CUDA
"index out of bounds" assert. Verified separately that no tokeniser emits an
out-of-range id on dev-clean, so that class of bug is genuinely gone rather than
merely guarded.

**No sequence approaches the 2048-position limit.** The longest dev-clean source
sequence is 372 tokens, at the phone unit; every other unit is shorter. So no
utterance is silently truncated at evaluation.

### The Qwen side experiment and its primer

`qwen/` scores dev-clean with a pretrained Qwen2.5 rather than anything trained
here, as an extreme point on the (unit, model class, data budget) triple. It is
consistent with the rest on every axis that defines the metric, and inconsistent
on one that does not, which is worth stating explicitly because the numbers sit in
the same table as everything else.

**Consistent.** Same dev-clean set (2,703 utterances) and the same 15,845.690 s
denominator. Same boundary convention: each utterance is bracketed as
`[sep_id, *utt_ids, sep_id]` and scored on `targets = new_tensor[:, 1:]`, so the
trailing end-of-utterance token is a scored target while the leading one is
context only, structurally identical to `collate`'s split in `llm/` and to
`</s>` in `ngram/`. Verified: 59,848 content tokens plus 2,703 boundaries is
exactly the 62,551 scored targets reported. Qwen's byte-level BPE has no
out-of-vocabulary case, so nothing goes unpaid and none of this project's
spell-out machinery is needed.

**The primer, and why it is not fully comparable.** Scored cold, a pretrained LM
is penalised for LibriSpeech's transcript register (lowercase, unpunctuated)
rather than for the content, and the predecessor project found the penalty is
*larger* for bigger models, enough to swamp the effect being measured and even
flip the 1.5B-versus-3B ordering. So a short primer in the same register, drawn
from train-clean-100, is prepended to every utterance as shared context. It is
never itself a prediction target, so the count of scored positions per utterance
is unchanged, and it is drawn from training data rather than dev-clean so the full
dev-clean set stays the scored set.

**But no other unit in this project gets extra context.** Qwen's numbers are
therefore conditioned on more than the OPT-20M ones are, and the gap between them
is not purely a model-class effect. Treat the primer as part of the estimator
being compared, not as a neutral preprocessing step.

**The larger caveat is the pretraining data.** Qwen has seen an unmeasurable
quantity of text, so this is not a held-out estimate in the sense the rest of the
project means. It answers "how much does the estimator matter", not "what is the
true information rate of English", which is why it sits outside the controlled
OPT-20M comparison rather than as another row in it.

Perplexity here divides by a boundary-inclusive token count, matching `llm/` but
not `ngram/`; see "Scope and metrics". Bit rate is unaffected.

## Standing design principles

Carried over from the predecessor project (`docs/assets/2026-08-30/notes.md` §3);
follow these rather than re-deriving them.

- **One fixed model class across every tokenisation** compared in a given
  experiment.
- **Every model assigns probability to whole sentences over an unbounded token
  set**, so no utterance is ever skipped and comparisons stay paired on the same
  evaluation set.
- **Vocabulary built from training data only**, never train plus test.
- **Tokenisers are factories re-fit per fold or split**, not fit once and reused,
  to avoid leakage across folds.
- **Held-out cross-entropy only**, never in-sample or plug-in entropy.
- **Shuffle a background corpus before sub-sampling** fractions of it, so corpus
  size is not confounded with speaker or chapter order.

## Open issues

Three things, in rough order of how much they would change a conclusion.

**dev-clean serves as both the model-selection and the reporting set.** For a final
write-up the numbers would be regenerated on test-clean with dev-clean kept for
selection; the units' ordering is not expected to move, but nothing here establishes
that.

**Two of the three subwordphnx units have two seeds rather than five**, so their noise
floor rests on one degree of freedom, and the middle size's own floor leaves every step
through it unsettled. See "Uncertainty".

**A tokeniser agreement that is not enforced.** `ngram/` and `llm/` learn word-BPE
merges from the same sentences with the same class, but feed them in different
orders (`ngram/` shuffled, `llm/` in filesystem order), and `BPETokeniser.learn`
breaks count ties by insertion order. Measured impact is near zero: at 200 merges
the merge sets are identical and dev-clean tokenisation is bit-identical; at 1000
merges 999 of 1000 merges are shared and 0.37% of utterances differ. But nothing in
the code enforces this, and it has already begun to break down at 1000 merges.

Deliberately not fixed: making tie-breaking order-independent is a robustness
improvement, not a correctness one, and it would change learned merges and so require
re-running both stages. Left as a known, measured property instead. Revisit if the two
stages' tokenisations ever need to be provably identical rather than empirically
near-identical. The same unenforced agreement covers `wordphn-BPE`, where it was never
measured, so the word-BPE figures are the only evidence for how far the two stages can
drift; pooling across homophones makes ties no rarer, so there is no reason to expect
it to behave better. The subwordphnx-BPE pair was measured, and agrees to 99.98% of
word tokens; see "The subwordphnx-BPE unit".
