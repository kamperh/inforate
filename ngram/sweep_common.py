"""
Shared building blocks for the seven ngram/ngram_*_sweep.py scripts: word,
syllable, phone, bpe, wordphn, wordphn_bpe and subwordphnx_bpe.

Each of those scripts pools and shuffles LibriSpeech 960h once (fixed seed),
sweeps 25/50/75/100% training fractions times n-gram orders 1-5, and
evaluates bit rate and perplexity on dev-clean, reusing the same estimator
throughout. What differs between them is only the tokenisation (which tier, and
for BPE, the learned merges) and `spelling_of` (how an out-of-vocabulary token
decomposes for the spell-out model). Everything else is unit-agnostic and lives
here.

Each script still owns its own `spelling_of`, `main()` orchestration and
`write_report` (titles/descriptions differ enough per unit that collapsing
those too would cost more readability than it saves).
"""

import random
from pathlib import Path

import tgt

from librispeech_stats import word_phone_units
from ngram_lm import AbsoluteDiscountingLM, bitrate, build_spelling_lm

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "librispeech-alignments"
TRAIN_SPLITS = ["train-clean-100", "train-clean-360", "train-other-500"]
DEV_SPLIT = "dev-clean"
FRACTIONS = [0.25, 0.5, 0.75, 1.0]
ORDERS = [1, 2, 3, 4, 5]
SEED = 0
# Not a real tier name. Accepted by `load_utterances` wherever one is, so the wordphn
# sweeps look like every other sweep; see `load_wordphn_utterances`
WORDPHN_TIER = "wordphones"


def load_utterances(splits, tier_name, token_transform=None):
    """{utterance key: (sentence, speech-only duration in seconds)} for every
    TextGrid under `splits`, reconstructing each sentence from the tier named
    `tier_name` (tgt already drops silence, so summing interval durations
    gives speech-only time; see ngram/librispeech_stats.py). `token_transform`,
    if given, is applied to each interval's text before joining (e.g.
    stripping stress digits for phones).

    `WORDPHN_TIER` is accepted here too, although no TextGrid carries it, so that
    the wordphn sweeps need no special case of their own.
    """
    if tier_name == WORDPHN_TIER:
        assert token_transform is None, "wordphn units are destressed as they are built"
        return load_wordphn_utterances(splits)
    transform = token_transform or (lambda t: t)
    utterances = {}
    for split in splits:
        for path in (DATA_ROOT / split).rglob("*.TextGrid"):
            tier = tgt.read_textgrid(path).get_tier_by_name(tier_name)
            tokens = [transform(iv.text) for iv in tier]
            duration = sum(iv.duration() for iv in tier)
            utterances[path.stem] = (" ".join(tokens), duration)
    return utterances


def load_wordphn_utterances(splits):
    """`load_utterances`, but for the derived `wordphn` unit: one token per
    `words`-tier interval, written as that word's own destressed phones joined by
    "-" ("abbey" is `AE-B-IY`).

    No TextGrid carries such a tier, so it is derived from the `words` and `phones`
    tiers by `librispeech_stats.word_phone_units`, which applies the same grouping
    rule `llm/data_common.word_phone_intervals` uses in the neural stage. Durations
    come from the `words` tier untouched, so this unit's denominator is bit-identical
    to the word sweep's and the two bit rates are directly comparable.
    """
    utterances = {}
    for split in splits:
        for path in (DATA_ROOT / split).rglob("*.TextGrid"):
            textgrid = tgt.read_textgrid(path)
            tokens = [realisation for _, realisation in word_phone_units(textgrid)]
            duration = sum(
                iv.duration() for iv in textgrid.get_tier_by_name("words")
            )
            utterances[path.stem] = (" ".join(tokens), duration)
    return utterances


def pooled_train_keys(train_utterances, seed=SEED):
    """Shuffle all training keys once with a fixed seed; the four fractions
    are nested prefixes of this single order.
    """
    keys = sorted(train_utterances)  # deterministic base order before shuffling
    random.Random(seed).shuffle(keys)
    return keys


def fit_and_score(order, train_token_seqs, dev_token_seqs, spelling_of, spelling_lm):
    """Fit an order-`order` LM on `train_token_seqs` (already tokenised),
    score every utterance in `dev_token_seqs` ({key: (tokens, duration)},
    already tokenised with the same scheme) against it and the precomputed
    `spelling_lm` (see `run_order_sweep`: built once per fraction, not here,
    since it depends only on the training vocabulary, which doesn't vary
    with `order`). Returns:

        (results, discounts, spelling_discounts)

    `results` is {key: (bits, duration)}. `discounts` is `lm.discounts[1:]`
    (one data-driven discount per order 1..`order`) and `spelling_discounts`
    is `spelling_lm.discounts[1:]` (one per order 1..3) for the spell-out
    model. These are two different models, so any "falling back to D=0.75"
    message is tagged with the class name (`[SpellingLM]` vs
    `[AbsoluteDiscountingLM]`) to show which it came from.
    """
    lm = AbsoluteDiscountingLM(order).fit(train_token_seqs)

    results = {}
    for key, (tokens, duration) in dev_token_seqs.items():
        bits = lm.sentence_bits(
            tokens, spelling_lm=spelling_lm, spelling_of=spelling_of
        )
        results[key] = (bits, duration)
    return results, lm.discounts[1:], spelling_lm.discounts[1:]


def perplexity(results, n_tokens):
    """2^(total bits / total tokens): the same quantity as bit rate,
    normalised by token count instead of duration. Not comparable across
    units (or, for BPE, across fractions, since tokenisation itself
    differs); bit rate is the unit-agnostic quantity for cross-unit and
    cross-fraction comparison.
    """
    total_bits = sum(bits for bits, _ in results.values())
    return 2 ** (total_bits / n_tokens)


def tokenize_dev(dev_utterances):
    """{key: (tokens, duration)} from {key: (sentence, duration)}, splitting
    each sentence on whitespace, plus the total token count. Shared by the
    word/syllable/phone sweeps' plain-split tokenisation (the BPE sweep
    tokenises via a learned tokeniser instead, and doesn't use this).
    """
    dev_token_seqs = {
        key: (sentence.split(), duration)
        for key, (sentence, duration) in dev_utterances.items()
    }
    n_dev_tokens = sum(len(toks) for toks, _ in dev_token_seqs.values())
    return dev_token_seqs, n_dev_tokens


def tokenize_train_subset(train_utterances, subset_keys):
    """Token sequences (whitespace split) for one training fraction, in
    `subset_keys` order. Shared by the word/syllable/phone sweeps.
    """
    return [train_utterances[k][0].split() for k in subset_keys]


def run_order_sweep(
    fraction,
    train_token_seqs,
    dev_token_seqs,
    spelling_of,
    n_dev_tokens,
    unit_label,
    bitrate_table,
    ppl_table,
    discount_table,
):
    """Fit and score every order in `ORDERS` for one training fraction,
    filling `bitrate_table`/`ppl_table`/`discount_table` (keyed by
    `(fraction, order)`) in place and printing per-order progress.
    `unit_label` names the discounts line (e.g. "word", "syllable", "phone",
    "BPE"). Shared inner loop of every ngram_{word,syllable,phone,bpe}_sweep.py's
    `main()`.

    The order-3 spelling model is built once here, not inside `fit_and_score`:
    it depends only on this fraction's training vocabulary (via
    `AbsoluteDiscountingLM.fit`'s own `vocab.update(tokens)`, independent of
    `order`), so refitting it per order would be redundant computation.

    """
    vocab = set()
    for tokens in train_token_seqs:
        vocab.update(tokens)
    spelling_lm = build_spelling_lm(vocab, spelling_of, order=3)

    for order in ORDERS:
        print(f"  order {order}:")
        results, discounts, spelling_discounts = fit_and_score(
            order, train_token_seqs, dev_token_seqs, spelling_of, spelling_lm
        )
        br = bitrate(results)
        ppl = perplexity(results, n_dev_tokens)
        bitrate_table[(fraction, order)] = br
        ppl_table[(fraction, order)] = ppl
        discount_table[(fraction, order)] = discounts[-1]  # top-order discount
        print(f"    {unit_label} discounts (order 1..{order}): {[round(d, 3) for d in discounts]}")
        print(f"    spelling discounts (order 1..3): {[round(d, 3) for d in spelling_discounts]}")
        print(f"    bit rate {br:.2f} bits/s, perplexity {ppl:.2f}")


def format_table(title, values, cell_fmt):
    header = "| fraction | " + " | ".join(f"order {o}" for o in ORDERS) + " |"
    sep = "|---|" + "---:|" * len(ORDERS)
    rows = []
    for fraction in FRACTIONS:
        cells = " | ".join(cell_fmt(values[(fraction, order)]) for order in ORDERS)
        rows.append(f"| {fraction:.0%} | {cells} |")
    return f"### {title}\n\n{header}\n{sep}\n" + "\n".join(rows) + "\n"
