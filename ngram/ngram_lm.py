#!/usr/bin/env python

"""
N-gram estimator: interpolated absolute discounting.

Ported from `docs/assets/2026-08-30/inforate.py` (`AbsoluteDiscountingLM`,
`CharacterLM`, `build_char_lm`, `bitrate`), generalised so the out-of-vocabulary
spell-out model is not hardcoded to "character": `CharacterLM` -> `SpellingLM`,
`build_char_lm` -> `build_spelling_lm`, `string_bits` -> `spelling_bits`. The
token-to-subunit decomposition moves out of the class into a caller-supplied
`spelling_of(token)` function, since a later stage backs syllables off to
phones rather than characters.

Also generalised: the discount is estimated per order from that fit's own
counts (Chen & Goodman, 1999), rather than fixed at a single value across
every order and unit.

Design, unchanged from the predecessor project (design.md, "The n-gram
estimator", carries the full equations):
- Every model is a proper distribution over whole sentences: sentence-final
  </s> is scored, and no token can receive zero probability, because unseen
  tokens back off to an <unk> event whose spelling is then paid for with a
  spelling-level model. Nothing is ever skipped.
- Smoothing is interpolated absolute discounting at every order, which does
  not depend on a vocabulary size taken from the test set.
- The reported quantity is bits per utterance divided by speech seconds.
"""

import math
from collections import Counter

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"


class AbsoluteDiscountingLM:
    """Interpolated absolute discounting n-gram model.

        P_k(w|h) = [max(c(hw) - D_k, 0) + D_k N1+(h) P_{k-1}(w|h')] / c(h)

    with the unigram level interpolated towards a uniform floor over the
    training vocabulary plus <unk>. Every conditional sums to one over that
    vocabulary, checked by `assert_normalised`.

    The discount D_k is estimated per order from that order's own training
    counts (Chen & Goodman, 1999), not fixed across orders; see
    `_estimate_discount`.
    """

    def __init__(self, order, discount=0.75):
        self.order = order
        self.discount = discount  # fallback only; see `_estimate_discount`

    def fit(self, sequences):
        """`sequences` is an iterable of token lists (no BOS/EOS)."""
        self.counts = [Counter() for _ in range(self.order + 1)]
        self.ctx_total = [Counter() for _ in range(self.order + 1)]
        self.ctx_types = [Counter() for _ in range(self.order + 1)]

        vocab = set()
        for tokens in sequences:
            padded = [BOS] * (self.order - 1) + list(tokens) + [EOS]
            vocab.update(tokens)
            for i in range(self.order - 1, len(padded)):
                for k in range(1, self.order + 1):
                    if i - k + 1 < 0:
                        continue
                    gram = tuple(padded[i - k + 1 : i + 1])
                    self.counts[k][gram] += 1

        for k in range(1, self.order + 1):
            for gram, c in self.counts[k].items():
                self.ctx_total[k][gram[:-1]] += c
                self.ctx_types[k][gram[:-1]] += 1

        self.discounts = [None] + [
            self._estimate_discount(k) for k in range(1, self.order + 1)
        ]

        vocab.add(EOS)
        self.vocab = vocab
        self.n_floor = len(vocab) + 1  # + <unk>
        self.total_unigrams = self.ctx_total[1][()]
        assert self.total_unigrams > 0
        return self

    def _estimate_discount(self, k):
        """Chen & Goodman's data-driven discount, D = n1/(n1 + 2*n2), where
        n1 and n2 are the number of order-k n-grams occurring exactly once
        and exactly twice in training (Chen & Goodman, 1999). Falls back to
        `self.discount` when there is no evidence to estimate from
        (n1 = n2 = 0), and is otherwise clamped to [0.1, 0.9] against
        near-0/near-1 estimates from small samples; the clamp is a pragmatic
        guard, not part of the formula.
        """
        n1 = sum(1 for c in self.counts[k].values() if c == 1)
        n2 = sum(1 for c in self.counts[k].values() if c == 2)
        if n1 + 2 * n2 == 0:
            print(
                f"  [{type(self).__name__}] order {k}: no singleton/doubleton "
                f"{k}-grams, falling back to D={self.discount}"
            )
            return self.discount
        return min(max(n1 / (n1 + 2 * n2), 0.1), 0.9)

    def _prob(self, token, context):
        """P(token | context) with |context| < order."""
        # Unigram level, interpolated towards the uniform floor
        D1 = self.discounts[1]
        c = self.counts[1].get((token,), 0)
        n_types = self.ctx_types[1][()]
        p = (max(c - D1, 0.0) + D1 * n_types / self.n_floor) / self.total_unigrams

        # Higher orders
        for k in range(2, len(context) + 2):
            Dk = self.discounts[k]
            h = tuple(context[-(k - 1) :])
            total = self.ctx_total[k].get(h, 0)
            if total == 0:
                continue  # unseen context: fall through with the lower-order p
            gram = h + (token,)
            c = self.counts[k].get(gram, 0)
            p = (max(c - Dk, 0.0) + Dk * self.ctx_types[k][h] * p) / total
        return p

    def assert_normalised(self, contexts, tol=1e-9):
        """Check that each conditional sums to one over vocab + <unk>."""
        for context in contexts:
            total = sum(self._prob(w, context) for w in self.vocab)
            total += self._prob(UNK, context)
            assert abs(total - 1.0) < tol, (context, total)

    def sentence_bits(self, tokens, spelling_lm=None, spelling_of=None):
        """Bits to encode the full sentence, including </s>.

        Unseen tokens are scored as <unk> and then spelled out via
        `spelling_lm.spelling_bits(spelling_of(token))`, so the model remains
        a distribution over *all* sentences and no utterance ever has to be
        discarded.
        """
        mapped = [t if t in self.vocab else UNK for t in tokens]
        padded = [BOS] * (self.order - 1) + mapped + [EOS]
        original = list(tokens) + [EOS]

        bits = 0.0
        for i in range(self.order - 1, len(padded)):
            context = padded[max(0, i - self.order + 1) : i]
            p = self._prob(padded[i], context)
            bits -= math.log2(p)
            if padded[i] == UNK:
                token = original[i - (self.order - 1)]
                bits += spelling_lm.spelling_bits(spelling_of(token))
        return bits


class SpellingLM(AbsoluteDiscountingLM):
    """Model used to spell out an out-of-vocabulary token.

    Fit over subunit sequences ("spellings") of the training vocabulary, e.g.
    characters for words, phones for syllables. Scores a final </s>, so it is a
    distribution over strings of any length, and bottoms out in its own uniform
    floor, so every spelling has non-zero probability with no further fallback.
    """

    def sentence_bits(self, tokens, spelling_lm=None, spelling_of=None):
        mapped = [t if t in self.vocab else UNK for t in tokens]
        padded = [BOS] * (self.order - 1) + mapped + [EOS]
        bits = 0.0
        for i in range(self.order - 1, len(padded)):
            context = padded[max(0, i - self.order + 1) : i]
            bits -= math.log2(self._prob(padded[i], context))
        return bits

    def spelling_bits(self, spelling):
        return self.sentence_bits(spelling)


def build_spelling_lm(token_types, spelling_of, order=3, discount=0.75):
    """Fit a `SpellingLM` on the spellings of every vocabulary token type."""
    return SpellingLM(order, discount).fit(
        [spelling_of(t) for t in token_types]
    )


def bitrate(results, keys=None):
    """Total bits over total speech seconds.

    `results` maps utterance key -> (bits, duration_seconds).
    """
    keys = list(results) if keys is None else keys
    bits = sum(results[k][0] for k in keys)
    duration = sum(results[k][1] for k in keys)
    return bits / duration


if __name__ == "__main__":
    # Toy sanity check: fit an order-2 word LM plus a character-level
    # spelling model, score a sentence with one out-of-vocabulary word, and
    # confirm every conditional distribution sums to one.
    train = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "sat", "on", "the", "rug"],
    ]
    lm = AbsoluteDiscountingLM(order=2).fit(train)
    lm.assert_normalised([(), ("the",), ("cat",)])
    print(f"word LM discounts (by order): {lm.discounts[1:]}")

    spelling_of = list  # word -> characters
    spelling_lm = build_spelling_lm(lm.vocab - {EOS}, spelling_of, order=3)
    spelling_lm.assert_normalised([(), ("c",), ("c", "a")])
    print(f"spelling LM discounts (by order): {spelling_lm.discounts[1:]}")

    test = ["the", "cat", "napped"]  # "napped" is out of vocabulary
    bits = lm.sentence_bits(
        test, spelling_lm=spelling_lm, spelling_of=spelling_of
    )
    print(f"bits for {test!r}: {bits:.3f}")
    print("assert_normalised passed for both models.")
