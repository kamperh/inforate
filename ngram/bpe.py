"""
Two BPE tokenisers, differing in what their atomic symbol is.

`BPETokeniser` merges over whole words, so a unit is a multi-word expression (e.g. "of
the") and never a subword piece. This is not subword BPE in the usual NLP sense; it is
the collocation tokeniser the word-BPE and wordphn-BPE sweeps use.

`SubwordPhoneXBPETokeniser` is subword BPE in the usual sense, with phones as the
atomic symbols and every merge confined to a single word, so a piece is a run of phones
inside one word. It is what `ngram_subwordphnx_bpe_sweep.py` uses, and the n-gram
counterpart of the neural stage's `SubwordPhoneXBpeTokenizer`.

The two are independent: one grows units upwards from words, the other grows them
upwards from phones, and they share only the `_apply_merge` helper and the merge-rank
bookkeeping in `_RankedMerges`.
"""

from collections import Counter, defaultdict


class _RankedMerges:
    """The merge-rank lookup both tokenisers encode with.

    BPE is greedy, so a merge's position in the learned list is its priority, and
    encoding applies the highest-priority applicable merge rather than working left to
    right. The dict is rebuilt only when the merge list has changed length, since
    `learn` appends to it one merge at a time.
    """

    @property
    def ranks(self):
        if getattr(self, "_ranks_for", None) != len(self.merges):
            self._ranks = {pair: i for i, pair in enumerate(self.merges)}
            self._ranks_for = len(self.merges)
        return self._ranks


class BPETokeniser(_RankedMerges):
    """Byte-pair encoding applied at the *word* level: repeatedly merge the
    most frequent adjacent pair of units into a single unit.
    """

    def __init__(self, n_merges):
        self.n_merges = n_merges
        self.merges = []
        self.name = f"bpe-{n_merges}"

    def learn(self, sentences, verbose=False):
        corpus = [[(w,) for w in s.split()] for s in sentences]

        pair_counts = Counter()
        pair_where = defaultdict(set)

        def add(i):
            toks = corpus[i]
            for j in range(len(toks) - 1):
                pair = (toks[j], toks[j + 1])
                pair_counts[pair] += 1
                pair_where[pair].add(i)

        def remove(i):
            toks = corpus[i]
            for j in range(len(toks) - 1):
                pair = (toks[j], toks[j + 1])
                pair_counts[pair] -= 1
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]

        for i in range(len(corpus)):
            add(i)

        while len(self.merges) < self.n_merges:
            if not pair_counts:
                break
            best, count = pair_counts.most_common(1)[0]
            if count < 2:
                break
            A, B = best
            self.merges.append(best)
            for i in list(pair_where[best]):
                toks = corpus[i]
                if not any(
                    toks[j] == A and toks[j + 1] == B for j in range(len(toks) - 1)
                ):
                    continue
                remove(i)
                corpus[i] = _apply_merge(toks, A, B)
                add(i)
            pair_where.pop(best, None)
            pair_counts.pop(best, None)
            if verbose and len(self.merges) % 2000 == 0:
                print("  merges:", len(self.merges), flush=True)
        return self

    def encode_units(self, sentence):
        """Return the merged units as tuples of words.

        Merges are applied in rank order rather than list order, which is
        the standard BPE encoding: applying one merge can expose a
        higher-priority pair, and rank order picks it up.
        """
        toks = [(w,) for w in sentence.split()]
        ranks = self.ranks
        if not ranks:
            return toks
        while len(toks) > 1:
            best_rank, best_i = None, None
            for i in range(len(toks) - 1):
                r = ranks.get((toks[i], toks[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i is None:
                break
            toks = (
                toks[:best_i]
                + [toks[best_i] + toks[best_i + 1]]
                + toks[best_i + 2 :]
            )
        return toks

    def encode(self, sentence):
        return [" ".join(t) for t in self.encode_units(sentence)]


def _apply_merge(toks, A, B):
    out, j, n = [], 0, len(toks)
    while j < n:
        if j < n - 1 and toks[j] == A and toks[j + 1] == B:
            out.append(A + B)
            j += 2
        else:
            out.append(toks[j])
            j += 1
    return out


class SubwordPhoneXBPETokeniser(_RankedMerges):
    """Subword BPE over phones, with every merge confined to a single word.

    The n-gram counterpart of the neural stage's `SubwordPhoneXBpeTokenizer`: the
    atomic symbols are phones rather than words, and a piece is a run of phones
    inside one word, written back out as "-"-joined phones ("the" is one piece,
    `DH-AH`). Input sentences are `WORDPHN_TIER` sentences, so a token is a word's
    own destressed phones ("abbey" is `AE-B-IY`).

    **Merges are learned per word type, weighted by that type's token count**, which
    is the standard formulation and what the neural stage's `BpeTrainer` does
    internally. Merge frequencies are over token occurrences either way, so the
    result is the same as holding 9.4M word tokens in memory, which this avoids.
    Confining merges to a word then falls out for free: there is nothing adjacent to
    a word's last phone.

    The encoded stream carries no word boundary of its own, exactly as in the neural
    stage. A piece sequence does imply where the words divide, which is the confound
    `design.md` records under "The subwordphnx-BPE unit"; it is a property of the
    unit, not of this implementation.
    """

    def __init__(self, n_merges):
        self.n_merges = n_merges
        self.merges = []
        self.name = f"subwordphnx-bpe-{n_merges}"
        self._encoded = {}  # Per word type; a sentence repeats types constantly

    def learn(self, sentences, verbose=False):
        """Learn up to `n_merges` merges from `sentences` (whitespace-separated
        realisations). Stops early if no pair occurs twice, as `BPETokeniser` does.
        """
        counts = Counter()
        for sentence in sentences:
            counts.update(sentence.split())
        corpus = {word: [(p,) for p in word.split("-")] for word in counts}

        pair_counts = Counter()
        pair_where = defaultdict(set)

        def add(word):
            syms, weight = corpus[word], counts[word]
            for j in range(len(syms) - 1):
                pair = (syms[j], syms[j + 1])
                pair_counts[pair] += weight
                pair_where[pair].add(word)

        def remove(word):
            syms, weight = corpus[word], counts[word]
            for j in range(len(syms) - 1):
                pair = (syms[j], syms[j + 1])
                pair_counts[pair] -= weight
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]

        for word in corpus:
            add(word)

        while len(self.merges) < self.n_merges:
            if not pair_counts:
                break
            best, count = pair_counts.most_common(1)[0]
            if count < 2:
                break
            A, B = best
            self.merges.append(best)
            for word in list(pair_where[best]):
                syms = corpus[word]
                if not any(
                    syms[j] == A and syms[j + 1] == B for j in range(len(syms) - 1)
                ):
                    continue
                remove(word)
                corpus[word] = _apply_merge(syms, A, B)
                add(word)
            pair_where.pop(best, None)
            pair_counts.pop(best, None)
            if verbose and len(self.merges) % 500 == 0:
                print("  merges:", len(self.merges), flush=True)
        return self

    def encode_word(self, word):
        """One word's phones as a list of pieces, memoised by word type."""
        if word in self._encoded:
            return self._encoded[word]
        syms = [(p,) for p in word.split("-")]
        ranks = self.ranks
        while len(syms) > 1:
            best_rank, best_i = None, None
            for i in range(len(syms) - 1):
                r = ranks.get((syms[i], syms[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i is None:
                break
            syms = (
                syms[:best_i]
                + [syms[best_i] + syms[best_i + 1]]
                + syms[best_i + 2 :]
            )
        pieces = ["-".join(t) for t in syms]
        self._encoded[word] = pieces
        return pieces

    def encode(self, sentence):
        """Every word's pieces, concatenated with no boundary symbol between words."""
        pieces = []
        for word in sentence.split():
            pieces.extend(self.encode_word(word))
        return pieces
