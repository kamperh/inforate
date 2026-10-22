"""Unit tokenisers in HuggingFace format, for the neural LM stage."""

import json
import sys
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

# Ngram/bpe.py's BPETokeniser is reused directly for the word-BPE unit, not duplicated
# (CLAUDE.md/data_common.py: "duplicating an algorithm this size would risk drift").
# ngram/ is a sibling directory, not on sys.path by default when running an llm/ script,
# so add it explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ngram"))
from bpe import BPETokeniser  # noqa: E402

PHONES = (
    "AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH "
    "K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH"
).split()

# Words-tier text matches ^[a-z']+$ (librispeech_stats.md cleanliness check), so these
# 27 symbols are exactly what a spelled-out word can ever decompose into.
CHARACTERS = list("abcdefghijklmnopqrstuvwxyz'")

# Marks a vocab entry as playing the *fallback* role (a character spelling out an OOV
# word, or a phone spelling out an OOV syllable), distinct from a whole-unit vocab entry
# that happens to share the same bare string (e.g. the word "a" vs. the character "a"
# used inside a spelled-out OOV word). Confirmed live: all 26 letters for word/word-BPE,
# and 19 single-phone-syllable collisions for syllable. Without this, both roles
# silently collapse onto the same vocab id. Safe from collision with any real vocab
# entry: words/BPE units match `^[a-z']+$` (see CHARACTERS above) and phone symbols are
# plain uppercase ARPABET strings -- neither can ever contain "~". Every fallback entry
# is marked uniformly, not just the ones that collide in the current vocabularies, so
# it stays correct if a vocabulary change makes a new entry start colliding.
FALLBACK_MARKER = "~"


def _fallback_vocab(subunits):
    """`FALLBACK_MARKER`-prefixed vocab entries for each subunit in `subunits`
    (characters or phones), so each gets its own id distinct from any whole-unit vocab
    entry. Shared by `WordTokenizer`/`SyllableTokenizer`/ `WordBpeTokenizer`; pair with
    `_MarkedFallbackTokenizer`'s `_decode`, which strips the marker back out.
    """
    return [FALLBACK_MARKER + s for s in subunits]


def _map_sequences(text, is_split_into_words, transform):
    """Apply `transform(seq) -> normed_seq` to one sequence or a batch of sequences
    uniformly, handling HuggingFace's single-vs-batch `text`/ `is_split_into_words`
    input-shape ambiguity once, shared by every tokeniser's `_encode_plus` below instead
    of each reimplementing it.
    """
    single = isinstance(text, str) or (
        is_split_into_words and text and isinstance(text[0], str)
    )
    seqs = [text] if single else text
    normed = [transform(seq) for seq in seqs]
    return normed[0] if single else normed


class _MarkedFallbackTokenizer(PreTrainedTokenizerFast):
    """Shared by `WordTokenizer`/`SyllableTokenizer`/`WordBpeTokenizer`: strips
    `FALLBACK_MARKER` back out of decoded text, so the marker used internally to keep
    fallback-role ids distinct from whole-unit ids (`_fallback_vocab`) is invisible in
    decoded/printed output: decoding a marked fallback token gives back the plain
    character or phone.
    """

    def _decode(self, *args, **kwargs):
        return super()._decode(*args, **kwargs).replace(FALLBACK_MARKER, "")


class ArpabetTokenizer(PreTrainedTokenizerFast):
    def __init__(self):
        vocab = {t: i for i, t in enumerate(["<bos>", "<pad>", *PHONES])}
        backend = Tokenizer(WordLevel(vocab=vocab))
        backend.pre_tokenizer = WhitespaceSplit()
        super().__init__(
            tokenizer_object=backend,
            bos_token="<bos>",
            pad_token="<pad>",
            unk_token=None,
            eos_token=None,
            add_bos_token=True,
            add_eos_token=False,
        )

    def _encode_plus(self, text, text_pair=None, is_split_into_words=False, **kwargs):
        """Strip ARPABET stress digits, reject unknown phones, then encode."""

        def transform(seq):
            phones = [
                p.rstrip("012") for p in (seq if is_split_into_words else seq.split())
            ]
            unknown = [p for p in phones if self._tokenizer.token_to_id(p) is None]
            if unknown:
                raise ValueError(
                    f"Unknown ARPABET symbol(s): {', '.join(map(repr, unknown))}"
                )
            return phones if is_split_into_words else " ".join(phones)

        text = _map_sequences(text, is_split_into_words, transform)
        return super()._encode_plus(
            text, text_pair=text_pair, is_split_into_words=is_split_into_words, **kwargs
        )


class SubwordBpeTokenizer(PreTrainedTokenizerFast):
    """Standard subword BPE: genuine sub-word merging via HuggingFace `tokenizers`' own
    `BpeTrainer`, distinct from `ngram/bpe.py`'s word-level collocation-merging "BPE" (a
    different, deliberately differently-named unit). Trained on the `words`-tier
    LibriSpeech 960h training sentences only, vocab size 2000.

    Unlike `ArpabetTokenizer`, the vocabulary here is learned, not fixed, so training
    and loading are two different paths. `train_new` learns fresh merges and returns the
    `tokenizers.Tokenizer` backend; call `.save(path)` on it to persist, then load with
    `SubwordBpeTokenizer(path)`. The mmap-prep script and the training script must use
    the *same* saved tokeniser file, never train two independently, or validation and
    training would silently use different vocabularies.
    """

    def __init__(self, tokenizer_file):
        backend = Tokenizer.from_file(str(tokenizer_file))
        super().__init__(
            tokenizer_object=backend,
            bos_token="<bos>",
            pad_token="<pad>",
            unk_token="<unk>",
            eos_token=None,
            add_bos_token=True,
            add_eos_token=False,
        )

    @staticmethod
    def train_new(sentences, vocab_size=2000):
        """Learn a fresh BPE vocabulary from `sentences` (plain text strings; training
        data only, per CLAUDE.md's "vocabulary from training data only" design
        principle). Returns the `tokenizers.Tokenizer` backend, not a
        `SubwordBpeTokenizer`; save it, then load with the constructor.
        """
        backend = Tokenizer(BPE(unk_token="<unk>"))
        backend.pre_tokenizer = WhitespaceSplit()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<bos>", "<pad>", "<unk>"],
        )
        backend.train_from_iterator(sentences, trainer=trainer)
        return backend


# One private-use code point per phone, so HuggingFace's own `BpeTrainer` can treat a
# phone as an atomic symbol. Its alphabet is single characters, and ARPABET symbols are
# one to three characters long, so training on the bare strings would merge "D" with "H"
# into "DH" instead of merging whole phones with each other. The bijection is an
# implementation detail that never reaches anything a human reads:
# `SubwordPhoneXBpeTokenizer.as_phones` writes a piece back out as phones.
_PHONE_TO_CHAR = {p: chr(0xE000 + i) for i, p in enumerate(PHONES)}
_CHAR_TO_PHONE = {c: p for p, c in _PHONE_TO_CHAR.items()}


class SubwordPhoneXBpeTokenizer(PreTrainedTokenizerFast):
    """The `subwordphnx-BPE` unit: standard subword BPE over *phones*, the phonetic
    counterpart of `SubwordBpeTokenizer`'s BPE over characters. Trained on LibriSpeech
    960h training utterances only, with the same `BpeTrainer`, so the only thing that
    differs from the character version is the alphabet: 39 phones in place of 27
    letters. Run at vocabulary 200, 2000, 5000 and 50000.

    Merges are **confined to a word**, so no piece ever crosses a word boundary; the
    `x` in the unit name marks that. This is the faithful phonetic analogue of
    `SubwordBpeTokenizer`, which is word-confined too by virtue of its
    `WhitespaceSplit` pre-tokeniser.

    It reads `WORDPHN_TIER` rather than the `phones` tier, because that is where the
    word grouping lives: a token is one word's destressed phones joined by "-", which
    `pre_tokens` splits back into phones per word. Destressing is already done by
    `data_common.word_phone_intervals`, so `as_chars`'s own `rstrip("012")` is a no-op
    here. The speech-time denominator is bit-identical to the `words` and `phones`
    tiers, so bit rates stay paired across every unit.

    **It is not information-equivalent to `phone`.** The piece sequence recovers the
    phone string, but the segmentation itself depended on where the words divided:
    `DH-AH K-AE-T` ("the cat") and a hypothetical one-word `DH-AH-K AE-T` would
    tokenise differently from the same phones. So this unit is handed partial
    word-boundary information for free, as `subword-BPE` is, and `wordphn` is handed
    all of it. Any gap to `phone` therefore mixes better modelling with more
    information, and cannot be read as modelling alone.

    No fallback cascade and no spell-out augmentation, for `SubwordBpeTokenizer`'s own
    reason: every phone is in the alphabet by construction, so nothing can be out of
    vocabulary and `<unk>` is unreachable.

    Train and load are two paths, again as for `SubwordBpeTokenizer`: `train_new`
    returns the `tokenizers.Tokenizer` backend, `.save(path)` persists it, then
    `SubwordPhoneXBpeTokenizer(path)` loads it. The mmap-prep script and the training
    script must use the same saved file.
    """

    def __init__(self, tokenizer_file):
        backend = Tokenizer.from_file(str(tokenizer_file))
        super().__init__(
            tokenizer_object=backend,
            bos_token="<bos>",
            pad_token="<pad>",
            unk_token="<unk>",
            eos_token=None,
            add_bos_token=True,
            add_eos_token=False,
        )

    @staticmethod
    def as_chars(phones):
        """Destressed phones as one private-use character each, joined by nothing, so
        one word is a single BPE word
        """
        return "".join(_PHONE_TO_CHAR[p.rstrip("012")] for p in phones)

    @staticmethod
    def as_phones(piece):
        """A vocabulary piece written back out as "-"-joined phones, for inspection.
        A piece holding no mapped character is returned untouched, so the special tokens
        print as themselves
        """
        if not any(c in _CHAR_TO_PHONE for c in piece):
            return piece
        return "-".join(_CHAR_TO_PHONE.get(c, c) for c in piece)

    @classmethod
    def pre_tokens(cls, tokens):
        """One BPE word per `WORDPHN_TIER` token, i.e. per spoken word, each already
        mapped to private-use characters. `train_new` joins them with spaces for a
        `WhitespaceSplit` pre-tokeniser to split again, which is what confines a merge
        to a single word.
        """
        return [cls.as_chars(t.split("-")) for t in tokens]

    @classmethod
    def train_new(cls, utterances, vocab_size=2000):
        """Learn a fresh BPE vocabulary from `utterances`, each a list of tier tokens
        (training data only, per CLAUDE.md's "vocabulary built from training data only"
        principle). Returns the `tokenizers.Tokenizer` backend, not a
        `SubwordPhoneXBpeTokenizer`; save it, then load with the constructor.
        """
        backend = Tokenizer(BPE(unk_token="<unk>"))
        backend.pre_tokenizer = WhitespaceSplit()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<bos>", "<pad>", "<unk>"],
        )
        backend.train_from_iterator(
            (" ".join(cls.pre_tokens(u)) for u in utterances), trainer=trainer
        )
        return backend

    def _encode_plus(self, text, text_pair=None, is_split_into_words=False, **kwargs):
        """Map each word's phones to their private-use characters and hand the result
        to the backend as pre-split words, so one word is one BPE word.
        """

        def transform(seq):
            return self.pre_tokens(seq if is_split_into_words else seq.split())

        text = _map_sequences(text, is_split_into_words, transform)
        return super()._encode_plus(
            text, text_pair=text_pair, is_split_into_words=True, **kwargs
        )

    def _decode(self, *args, **kwargs):
        """Decoded text as phones rather than private-use characters, one "-"-joined
        group per piece
        """
        decoded = super()._decode(*args, **kwargs)
        return " ".join(self.as_phones(piece) for piece in decoded.split())

class WordTokenizer(_MarkedFallbackTokenizer):
    """Word-level tokeniser with hybrid word/character backoff: a word not in the
    training-only vocabulary is spelled out as its own characters instead of getting a
    dedicated slot, so there is no separate `<unk>` case and no utterance is ever
    skipped.

    The vocabulary (word types occurring at least `min_count` times in training) is
    data-dependent, so, like `SubwordBpeTokenizer`, training and loading are two
    different paths: `save_vocab` persists the word-type list computed via
    `data_common.build_min_count_vocab`, then the constructor loads that same file. The
    mmap-prep script and the training script must use the same saved vocab file, never
    recompute it independently, or validation and training would silently use different
    vocabularies.

    Random augmentation (2% of all occurrences, independent of frequency, also spelled
    out during training, so the model gets enough exposure to the fallback pathway) is
    deliberately *not* implemented here: it must never apply at validation/evaluation
    time, only when building the training mmap, so it lives in
    `llm/word_prepare_mmap.py` instead, as a pre-processing step before
    `tokenizer.encode()` ever runs. This class only ever applies the deterministic
    in-vocab/spell-out rule, identically at train and test time.
    """

    def __init__(self, vocab_path):
        self._word_vocab = set(json.loads(Path(vocab_path).read_text()))
        tokens = [
            "<bos>",
            "<pad>",
            *sorted(self._word_vocab),
            *_fallback_vocab(CHARACTERS),
        ]
        vocab = {t: i for i, t in enumerate(tokens)}
        backend = Tokenizer(WordLevel(vocab=vocab))
        backend.pre_tokenizer = WhitespaceSplit()
        super().__init__(
            tokenizer_object=backend,
            bos_token="<bos>",
            pad_token="<pad>",
            unk_token=None,
            eos_token=None,
            add_bos_token=True,
            add_eos_token=False,
        )

    def _encode_plus(self, text, text_pair=None, is_split_into_words=False, **kwargs):
        """Words in vocab are kept whole; anything else (rare training words, genuinely
        unseen dev-clean words) is spelled out as characters, which are always in vocab
        by construction, so this never has to recurse.
        """

        def transform(seq):
            words = seq if is_split_into_words else seq.split()
            expanded = []
            for w in words:
                if w.startswith(FALLBACK_MARKER):
                    expanded.append(w)  # Already a fallback piece, from augmentation
                elif w in self._word_vocab:
                    expanded.append(w)
                else:
                    expanded.extend(FALLBACK_MARKER + c for c in w)
            return expanded if is_split_into_words else " ".join(expanded)

        text = _map_sequences(text, is_split_into_words, transform)
        return super()._encode_plus(
            text, text_pair=text_pair, is_split_into_words=is_split_into_words, **kwargs
        )

    @staticmethod
    def save_vocab(word_types, path):
        """Persist a computed word vocabulary (e.g. from
        `data_common.build_min_count_vocab`) so the constructor can load the exact same
        set later.
        """
        Path(path).write_text(json.dumps(sorted(word_types)))


class SyllableTokenizer(_MarkedFallbackTokenizer):
    """Syllable-level tokeniser with hybrid syllable/phone backoff: a syllable not in
    the training-only vocabulary is spelled out as its own constituent phones instead of
    getting a dedicated slot. Structurally identical to `WordTokenizer` (same
    save/load-vocab pattern, same `_encode_plus`-normalises-then-delegates shape), with
    two differences:

    - The finer unit is phones (`PHONES`, the 39-symbol unstressed ARPABET set already
      used by `ArpabetTokenizer`), not characters.
    - Splitting a syllable into phones needs no lookup: the `syllables` tier already
      writes each syllable as its phones joined by "-" (e.g. `B-AH-T`), verified
      exhaustively (`ngram/ngram_syllable_sweep.py`'s `spelling_of`, all 292,365
      utterances, every split) to reconstruct the destressed phone sequence with zero
      mismatches. So the backoff is a plain `token.split("-")`, matching that existing
      convention exactly.
    """

    def __init__(self, vocab_path):
        self._syllable_vocab = set(json.loads(Path(vocab_path).read_text()))
        tokens = [
            "<bos>",
            "<pad>",
            *sorted(self._syllable_vocab),
            *_fallback_vocab(PHONES),
        ]
        vocab = {t: i for i, t in enumerate(tokens)}
        backend = Tokenizer(WordLevel(vocab=vocab))
        backend.pre_tokenizer = WhitespaceSplit()
        super().__init__(
            tokenizer_object=backend,
            bos_token="<bos>",
            pad_token="<pad>",
            unk_token=None,
            eos_token=None,
            add_bos_token=True,
            add_eos_token=False,
        )

    def _encode_plus(self, text, text_pair=None, is_split_into_words=False, **kwargs):
        """Syllables in vocab are kept whole; anything else (rare training syllables,
        genuinely unseen dev-clean syllables) is spelled out as phones via
        `token.split("-")`, which are always in vocab by construction, so this never has
        to recurse.
        """

        def transform(seq):
            syllables = seq if is_split_into_words else seq.split()
            expanded = []
            for s in syllables:
                if s.startswith(FALLBACK_MARKER):
                    expanded.append(s)  # Already a fallback piece, from augmentation
                elif s in self._syllable_vocab:
                    expanded.append(s)
                else:
                    expanded.extend(FALLBACK_MARKER + p for p in s.split("-"))
            return expanded if is_split_into_words else " ".join(expanded)

        text = _map_sequences(text, is_split_into_words, transform)
        return super()._encode_plus(
            text, text_pair=text_pair, is_split_into_words=is_split_into_words, **kwargs
        )

    @staticmethod
    def save_vocab(syllable_types, path):
        """Persist a computed syllable vocabulary (e.g. from
        `data_common.build_min_count_vocab`) so the constructor can load the exact same
        set later.
        """
        Path(path).write_text(json.dumps(sorted(syllable_types)))


class WordPhoneTokenizer(SyllableTokenizer):
    """The `wordphn` unit: whole words identified by their realised phones rather than
    their spelling, e.g. "abbey" -> `AE-B-IY` (`data_common.word_phone_intervals`).

    A subclass with no body, deliberately. The token format is the one
    `SyllableTokenizer` already handles -- destressed ARPABET joined by "-" -- and so is
    the backoff, `token.split("-")` into `~`-marked phones from `PHONES`, which the
    39-symbol inventory covers by construction, so it can never fail. Reimplementing any
    of that would be duplication that could drift; the subclass exists only to name the
    unit, so the training and evaluation scripts read as the unit they are about.

    The fallback/whole-unit collision `FALLBACK_MARKER` guards against is if anything
    more common here than for syllables: every single-phone word ("a" -> `AH`, "I" ->
    `AY`, "owe" -> `OW`) is both a whole-unit vocabulary entry and a phone that a longer
    word's spell-out produces.
    """


class WordBpeTokenizer(_MarkedFallbackTokenizer):
    """Word-level BPE tokeniser: `ngram/bpe.py`'s `BPETokeniser` (repeatedly merging the
    most frequent adjacent pair of units, starting from whole words), distinct from
    `SubwordBpeTokenizer`'s genuine subword BPE. A merged unit can contain an internal
    space (e.g. "of the"), so unlike every other tokeniser here this backend sets no
    pre-tokenizer at all: `BPETokeniser.encode()` does the segmentation itself,
    upstream, and the already-segmented units are always passed to the HF backend via
    `is_split_into_words=True`, which skips pre-tokenisation entirely for
    already-pre-tokenised input -- a `WhitespaceSplit` pre-tokenizer (used by every
    other class here) would wrongly re-split "of the" into two pieces.

    Vocabulary: every unit `BPETokeniser.encode()` can produce over the training corpus
    it was learned from -- every training word type (an unmerged word is still a valid
    singleton unit) plus the learned merges. Computed once (by re-encoding the training
    corpus) and saved alongside the merges, not recomputed on each load, since that
    would mean re-reading every training TextGrid again. A unit dev-clean produces that
    isn't in this set can only be a genuinely unseen word: merges only ever fire on
    pairs of *known* training words, so an unseen word always emerges as its own
    singleton, unmerged unit. It's spelled out via its own characters, exactly
    `WordTokenizer`'s rule.

    Like `SubwordBpeTokenizer`, the vocabulary is learned, so training and loading are
    two paths: `train_new` learns fresh merges and returns `(bpe, vocab)`; `save`/the
    constructor persist and reload that pair together, so mmap-prep and later
    training/validation never see a different tokeniser.

    The last level of the cascade is parametrised by `SUBUNITS`/`split_subunits` so
    `WordPhoneBpeTokenizer` can reuse everything else; the defaults here are the word
    level's own, so the vocabulary ids are unaffected.
    """

    # The finer unit an out-of-vocabulary word backs off to, and how to split a word
    # into those pieces. Overridden by `WordPhoneBpeTokenizer` (phones, split on "-")
    SUBUNITS = CHARACTERS

    @staticmethod
    def split_subunits(word):
        return list(word)

    def __init__(self, saved_path):
        saved = json.loads(Path(saved_path).read_text())
        self.bpe = BPETokeniser(saved["n_merges"])
        self.bpe.merges = [(tuple(a), tuple(b)) for a, b in saved["merges"]]
        self._vocab = set(saved["vocab"])
        tokens = [
            "<bos>",
            "<pad>",
            *sorted(self._vocab),
            *_fallback_vocab(self.SUBUNITS),
        ]
        vocab = {t: i for i, t in enumerate(tokens)}
        backend = Tokenizer(WordLevel(vocab=vocab))
        super().__init__(
            tokenizer_object=backend,
            bos_token="<bos>",
            pad_token="<pad>",
            unk_token=None,
            eos_token=None,
            add_bos_token=True,
            add_eos_token=False,
        )

    def _encode_plus(self, text, text_pair=None, is_split_into_words=False, **kwargs):
        """BPE-segment each sequence first (joining a word list into a sentence if
        needed), then back off any unit not in vocab, always delegating with
        `is_split_into_words=True` regardless of the original input form, since the
        result is always an already-segmented list of units.

        **Two-level backoff**: an out-of-vocabulary unit falls back to its own
        constituent *words* first, and only a word that is itself out of vocabulary is
        spelled out as characters. With `min_count=1` an OOV unit was always a single
        unseen word, so this was exactly equivalent to spelling the unit out directly;
        the distinction matters at the current `min_count=2` default, under which a
        *merged*, multi-word unit can be out of vocabulary.

        Spelling a multi-word unit out directly would be wrong twice over: it would
        charge character-level cost for words the model knows perfectly well, and it
        would emit the unit's internal space, which has no `CHARACTERS` entry and so
        fails outright ("WordLevel error: Missing [UNK] token"). Backing off to words
        first avoids both, and matches `ngram/`'s own unit -> words -> characters
        cascade.
        """

        def transform(seq):
            expanded = []
            pending = []

            def flush():
                """BPE-segment the plain words seen so far, then back them off."""
                if not pending:
                    return
                for u in self.bpe.encode(" ".join(pending)):
                    if u in self._vocab:
                        expanded.append(u)
                        continue
                    for w in u.split():
                        if w in self._vocab:
                            expanded.append(w)
                        else:
                            expanded.extend(
                                FALLBACK_MARKER + s for s in self.split_subunits(w)
                            )
                pending.clear()

            # A marked piece (from augmentation) is atomic: it bypasses BPE entirely,
            # so it can neither be merged into a unit nor let its neighbours merge
            # across it -- matching test time, where an unseen word never merges.
            # With no marked pieces, everything lands in one flush, which is exactly
            # the single `" ".join(seq)` encode this replaced.
            for tok in (seq.split() if isinstance(seq, str) else seq):
                if tok.startswith(FALLBACK_MARKER):
                    flush()
                    expanded.append(tok)
                else:
                    pending.append(tok)
            flush()
            return expanded

        text = _map_sequences(text, is_split_into_words, transform)
        return super()._encode_plus(
            text, text_pair=text_pair, is_split_into_words=True, **kwargs
        )

    @staticmethod
    def train_new(sentences, n_merges, min_count=2):
        """Learn fresh merges from `sentences` (plain training sentences, one
        whitespace-joined string per utterance; training data only, per CLAUDE.md's
        "vocabulary from training data only" design principle). Returns `(bpe, vocab)`:
        `bpe` is the fitted `BPETokeniser`, `vocab` is every unit `bpe.encode()`
        produces over `sentences`. Save via `save`, then load with the constructor.

        `min_count` keeps only units occurring at least that many times, the same rule
        `data_common.build_min_count_vocab` applies for word and syllable; anything
        below the threshold falls through to the existing character spell-out in
        `_encode_plus`, exactly as a genuinely unseen word does.

        **Defaults to 2**, matching word and syllable's `MIN_COUNT`, so all three
        units share one vocabulary rule. Without it word-BPE keeps 31,071 hapax word
        types that word discards, inflating its vocabulary to 1.54x word's through the
        threshold rather than through the merges, and costing 4.99 bits/s at 200
        merges. Changing this changes the vocabulary, so the mmap must be regenerated
        and the model retrained.
        """
        bpe = BPETokeniser(n_merges).learn(sentences)
        counts = Counter()
        for sentence in sentences:
            counts.update(bpe.encode(sentence))
        vocab = {u for u, c in counts.items() if c >= min_count}
        return bpe, vocab

    @staticmethod
    def save(bpe, vocab, path):
        """Persist the learned merges and vocabulary together (e.g. from `train_new`),
        so the constructor can reload the exact same tokeniser.
        """
        Path(path).write_text(
            json.dumps(
                {
                    "n_merges": bpe.n_merges,
                    "merges": [[list(a), list(b)] for a, b in bpe.merges],
                    "vocab": sorted(vocab),
                }
            )
        )


class WordPhoneBpeTokenizer(WordBpeTokenizer):
    """The `wordphn-BPE-k` unit: word-level BPE over `wordphn` units, so a unit is a
    multi-word collocation identified by its realised phones rather than its spelling,
    e.g. "of the" -> `AH-V DH-AH` (`data_common.word_phone_intervals` for the unit,
    `ngram/bpe.py`'s `BPETokeniser` for the merging).

    The two separators do not collide: a space joins the words inside a merged unit,
    "-" joins the phones inside a word. So `BPETokeniser` needs no change at all (it
    splits sentences on whitespace, and a realisation never contains one), and the
    backoff cascade is the exact analogue of the word level's, one notch finer at the
    last step:

        BPE unit -> wordphn units -> phones

    rather than word-BPE's unit -> words -> characters. Only `SUBUNITS` and
    `split_subunits` differ from the parent; keeping the cascade itself in
    `WordBpeTokenizer` matters because its middle level is subtle (a multi-word unit
    must back off to its component *words* first, never straight to the finest level,
    which would both overcharge known words and emit the unit's internal space, for
    which there is no vocabulary entry).

    The last level can never fail: `PHONES` is the full 39-symbol unstressed inventory
    and every realisation is built from it by construction, exactly as for
    `WordPhoneTokenizer`.
    """

    SUBUNITS = PHONES

    @staticmethod
    def split_subunits(unit):
        return unit.split("-")

