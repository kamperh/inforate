"""Train word-confined subword phone BPE tokenisers on LibriSpeech 960h training
`wordphones`-tier utterances, save each, then dump a flat token-id mmap per vocabulary
size from the same training data.

`llm/subwordphn_bpe_prepare_mmap.py` with merges confined to a word, which makes it the
faithful phonetic counterpart of `llm/subword_bpe_prepare_mmap.py`, whose
`WhitespaceSplit` pre-tokeniser confines its own merges the same way. It reads
`WORDPHN_TIER` rather than the `phones` tier because that is where the word grouping
lives; the phones are the same phones and the speech-time denominator is unchanged. See
llm/tokenisers.py, `SubwordPhoneXBpeTokenizer`, including why this unit, unlike the
crossing one, is not information-equivalent to `phone`.

`--vocab-size` takes **several** sizes and loops over them in one process, which is the
one place this script deviates from `llm/word_bpe_prepare_mmap.py`'s one-size-per-run
flag: deriving `WORDPHN_TIER` over 281,239 TextGrids is the dominant cost here and
re-reading it per size would waste a quarter of an hour each time. Every output filename
embeds the size (`data/subwordphnx-bpe-mmap/subwordphnx-bpe{v}-*`, and
`checkpoints/opt125m-subwordphnx-bpe{v}-librispeech/` from
`llm/subwordphnx_bpe_train.py`), so different sizes never collide.

The tokeniser is saved per size so `llm/subwordphnx_bpe_train.py` loads the exact same
one for validation-time encoding, rather than training a second, possibly different,
tokeniser.
"""

import argparse
from pathlib import Path

import numpy as np

from data_common import TRAIN_SPLITS, WORDPHN_TIER, read_tier, write_train_mmap
from tokenisers import SubwordPhoneXBpeTokenizer

OUT_DIR = Path(__file__).resolve().parent / "data" / "subwordphnx-bpe-mmap"
DTYPE = np.uint16

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vocab-size",
        type=int,
        nargs="+",
        default=[2000],
        help="Tokeniser vocabulary sizes to build, including the three specials and the "
        "39 single phones, so the merge count is this minus 42. 2000 matches "
        "subword-bpe exactly and is the default; 200, 5000 and 50000 trace how the "
        "unit sits between phone and wordphn, and past it (results.md, "
        "'Vocabulary size').",
    )
    vocab_sizes = parser.parse_args().vocab_size

    print("Reading wordphones tier for:", TRAIN_SPLITS)
    utterances = read_tier(TRAIN_SPLITS, WORDPHN_TIER)
    print(f"Utterances: {len(utterances):,}")
    sentences = [tokens for tokens, _ in utterances.values()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for vocab_size in vocab_sizes:
        name = f"subwordphnx-bpe{vocab_size}"
        print(f"\n=== {name} ===", flush=True)
        print(f"Training word-confined subword phone BPE (vocab_size={vocab_size})...")
        backend = SubwordPhoneXBpeTokenizer.train_new(
            sentences, vocab_size=vocab_size
        )
        tokenizer_path = OUT_DIR / f"{name}-tokenizer.json"
        backend.save(str(tokenizer_path))
        print("Saved tokeniser to:", tokenizer_path)

        tokenizer = SubwordPhoneXBpeTokenizer(tokenizer_path)
        print(f"Actual vocab size: {tokenizer.vocab_size}")
        assert tokenizer.vocab_size < np.iinfo(DTYPE).max

        # The longest pieces are the readable check that the confinement held: none may
        # be longer than a single word's realisation
        pieces = [
            SubwordPhoneXBpeTokenizer.as_phones(t)
            for t in sorted(tokenizer.get_vocab(), key=tokenizer.get_vocab().get)
        ]
        print("First 20 pieces learned:", pieces[42:62])
        print("Longest pieces:", sorted(pieces, key=len)[-5:])

        write_train_mmap(tokenizer, utterances, OUT_DIR, name, dtype=DTYPE)
