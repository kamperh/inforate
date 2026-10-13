"""Train a subword BPE tokeniser (vocab 2000) on LibriSpeech 960h training `words`-tier
sentences, save it, then dump a flat token-id mmap from the same training data.

Standard subword BPE (distinct from `ngram/bpe.py`'s word-level collocation merging),
via HuggingFace `tokenizers`' own `BpeTrainer`; see llm/tokenisers.py,
`SubwordBpeTokenizer`. The tokeniser is saved to `TOKENIZER_PATH` so
`llm/subword_bpe_train.py` loads the exact same one for validation-time encoding, rather
than training a second, possibly different, tokeniser.
"""

from pathlib import Path

import numpy as np

from data_common import TRAIN_SPLITS, read_tier, write_train_mmap
from tokenisers import SubwordBpeTokenizer

OUT_DIR = Path(__file__).resolve().parent / "data" / "subword-bpe-mmap"
TOKENIZER_PATH = OUT_DIR / "tokenizer.json"
VOCAB_SIZE = 2000
DTYPE = np.uint16

if __name__ == "__main__":
    print("Reading words tier for:", TRAIN_SPLITS)
    utterances = read_tier(TRAIN_SPLITS, "words")
    print(f"Utterances: {len(utterances):,}")

    train_sentences = [" ".join(tokens) for tokens, _ in utterances.values()]

    print(f"Training subword BPE (vocab_size={VOCAB_SIZE})...")
    backend = SubwordBpeTokenizer.train_new(train_sentences, vocab_size=VOCAB_SIZE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    backend.save(str(TOKENIZER_PATH))
    print("Saved tokeniser to:", TOKENIZER_PATH)

    tokenizer = SubwordBpeTokenizer(TOKENIZER_PATH)
    print(f"Actual vocab size: {tokenizer.vocab_size}")
    assert tokenizer.vocab_size < np.iinfo(DTYPE).max

    write_train_mmap(tokenizer, utterances, OUT_DIR, "subword-bpe", dtype=DTYPE)
