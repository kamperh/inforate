"""Dump a flat phone-id mmap from LibriSpeech 960h train TextGrids."""

from pathlib import Path

import numpy as np

from data_common import TRAIN_SPLITS, read_tier, write_train_mmap
from tokenisers import ArpabetTokenizer

OUT_DIR = Path(__file__).resolve().parent / "data" / "phone-mmap"
DTYPE = np.uint16

if __name__ == "__main__":
    tokenizer = ArpabetTokenizer()
    assert tokenizer.vocab_size < np.iinfo(DTYPE).max

    print("Reading phones tier for:", TRAIN_SPLITS)
    utterances = read_tier(TRAIN_SPLITS, "phones")
    print(f"Utterances: {len(utterances):,}")

    write_train_mmap(tokenizer, utterances, OUT_DIR, "phone", dtype=DTYPE)
