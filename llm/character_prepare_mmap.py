"""Dump a flat character-id mmap from LibriSpeech 960h train TextGrids.

Reads the `words` tier and spells each word out into its letters; the tokeniser does
the splitting, so this script stays as thin as phone_prepare_mmap.py. No vocabulary is
built and no augmentation is applied: the 27-symbol inventory is fixed and covers every
spelling the tier can contain, so there is nothing to threshold and nothing to spell
out (see tokenisers.CharacterTokenizer).
"""

from pathlib import Path

import numpy as np

from data_common import TRAIN_SPLITS, read_tier, write_train_mmap
from tokenisers import CharacterTokenizer

OUT_DIR = Path(__file__).resolve().parent / "data" / "character-mmap"
DTYPE = np.uint16

if __name__ == "__main__":
    tokenizer = CharacterTokenizer()
    assert tokenizer.vocab_size < np.iinfo(DTYPE).max

    print("Reading words tier for:", TRAIN_SPLITS)
    utterances = read_tier(TRAIN_SPLITS, "words")
    print(f"Utterances: {len(utterances):,}")

    write_train_mmap(tokenizer, utterances, OUT_DIR, "character", dtype=DTYPE)
