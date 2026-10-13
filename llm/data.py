"""Datasets for llm/<unit>_train.py scripts.

Adapted from docs/assets/opt125m-trainer-example/data.py, generalised in one place:
`TokenizedUnitsUtteranceDataset` takes a `tier_name` ("phones", "syllables",
"words") instead of hardcoding "phones", since the same class serves all five
units' validation loaders. Also uses `tokenizer.encode(tokens,
is_split_into_words=True)` on a token list rather than joining to a string
first, matching llm/data_common.py's `write_train_mmap` convention. TextGrid
walking itself is `data_common.iter_tier`, shared with `read_tier`.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data_common import iter_tier
from trainer import DataItem


# dataset for training:
class TokenizedUnitsChunkedMMapDataset(Dataset):
    """Random fixed-length windows over a flat unit mmap.
    ``__getitem__`` ignores ``index`` and samples a start offset.

    Windows are raw slices of the flat stream, with no synthetic per-window
    ``<bos>`` prepended. The stream already carries real utterance-boundary tokens
    throughout (`data_common.write_train_mmap` keeps each utterance's leading
    ``<bos>``), so a synthetic per-window one would mark an arbitrary chunk start as
    a fake boundary on top of the real ones. This matches how GPT-2/Qwen-style
    corpora are pretrained: concatenated documents with in-stream separators, not
    per-chunk synthetic prefixes.
    """

    def __init__(
        self,
        mmap_data_path: str | Path,
        *,
        mmap_dtype: np.dtype,
        chunk_size: int,
    ):
        self.mmap = np.memmap(mmap_data_path, dtype=mmap_dtype, mode="r")
        assert self.mmap.ndim == 1
        self.chunk_size = chunk_size

    def __len__(self) -> int:
        return int(self.mmap.shape[0]) // self.chunk_size

    def __getitem__(self, index: int) -> DataItem:
        # chunk_size+1 raw tokens, not chunk_size: collate()'s [:-1]/[1:] split needs
        # chunk_size+1 in to yield src/tgt of length chunk_size.
        max_start = len(self.mmap) - (self.chunk_size + 1)
        start = int(np.random.randint(0, max_start))
        units = self.mmap[start : start + self.chunk_size + 1].astype(np.int64)
        tokens = torch.from_numpy(units).long()
        assert tokens.shape[0] == self.chunk_size + 1, tokens.shape
        return DataItem(tokens)


# dataset for validation:
class TokenizedUnitsUtteranceDataset(Dataset):
    """Per-utterance token ids from LibriSpeech alignment TextGrids.

    Appends the boundary token (``tokenizer.bos_token_id``, reused as the
    end-of-utterance marker) after each utterance's own content, matching the
    boundary tokens `data_common.write_train_mmap` bakes into the training stream
    between utterances. `collate()`'s `tgt_tokens = item.tokens[1:]` then scores it
    as the last target. The same class serves both training-time validation (every
    `<unit>_train.py`'s `valid_loader`) and `llm/eval_bitrate.py`'s scoring, so both
    get this for free.
    """

    def __init__(self, roots: list[Path], tier_name: str, tokenizer):
        self.tokenizer = tokenizer
        self.utterances: list[list[str]] = []
        # Kept parallel to `utterances`, purely so a caller that wants
        # per-utterance results can pair them up (llm/eval_bitrate.py's
        # bootstrap dump). `__getitem__` ignores both, so nothing about
        # training-time validation changes.
        self.keys: list[str] = []
        self.durations: list[float] = []
        for root in roots:
            for key, tier in iter_tier(
                root, tier_name, sort=True, desc=f"Loading {Path(root).name}"
            ):
                tokens = [iv.text for iv in tier if iv.text]
                if tokens:
                    self.utterances.append(tokens)
                    self.keys.append(key)
                    # Speech-only seconds: tgt already drops silence, so this
                    # matches `data_common.read_tier`'s own duration exactly.
                    self.durations.append(sum(iv.duration() for iv in tier))

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> DataItem:
        ids = self.tokenizer.encode(self.utterances[idx], is_split_into_words=True)
        ids = ids + [self.tokenizer.bos_token_id]
        return DataItem(torch.tensor(ids, dtype=torch.long))


def worker_init_fn(worker_id: int, seed: int) -> None:
    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
