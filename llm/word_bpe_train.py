"""OPT-125M-derived training script for the word-BPE unit."""

import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import (
    TokenizedUnitsChunkedMMapDataset,
    TokenizedUnitsUtteranceDataset,
    worker_init_fn,
)
from data_common import DATA_ROOT
from train_args import build_parser
from tokenisers import WordBpeTokenizer
from trainer import OPTConfig, TrainConfig, Trainer, collate

if __name__ == "__main__":
    parser = build_parser(__doc__)
    parser.add_argument(
        "--n-merges",
        type=int,
        default=200,
        help="Which llm/word_bpe_prepare_mmap.py --n-merges run to train on (default: 200)",
    )
    args = parser.parse_args()
    n_merges = args.n_merges
    unit_name = f"word-bpe{n_merges}"
    tag_suffix = f"-{args.tag}" if args.tag else ""

    seed = args.seed
    batch_size = 2
    accumulation_steps = 20
    num_workers = 0
    ctx_win_size = 2048
    run_name = f"opt125m-{unit_name}{tag_suffix}-librispeech"  # Checkpoint name and wandb run name

    train_dataset_dir = Path(__file__).resolve().parent / "data" / "word-bpe-mmap"
    saved_path = train_dataset_dir / f"{unit_name}-vocab.json"
    valid_dirs = [
        DATA_ROOT / "dev-clean"
    ]  # Dev-other stays untouched; see phone_train.py

    checkpoint_dir = Path(__file__).resolve().parent / "checkpoints" / run_name
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"{checkpoint_dir} already has checkpoint(s). Pass --tag to give this "
            f"run its own name, or --overwrite to replace them anyway."
        )

    torch.set_float32_matmul_precision("high")  # Optimization for certain GPUs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    assert saved_path.exists(), (
        f"Could not find saved merges/vocab at {saved_path}; "
        f"run llm/word_bpe_prepare_mmap.py --n-merges {n_merges} first."
    )
    tokenizer = WordBpeTokenizer(saved_path)
    bos_token_id = tokenizer.bos_token_id
    pad_token_id = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size

    train_bin = train_dataset_dir / f"{unit_name}-train.bin"
    train_json = train_dataset_dir / f"{unit_name}-train.json"
    assert (
        train_bin.exists() and train_json.exists()
    ), f"Could not find training files in {train_dataset_dir}"

    # Same 81920-token effective batch / --epochs target as every other unit, but
    # computed here from this run's actual token count, not hardcoded; see module
    # docstring for why.
    num_tokens = json.loads(train_json.read_text())["num_tokens"]
    epochs = args.epochs
    steps_per_epoch = num_tokens / 81920
    max_steps = (
        round(steps_per_epoch * epochs / 100) * 100
    )  # Rounded to the nearest 100
    warmup_steps = round(max_steps * 0.08)  # 8% of max_steps
    print(
        f"--n-merges {n_merges}: {num_tokens:,} training tokens, "
        f"{steps_per_epoch:.1f} steps/epoch, {epochs} epochs, "
        f"max_steps={max_steps}, warmup_steps={warmup_steps}"
    )

    train_dataset = TokenizedUnitsChunkedMMapDataset(
        train_bin,
        mmap_dtype=np.uint32,  # Not uint16; see module docstring
        chunk_size=ctx_win_size,
    )
    # Tier_name="words": raw dev-clean words; BPE segmentation and the OOV
    # character-backoff rule both happen inside WordBpeTokenizer.encode() itself.
    valid_dataset = TokenizedUnitsUtteranceDataset(valid_dirs, "words", tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate, pad_token_id=pad_token_id),
        worker_init_fn=partial(worker_init_fn, seed=seed),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=partial(collate, pad_token_id=pad_token_id),
    )

    # Same downsized config as every other unit (d=512, 6 layers, 8 heads); only
    # vocab_size/pad_token_id/bos_token_id differ, from this unit's tokeniser.
    model_cfg = OPTConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        num_hidden_layers=6,
        ffn_dim=2048,
        max_position_embeddings=2048,
        do_layer_norm_before=True,
        _remove_final_layer_norm=False,
        word_embed_proj_dim=512,
        dropout=0.1,
        attention_dropout=0.0,
        num_attention_heads=8,
        activation_function="relu",
        layerdrop=0.0,
        init_std=0.02,
        use_cache=True,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=bos_token_id,
        enable_bias=True,
        layer_norm_elementwise_affine=True,
    )
    train_cfg = TrainConfig(
        entity=None,  # <--- Keep as None (auto) or change to a specifc wandb account name
        project="inforate2-llm",  # <--- Change this to group runs in wandb
        name=run_name,
        device="cuda",
        dtype=torch.bfloat16,
        accumulation_steps=accumulation_steps,
        grad_clip_max_norm=1.0,
        lr_init=1e-7,
        lr_max=args.lr_max,
        lr_final=2e-5,
        n_linear_steps=warmup_steps,
        n_decay_steps=max_steps - warmup_steps,
        betas=(0.9, 0.98),
        weight_decay=0.01,
        eps=1e-8,
        checkpoint_dir=checkpoint_dir,
        wandb_mode="online",  # Use "offline" if you do not want to sync
    )
    trainer = Trainer(
        model_cfg,
        train_cfg,
        log_dict={
            "batch_size": batch_size,
            "seed": seed,
            "vocab_size": vocab_size,
            "n_merges": n_merges,
        },
    )
    trainer.train(
        train_loader,
        valid_loader,
        max_global_step=max_steps,
        log_every=10,
        validate_every=500,
    )
