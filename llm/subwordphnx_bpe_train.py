"""OPT-125M-derived training script for the word-confined subword phone BPE unit."""

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
from data_common import DATA_ROOT, WORDPHN_TIER
from train_args import build_parser
from tokenisers import SubwordPhoneXBpeTokenizer
from trainer import OPTConfig, TrainConfig, Trainer, collate

if __name__ == "__main__":
    parser = build_parser(__doc__)
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=2000,
        help="Which llm/subwordphnx_bpe_prepare_mmap.py --vocab-size run to train on "
        "(default: 2000)",
    )
    args = parser.parse_args()
    vocab_size_arg = args.vocab_size
    tag_suffix = f"-{args.tag}" if args.tag else ""

    # Same 81920-token effective batch and ~20 epoch target as every other unit, on
    # this unit's own token count, which here depends on --vocab-size: a smaller
    # vocabulary merges less and so leaves a longer stream and a larger step budget.
    # See the printed line below for the actual one.

    seed = args.seed
    # The softmax in cross_entropy is the binding memory cost and it scales with the
    # vocabulary, so the batch follows the size rather than being fixed: 200 to 5000
    # take subword-bpe's 10/4, as syllable does at 11,427, while anything larger takes
    # word and wordphn's 2/20, verified at their own 58,006 and 56,296. The
    # 81920-token effective batch is identical either way, so the canonical setup is
    # unchanged and only the memory profile differs
    batch_size, accumulation_steps = (2, 20) if vocab_size_arg > 20000 else (10, 4)
    num_workers = 0
    ctx_win_size = 2048
    unit_name = f"subwordphnx-bpe{vocab_size_arg}"  # Must match the `name` passed to
    # write_train_mmap in llm/subwordphnx_bpe_prepare_mmap.py
    run_name = f"opt125m-{unit_name}{tag_suffix}-librispeech"  # Checkpoint/wandb name
    train_dataset_dir = Path(__file__).resolve().parent / "data" / "subwordphnx-bpe-mmap"
    tokenizer_path = train_dataset_dir / f"{unit_name}-tokenizer.json"
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

    assert tokenizer_path.exists(), (
        f"Could not find saved tokeniser at {tokenizer_path}; "
        f"run llm/subwordphnx_bpe_prepare_mmap.py --vocab-size {vocab_size_arg} "
        "first."
    )
    tokenizer = SubwordPhoneXBpeTokenizer(tokenizer_path)
    bos_token_id = tokenizer.bos_token_id
    pad_token_id = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size

    train_bin = train_dataset_dir / f"{unit_name}-train.bin"
    train_json = train_dataset_dir / f"{unit_name}-train.json"
    assert (
        train_bin.exists() and train_json.exists()
    ), f"Could not find training files in {train_dataset_dir}"

    # max_steps/warmup_steps derived from this unit's own token count and --epochs, at
    # the shared 81920-token effective batch. Same rule for every unit, so a
    # regenerated mmap cannot silently desynchronise the step budget.
    num_tokens = json.loads(train_json.read_text())["num_tokens"]
    steps_per_epoch = num_tokens / 81920
    max_steps = round(steps_per_epoch * args.epochs / 100) * 100  # Nearest 100
    warmup_steps = round(max_steps * 0.08)  # 8% of max_steps
    print(
        f"{unit_name}: {num_tokens:,} training tokens, {steps_per_epoch:.1f} "
        f"steps/epoch, {args.epochs} epochs, max_steps={max_steps}, "
        f"warmup_steps={warmup_steps}"
    )

    train_dataset = TokenizedUnitsChunkedMMapDataset(
        train_bin,
        mmap_dtype=np.uint16,
        chunk_size=ctx_win_size,
    )
    # WORDPHN_TIER, not "phones": this unit's merges are confined to a word, so the
    # tokeniser needs the word grouping at validation time exactly as it had it at
    # training time. The phones are identical either way; only the grouping differs
    valid_dataset = TokenizedUnitsUtteranceDataset(valid_dirs, WORDPHN_TIER, tokenizer)

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

    # Same downsized config as phone_train.py (d=512, 6 layers, 8 heads); only
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
        },
    )
    trainer.train(
        train_loader,
        valid_loader,
        max_global_step=max_steps,
        log_every=10,
        validate_every=500,
    )
