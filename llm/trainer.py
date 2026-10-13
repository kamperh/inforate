"""
Trainer module for language model training using HuggingFace OPT models.

Features
 - Causal LM training with HuggingFace OPT (shifted tokens, pad-aware CE loss)
 - Linear warmup then cosine decay (AdamW, gradient clipping)
 - Mixed precision (autocast; GradScaler for float16)
 - Gradient accumulation
 - Wandb logging of train/val loss, accuracy, LR, and epoch
 - Periodic validation; save best-val and final (and optional step) checkpoints

"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import torch
import wandb
from torch import amp, nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformers import OPTConfig, OPTForCausalLM


@dataclass
class TrainConfig:
    # --- Wandb details ---
    entity: str | None
    project: str
    name: str

    # --- General training control ---
    device: str
    dtype: torch.dtype
    accumulation_steps: int
    grad_clip_max_norm: float

    # --- Optimizer / learning rate schedule ---
    lr_init: float
    lr_max: float
    lr_final: float
    n_linear_steps: int
    n_decay_steps: int
    betas: tuple[float, float]
    weight_decay: float
    eps: float

    checkpoint_dir: Path
    wandb_mode: str = "online"


@dataclass
class DataItem:
    tokens: torch.LongTensor  # [T,]

    def __post_init__(self):
        assert isinstance(self.tokens, torch.LongTensor), self.tokens
        assert self.tokens.ndim == 1, self.tokens.shape


@dataclass
class DataBatch:
    src_tokens: torch.LongTensor  # [B,T]
    tgt_tokens: torch.LongTensor  # [B,T]

    def __post_init__(self):
        assert isinstance(self.src_tokens, torch.LongTensor), self.src_tokens
        assert isinstance(self.tgt_tokens, torch.LongTensor), self.tgt_tokens
        assert self.src_tokens.ndim == 2, self.src_tokens.shape
        assert self.tgt_tokens.ndim == 2, self.tgt_tokens.shape
        assert self.src_tokens.size(0) == self.tgt_tokens.size(0), (
            self.src_tokens.size(0),
            self.tgt_tokens.size(0),
        )

    def pin_memory(self):
        return DataBatch(
            src_tokens=self.src_tokens.pin_memory(),
            tgt_tokens=self.tgt_tokens.pin_memory(),
        )


def collate(items: list[DataItem], pad_token_id: int) -> DataBatch:
    src_tokens = [item.tokens[:-1] for item in items]
    tgt_tokens = [item.tokens[1:] for item in items]
    src_tokens = pad_sequence(src_tokens, batch_first=True, padding_value=pad_token_id)
    tgt_tokens = pad_sequence(tgt_tokens, batch_first=True, padding_value=pad_token_id)
    return DataBatch(src_tokens, tgt_tokens)


class Trainer:
    def __init__(
        self,
        model_cfg: OPTConfig,
        train_cfg: TrainConfig,
        *,
        log_dict: dict = {},
    ):
        # config
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.checkpoint_name = train_cfg.name

        # model
        self.model = OPTForCausalLM(model_cfg).to(self.train_cfg.device)

        # optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg.lr_max,
            betas=self.train_cfg.betas,
            weight_decay=self.train_cfg.weight_decay,
            eps=self.train_cfg.eps,
        )

        # scheduler: linear warmup from lr_init -> lr_max, then cosine to lr_final
        linear = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=self.train_cfg.lr_init / train_cfg.lr_max,
            end_factor=1.0,
            total_iters=self.train_cfg.n_linear_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.train_cfg.n_decay_steps,
            eta_min=self.train_cfg.lr_final,
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[linear, cosine],
            milestones=[self.train_cfg.n_linear_steps],
        )

        # use GradScaler only for float16; bfloat16 uses autocast without scaler
        self.scaler = (
            torch.amp.GradScaler() if self.train_cfg.dtype == torch.float16 else None
        )

        # logger
        self.run = wandb.init(
            entity=train_cfg.entity or None,
            project=train_cfg.project,
            name=train_cfg.name,
            mode=train_cfg.wandb_mode,
            config={
                "model_cfg": model_cfg.__dict__,
                "train_cfg": train_cfg.__dict__,
                **log_dict,
            },
        )

        self.current_step = 0
        self.current_global_step = 0
        self.current_epoch = 0
        self.best_loss = math.inf
        self.pbar = None
        self.steps_in_epoch = None

    def train_step(self, batch: DataBatch) -> torch.FloatTensor:
        src_tokens = batch.src_tokens.to(self.train_cfg.device)
        tgt_tokens = batch.tgt_tokens.to(self.train_cfg.device)
        attention_mask = (src_tokens != self.model_cfg.pad_token_id).long()
        logits: torch.FloatTensor = self.model(
            src_tokens, attention_mask=attention_mask
        ).logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            tgt_tokens.view(-1),
            ignore_index=self.model_cfg.pad_token_id,
        )
        return loss, {"train/loss": loss.detach().item()}

    def valid_step(self, batch: DataBatch) -> dict:
        src_tokens = batch.src_tokens.to(self.train_cfg.device)
        tgt_tokens = batch.tgt_tokens.to(self.train_cfg.device)
        attention_mask = (src_tokens != self.model_cfg.pad_token_id).long()
        logits: torch.FloatTensor = self.model(
            src_tokens, attention_mask=attention_mask
        ).logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            tgt_tokens.view(-1),
            ignore_index=self.model_cfg.pad_token_id,
        )
        preds = logits.argmax(dim=-1)
        mask = tgt_tokens != self.model_cfg.pad_token_id
        correct = (preds == tgt_tokens) & mask
        accuracy = correct.sum() / mask.sum()
        return {
            "val/loss": loss.detach().item(),
            "val/accuracy": accuracy.detach().item(),
        }

    def train_epoch(self, train_loader: DataLoader) -> Iterator[dict]:
        self.model.train()
        self.optimizer.zero_grad()
        self.steps_in_epoch = len(train_loader) // self.train_cfg.accumulation_steps

        global_step_metrics = {}

        for batch_idx, batch in enumerate(train_loader):
            assert isinstance(batch, DataBatch)

            # forward
            with amp.autocast(
                device_type=self.train_cfg.device, dtype=self.train_cfg.dtype
            ):
                loss, batch_metrics = self.train_step(batch)

                # scale down gradients if accumulating gradients
                if self.train_cfg.accumulation_steps > 1:
                    loss = loss / self.train_cfg.accumulation_steps

            # backward
            if self.scaler is None:
                loss.backward()
            else:
                self.scaler.scale(loss).backward()

            self.current_step += 1

            # accumulate step metrics
            for metric, value in batch_metrics.items():
                if metric not in global_step_metrics:
                    global_step_metrics[metric] = []
                global_step_metrics[metric].append(value)

            # Step optimizer when we've accumulated enough gradients
            # (will not step at the end of ecpoch if not enough batches left)
            # (this is fine, we have enough data in LibriSpeech)
            if self.current_step % self.train_cfg.accumulation_steps == 0:
                # unscale before clipping if using scaler
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                # gradient clipping
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.train_cfg.grad_clip_max_norm
                )

                # step optimizer (and scaler if using)
                if self.scaler is None:
                    self.optimizer.step()
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                # step scheduler
                self.scheduler.step()

                self.optimizer.zero_grad()
                self.current_global_step += 1

                yield global_step_metrics
                global_step_metrics = {}
                self.model.train()

        self.current_epoch += 1

    @torch.no_grad()
    def validate(self, valid_loader: DataLoader) -> dict:
        self.model.eval()
        valid_metrics = {}
        for batch in tqdm(valid_loader, desc="Validating", position=1, leave=False):
            assert isinstance(batch, DataBatch)

            # forward
            batch_metrics = self.valid_step(batch)

            # collect metrics
            for metric, value in batch_metrics.items():
                if metric not in valid_metrics:
                    valid_metrics[metric] = []
                valid_metrics[metric].append(value)

        return valid_metrics

    def save_checkpoint(self, name: str) -> None:
        checkpoint_path = self.train_cfg.checkpoint_dir / name
        checkpoint = {
            "model": self.model.state_dict(),
            "cfg": self.model.config.__dict__,
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        max_global_step: int,
        log_every: int,
        validate_every: int,
        checkpoint_every: int | None = None,
        on_epoch_end: Callable | None = None,
    ):
        if self.current_global_step >= max_global_step:
            print(f"Already trained up to {self.current_global_step} steps")
            return

        self.pbar = tqdm(
            total=max_global_step, desc="Training", unit="step", position=0, leave=True
        )

        while True:
            # start running training epoch and
            # get the yielded train_metrics for each global step:
            for train_metrics in self.train_epoch(train_loader):
                self.pbar.update(1)

                is_log_step = self.current_global_step % log_every == 0
                is_val_step = self.current_global_step % validate_every == 0

                log_metrics = {}  # will hold all metrics to be logged

                # compute average train metrics and update dict
                log_metrics |= {
                    metric: sum(values) / len(values)
                    for metric, values in train_metrics.items()
                }

                if is_val_step:
                    # run validation loop
                    valid_metrics = self.validate(valid_loader)

                    # compute average metrics and update dict
                    log_metrics |= {
                        metric: sum(values) / len(values)
                        for metric, values in valid_metrics.items()
                    }

                    # save best_loss (reference only; final weights are scored)
                    current_val_loss = log_metrics["val/loss"]
                    if current_val_loss < self.best_loss:
                        self.best_loss = current_val_loss
                        self.save_checkpoint(f"{self.checkpoint_name}-best.pt")

                if is_log_step or is_val_step:
                    log_metrics |= {
                        "trainer/epoch": self.current_epoch,
                        "trainer/lr": self.scheduler.get_last_lr()[0],
                    }

                    # log to wandb
                    self.run.log(log_metrics, self.current_global_step)

                    # update tqdm
                    self.pbar.set_postfix(
                        {
                            "epoch": self.current_epoch,
                            "lr": f"{self.scheduler.get_last_lr()[0]:.1e}",
                            "train_loss": log_metrics["train/loss"],
                            "val_loss": f"{self.best_loss:.4f}",
                            "steps/epoch": self.steps_in_epoch,
                        }
                    )
                    self.pbar.update(0)

                if (
                    checkpoint_every is not None
                    and self.current_global_step % checkpoint_every == 0
                ):
                    self.save_checkpoint(f"step-{self.current_global_step}.pt")

                # check for final condition
                if self.current_global_step >= max_global_step:
                    print(f"Reached max steps ({max_global_step})")

                    # Final weights are the checkpoint used for downstream
                    # scoring (`eval loglikelihood` reads `{name}.pt`).
                    self.save_checkpoint(f"{self.checkpoint_name}.pt")
                    self.pbar.close()
                    self.run.finish(0)
                    return

            if on_epoch_end is not None:
                on_epoch_end()
