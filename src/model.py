from pathlib import Path
from typing import Optional, Union

import lightning as L
import torch
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch.optim.lr_scheduler import _LRScheduler


def get_proj(in_dim, out_dim, dropout=0, activation=None, norm=None):
    return torch.nn.Sequential(
        norm if norm is not None else torch.nn.Identity(),
        torch.nn.Dropout(dropout) if dropout else torch.nn.Identity(),
        torch.nn.Linear(in_dim, out_dim) if in_dim is not None else torch.nn.LazyLinear(out_dim),
        activation if activation is not None else torch.nn.Identity(),
    )


class LightningModule(L.LightningModule):
    def __init__(
        self,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        interval: str = "epoch",
        frequency: int = 1,
        ckpt_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["backbone"])
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.interval = interval
        self.frequency = frequency
        self.ckpt_path = ckpt_path

    def maybe_restore_checkpoint(self):
        if self.ckpt_path is not None:
            ckpt_path = Path(self.ckpt_path)
            print(f"Restoring this model weights from {ckpt_path} state_dict ...")
            if ckpt_path.exists():
                checkpoint = torch.load(ckpt_path)
                try:
                    self.load_state_dict(checkpoint["state_dict"])
                except Exception as e:
                    print(e)
                    print("Setting strict to False. Take a look at the unmatching keys above.")
                    self.load_state_dict(checkpoint["state_dict"], strict=False)
                print("Weights loaded.")
            else:
                print("Couldn't load weights. Path doesn't exist.")

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        lr_scheduler = self.lr_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": self.interval, "frequency": self.frequency},
        }


def get_encoder(emb_dim, n_heads, n_layers, dropout):
    layer = torch.nn.TransformerEncoderLayer(
        emb_dim,
        n_heads,
        dropout=dropout,
        dim_feedforward=emb_dim * 2,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    encoder = torch.nn.TransformerEncoder(layer, n_layers, norm=torch.nn.LayerNorm(emb_dim), enable_nested_tensor=False)
    return encoder


class WarmupLR(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: Union[int, float] = 25000,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        return [
            lr * self.warmup_steps**0.5 * min(step_num**-0.5, step_num * self.warmup_steps**-1.5)
            for lr in self.base_lrs
        ]
