from typing import Union
import torch
import lightning as L
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from torch.optim.lr_scheduler import _LRScheduler


class LightningModule(L.LightningModule):
    def __init__(
        self,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        interval: str = "epoch",
        frequency: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.interval = interval
        self.frequency = frequency

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        lr_scheduler = self.lr_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": self.interval, "frequency": self.frequency},
        }


def get_transformer(emb_dim, n_heads, n_layers, dropout):
    layer = torch.nn.TransformerEncoderLayer(
        emb_dim, n_heads, dropout=dropout, dim_feedforward=emb_dim * 2, batch_first=True, norm_first=True
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
