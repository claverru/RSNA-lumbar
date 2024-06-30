import torch
import lightning as L
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable


class LightningModule(L.LightningModule):
    def __init__(
        self,
        optimizer: OptimizerCallable = torch.optim.Adam,
        lr_scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        interval: str = "epoch",
        frequency: int = 1,
        **kwargs
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
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.interval,
                "frequency": self.frequency
            },
        }
