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


def get_transformer(emb_dim, n_heads, n_layers, dropout):
    layer = torch.nn.TransformerEncoderLayer(
        emb_dim, n_heads, dropout=dropout, dim_feedforward=emb_dim * 2, batch_first=True, norm_first=True
    )
    encoder = torch.nn.TransformerEncoder(layer, n_layers, norm=torch.nn.LayerNorm(emb_dim), enable_nested_tensor=False)
    return encoder
