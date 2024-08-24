from typing import Dict

import timm
import torch

from src import constants, losses, model


def get_proj(in_dim, out_dim, dropout=0):
    return torch.nn.Sequential(
        torch.nn.Dropout(dropout) if dropout else torch.nn.Identity(),
        torch.nn.Linear(in_dim, out_dim) if in_dim is not None else torch.nn.LazyLinear(out_dim),
    )


class LightningModule(model.LightningModule):
    def __init__(
        self, arch, linear_dropout=0.2, pretrained=True, eval=True, do_any_severe_spinal: bool = False, **kwargs
    ):
        super().__init__(**kwargs)
        self.train_loss = losses.LumbarLoss(do_any_severe_spinal)
        self.val_metric = losses.LumbarMetric(do_any_severe_spinal)
        self.norm = torch.nn.InstanceNorm2d(1)
        self.backbone = timm.create_model(arch, num_classes=0, in_chans=1, pretrained=pretrained, img_size=96)
        if eval:
            self.backbone = self.backbone.eval()

        self.heads = torch.nn.ModuleDict({k: get_proj(None, 3, linear_dropout) for k in constants.CONDITIONS})

        self.maybe_restore_checkpoint()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.norm(x)
        x = self.backbone(x)
        return x

    def forward_train(self, x):
        x = self.forward(x)
        outs = {k: head(x) for k, head in self.heads.items()}
        return outs

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward_train(x)
        loss, _ = self.train_loss(y, pred)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward_train(x)
        self.val_metric.update(y, pred)

    def do_metric_on_epoch_end(self, prefix):
        loss, losses = self.val_metric.compute()
        self.val_metric.reset()
        for k, v in losses.items():
            self.log(f"{prefix}_{k}_loss", v, on_epoch=True, prog_bar=False, on_step=False)
        self.log(f"{prefix}_loss", loss, on_epoch=True, prog_bar=True, on_step=False)

    def on_validation_epoch_end(self, *args, **kwargs):
        self.do_metric_on_epoch_end("val")

    def on_test_epoch_end(self, *args, **kwargs):
        self.do_metric_on_epoch_end("test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        pred = self.forward_train(x)
        return pred
