from typing import Dict

import timm
import torch

from src import constants, losses, model


class LightningModule(model.LightningModule):
    def __init__(
        self,
        arch,
        linear_dropout=0.2,
        pretrained=True,
        eval=True,
        do_any_severe_spinal: bool = False,
        gamma: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_loss = losses.LumbarLoss(do_any_severe_spinal, gamma=gamma)
        self.val_metric = losses.LumbarMetric(do_any_severe_spinal, gamma=gamma)
        self.norm = torch.nn.InstanceNorm2d(1)
        self.backbone = timm.create_model(arch, num_classes=0, in_chans=1, pretrained=pretrained)
        if eval:
            self.backbone = self.backbone.eval()

        self.heads = torch.nn.ModuleDict({k: model.get_proj(None, 3, linear_dropout) for k in constants.CONDITIONS})

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
