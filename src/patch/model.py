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
    def __init__(self, arch, linear_dropout, pretrained=True, eval=True, **kwargs):
        super().__init__(**kwargs)
        self.train_loss = losses.LumbarLoss()
        self.val_loss = losses.LumbarMetric()
        self.norm = torch.nn.InstanceNorm2d(1)
        self.backbone = timm.create_model(arch, num_classes=0, in_chans=1, pretrained=pretrained)
        if eval:
            self.backbone = self.backbone.eval()

        num_features = 1280 if "mobilenetv4" in arch else self.backbone.num_features

        self.heads = torch.nn.ModuleDict({k: get_proj(num_features, 3, linear_dropout) for k in constants.CONDITIONS})

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
        batch_size = y[list(y)[0]].shape[0]

        losses: Dict[str, torch.Tensor] = self.train_loss(y, pred)

        for k, v in losses.items():
            self.log(f"train_{k}_loss", v, on_epoch=True, prog_bar=False, on_step=False, batch_size=batch_size)

        loss = sum(losses.values()) / len(losses)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, on_step=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward_train(x)
        self.val_loss.update(y, pred)

    def on_validation_epoch_end(self, *args, **kwargs):
        losses = self.val_loss.compute()
        self.val_loss.reset()
        for k, v in losses.items():
            self.log(f"val_{k}_loss", v, on_epoch=True, prog_bar=False, on_step=False)

        loss = sum(losses.values()) / len(losses)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, on_step=False)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        pred = self.forward(x)
        return pred
