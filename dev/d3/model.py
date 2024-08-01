from typing import Dict

import timm
import torch

from src import constants, losses, model
from src.d3.constants import PLANE2SIZE


def get_proj(in_dim, out_dim, dropout=0):
    return torch.nn.Sequential(
        torch.nn.Dropout(dropout) if dropout else torch.nn.Identity(),
        torch.nn.Linear(in_dim, out_dim) if in_dim is not None else torch.nn.LazyLinear(out_dim)
    )


class LightningModule(model.LightningModule):
    def __init__(self, arch, emb_dim, linear_dropout, pretrained=True, **kwargs):
        super().__init__(**kwargs)
        self.train_loss = losses.LumbarLoss()
        self.val_loss = losses.LumbarLoss()
        self.backbone = torch.nn.ModuleDict(
            {k: timm.create_model(arch, num_classes=0, in_chans=PLANE2SIZE[k], pretrained=pretrained).eval() for k in constants.DESCRIPTIONS}
        )
        self.norms = torch.nn.ModuleDict(
            {k: torch.nn.InstanceNorm2d(PLANE2SIZE[k]) for k in constants.DESCRIPTIONS}
        )

        self.projs = torch.nn.ModuleDict({k: get_proj(None, emb_dim, linear_dropout) for k in constants.LEVELS})
        self.heads = torch.nn.ModuleDict({k: get_proj(emb_dim, 3, linear_dropout) for k in constants.CONDITIONS_COMPLETE})


    def forward_one(self, x, plane):
        x = self.norms[plane](x[plane])
        x = self.backbone[plane](x)
        return x

    def forward(self, x) -> Dict[str, torch.Tensor]:
        feats = []
        for plane in constants.DESCRIPTIONS:
            feats.append(self.forward_one(x, plane))

        feats = torch.concat(feats, -1)

        outs = {}
        for level in constants.LEVELS:
            level_feats = self.projs[level](feats)
            for cond, head in self.heads.items():
                outs[f"{cond}_{level}"] = head(level_feats)
        return outs

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        batch_size = y[list(y)[0]].shape[0]

        losses: Dict[str, torch.Tensor] = self.train_loss.jit_loss(y, pred)

        for k, v in losses.items():
            self.log(f"train_{k}_loss", v, on_epoch=True, prog_bar=False, on_step=False, batch_size=batch_size)

        loss = sum(losses.values()) / len(losses)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, on_step=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        self.val_loss.update(y, pred)

    def on_validation_epoch_end(self, *args, **kwargs):
        losses = self.val_loss.compute()
        self.val_loss.reset()
        for k, v in losses.items():
            self.log(f"val_{k}_loss", v, on_epoch=True, prog_bar=False, on_step=False)

        loss = sum(losses.values()) / len(losses)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, on_step=False)
