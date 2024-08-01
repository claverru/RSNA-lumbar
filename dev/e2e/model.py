from typing import Dict

import timm
import torch

from src import constants, losses, model


def get_proj(feats, emb_dim, dropout):
    return torch.nn.Sequential(
        torch.nn.Dropout(dropout) if dropout else torch.nn.Identity(),
        torch.nn.Linear(feats, emb_dim)
    )


def get_head(in_features, dropout):
    return torch.nn.Sequential(
        torch.nn.Dropout(dropout) if dropout else torch.nn.Identity(),
        torch.nn.Linear(in_features, 3)
    )


class LightningModule(model.LightningModule):
    def __init__(self, arch, emb_dim, n_heads, n_layers, att_dropout, linear_dropout, pretrained=True, **kwargs):
        super().__init__(**kwargs)
        self.train_loss = losses.LumbarLoss()
        self.val_loss = losses.LumbarLoss()
        self.backbone = timm.create_model(arch, num_classes=0, in_chans=1, pretrained=pretrained).eval()

        self.F = self.backbone.num_features
        self.D = emb_dim
        self.norm = torch.nn.InstanceNorm2d(1)
        self.projs = torch.nn.ModuleDict({k: get_proj(self.F, self.D, linear_dropout) for k in constants.LEVELS})
        self.transformer = model.get_transformer(self.D, n_heads, n_layers, att_dropout)

        self.heads = torch.nn.ModuleDict({k: get_head(emb_dim, linear_dropout) for k in constants.CONDITIONS_COMPLETE})


    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        true_mask = mask.logical_not()

        B, T, C, H, W = x.shape

        filter_x = x[true_mask]
        filter_x = self.norm(filter_x)
        filter_feats: torch.Tensor = self.backbone(filter_x)

        feats = torch.zeros((B, T, self.F), device=filter_feats.device, dtype=filter_feats.dtype)
        feats[true_mask] = filter_feats

        outs = {}
        for level in constants.LEVELS:
            level_feats = self.projs[level](feats)
            for cond, head in self.heads.items():
                out = self.transformer(level_feats, src_key_padding_mask=mask)
                out[mask] = -100
                out = out.amax(1)
                outs[f"{cond}_{level}"] = head(out)
        return outs

    def training_step(self, batch, batch_idx):
        x, masks, y = batch
        pred = self.forward(x, masks)
        batch_size = y[list(y)[0]].shape[0]

        losses: Dict[str, torch.Tensor] = self.train_loss.jit_loss(y, pred)

        for k, v in losses.items():
            self.log(f"train_{k}_loss", v, on_epoch=True, prog_bar=False, on_step=False, batch_size=batch_size)

        loss = sum(losses.values()) / len(losses)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, on_step=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        x, masks, y = batch
        pred = self.forward(x, masks)
        self.val_loss.update(y, pred)

    def on_validation_epoch_end(self, *args, **kwargs):
        losses = self.val_loss.compute()
        self.val_loss.reset()
        for k, v in losses.items():
            self.log(f"val_{k}_loss", v, on_epoch=True, prog_bar=False, on_step=False)

        loss = sum(losses.values()) / len(losses)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, on_step=False)
