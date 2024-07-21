from typing import Dict

import timm
import torch

from src import constants, losses, model
from src.sequence.constants import *


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
        projs = {}
        transformers = {}
        norms = {}
        z_projs = {}
        for desc in constants.DESCRIPTIONS:
            if desc == "Axial T2":
                projs[desc] = get_proj(self.F * 2, emb_dim, linear_dropout)
            else:
                projs[desc] = get_proj(self.F * 5, emb_dim, linear_dropout)
            transformers[desc] = model.get_transformer(emb_dim, n_heads, n_layers, att_dropout)
            norms[desc] = torch.nn.InstanceNorm2d(1)
            z_projs[desc] = get_proj(1, emb_dim, 0)
        self.projs = torch.nn.ModuleDict(projs)
        self.z_projs = torch.nn.ModuleDict(z_projs)
        self.transformers = torch.nn.ModuleDict(transformers)
        self.norms = torch.nn.ModuleDict(norms)
        self.heads = torch.nn.ModuleDict({k: get_head(emb_dim * 3, linear_dropout) for k in constants.CONDITION_LEVEL})

    def forward_one(self, x: torch.Tensor, mask: torch.Tensor, z: torch.Tensor, desc: str):
        B, T, P, C, H, W = x.shape
        true_mask = mask.logical_not()[..., None].repeat(1, 1, P)

        filter_x = x[true_mask]
        filter_x = self.norms[desc](filter_x)
        filter_feats: torch.Tensor = self.backbone(filter_x)

        feats = torch.zeros((B, T, P, self.F), device=filter_feats.device, dtype=filter_feats.dtype)
        feats[true_mask] = filter_feats # (B, T, P, F)
        feats = feats.reshape(B, T, P * self.F)
        feats = self.projs[desc](feats) # (B, T, D)

        z = self.z_projs[desc](z[..., None])

        feats = feats + z

        seqs = self.transformers[desc](feats, src_key_padding_mask=mask)
        seqs[mask] = -100
        seq = seqs.amax(1)

        return seq

    def forward(
        self, x: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor], z: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        seqs = []
        for desc in constants.DESCRIPTIONS:
            seq = self.forward_one(x[desc], masks[desc], z[desc], desc)
            seqs.append(seq)

        seqs = torch.concat(seqs, -1)

        outs = {k: h(seqs) for k, h in self.heads.items()}

        return outs

    def training_step(self, batch, batch_idx):
        x, masks, z, y = batch
        pred = self.forward(x, masks, z)
        batch_size = y[constants.CONDITION_LEVEL[0]].shape[0]

        losses: Dict[str, torch.Tensor] = self.train_loss.jit_loss(y, pred)

        for k, v in losses.items():
            self.log(f"train_{k}_loss", v, on_epoch=True, prog_bar=False, on_step=False, batch_size=batch_size)

        loss = sum(losses.values()) / len(losses)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, on_step=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        x, masks, z, y = batch
        pred = self.forward(x, masks, z)
        self.val_loss.update(y, pred)

    def on_validation_epoch_end(self, *args, **kwargs):
        losses = self.val_loss.compute()
        self.val_loss.reset()
        for k, v in losses.items():
            self.log(f"val_{k}_loss", v, on_epoch=True, prog_bar=False, on_step=False)

        loss = sum(losses.values()) / len(losses)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, on_step=False)
