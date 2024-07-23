from typing import Dict

import timm
import torch

from src import constants, losses, model
from src.patchseq import constants as patchseq_constants


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
            transformers[desc] = model.get_transformer(emb_dim, n_heads, n_layers, att_dropout)
            norms[desc] = torch.nn.InstanceNorm2d(1)
            z_projs[desc] = get_proj(1, emb_dim, 0)
            projs[desc] = get_proj(self.F, emb_dim, 0)

        self.projs = torch.nn.ModuleDict(projs)
        self.z_projs = torch.nn.ModuleDict(z_projs)
        self.transformers = torch.nn.ModuleDict(transformers)
        self.norms = torch.nn.ModuleDict(norms)
        self.mid_transformer = model.get_transformer(emb_dim, n_heads, n_layers, att_dropout)
        self.heads = torch.nn.ModuleDict({k: get_head(emb_dim, linear_dropout) for k in patchseq_constants.CONDITIONS})

    def forward_one(self, x: torch.Tensor, mask: torch.Tensor, z: torch.Tensor, desc: str):
        B, T, P, C, H, W = x.shape
        true_mask = mask.logical_not()[..., None].repeat(1, 1, P)

        filter_x = x[true_mask]
        filter_x = self.norms[desc](filter_x)
        filter_feats: torch.Tensor = self.backbone(filter_x)
        filter_feats = self.projs[desc](filter_feats)

        feats = torch.zeros((B, T, P, self.D), device=filter_feats.device, dtype=filter_feats.dtype)
        feats[true_mask] = filter_feats # (B, T, P, D)

        seqs = []
        z = self.z_projs[desc](z[..., None])
        for i in range(P):
            seq = feats[:, :, i] + z
            seq = self.transformers[desc](seq, src_key_padding_mask=mask)
            seq[mask] = -100
            seq = seq.amax(1)
            seqs.append(seq)

        seqs = torch.stack(seqs, 1)

        return seqs

    def forward(
        self, x: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor], z: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        seqs = []
        for desc in constants.DESCRIPTIONS:
            seq = self.forward_one(x[desc], masks[desc], z[desc], desc)
            seqs.append(seq)

        lens = [s.shape[1] for s in seqs]
        seqs = torch.concat(seqs, 1)
        seqs: torch.Tensor = self.mid_transformer(seqs)
        seqs = seqs.split(lens, 1)

        outs = {}
        for desc_seqs, (desc, cond) in zip(seqs, patchseq_constants.PLANE2COND.items()):
            outs.update(self.get_out(desc, cond, desc_seqs))

        return outs

    def get_out(self, plane, cond, seqs):
        sides = ("right", "left")
        levels = constants.LEVELS
        outs = {}
        match plane:

            case "Axial T2":
                head = self.heads[cond]
                for i, side in enumerate(sides):
                    seq = seqs[:, i]
                    for level in levels:
                        k = f"{side}_{cond}_{level}"
                        outs[k] = head(seq)

            case "Sagittal T1":
                for side in sides:
                    k = f"{side}_{cond}"
                    head = self.heads[k]
                    for i, level in enumerate(levels):
                        seq = seqs[:, i]
                        k1 = f"{k}_{level}"
                        outs[k1] = head(seq)

            case "Sagittal T2/STIR":
                head = self.heads[cond]
                for i, level in enumerate(levels):
                    seq = seqs[:, i]
                    k = f"{cond}_{level}"
                    outs[k] = head(seq)

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
