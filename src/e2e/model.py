from typing import Dict

import timm
import torch

from src import constants, losses, model
from src.sequence.constants import *


def get_att(emb_dim, n_heads, n_layers, dropout):
    layer = torch.nn.TransformerEncoderLayer(
        emb_dim, n_heads, dropout=dropout, dim_feedforward=emb_dim * 2, batch_first=True, norm_first=True
    )
    encoder = torch.nn.TransformerEncoder(layer, n_layers, norm=torch.nn.LayerNorm(emb_dim), enable_nested_tensor=False)
    return encoder


def get_proj(feats, emb_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(feats, emb_dim)
    )

def get_proj(feats, emb_dim, dropout):
    return torch.nn.Sequential(
        torch.nn.Linear(feats, emb_dim * 2),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(emb_dim * 2, emb_dim),
    )



def get_head(in_features, dropout):
    return torch.nn.Sequential(
        torch.nn.Dropout(dropout),
        torch.nn.Linear(in_features, 3)
    )


class LightningModule(model.LightningModule):
    def __init__(self, arch, emb_dim, n_heads, n_layers, att_dropout, linear_dropout, pretrained=True, **kwargs):
        super().__init__(**kwargs)
        self.train_loss = losses.LumbarLoss()
        self.val_loss = losses.LumbarLoss()
        self.img_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 2., 4., 0.2]), ignore_index=-1)
        self.backbone = timm.create_model(arch, num_classes=0, in_chans=1, pretrained=pretrained).eval()

        self.F = self.backbone.num_features
        self.D = emb_dim
        self.L = len(constants.LEVELS)
        self.C = len(constants.CONDITIONS_COMPLETE)
        self.S = len(constants.SEVERITY2LABEL) + 1

        self.img_proj = get_proj(self.F, self.C * self.L * self.D, linear_dropout)
        self.desc = torch.nn.Embedding(4, P, padding_idx=3)
        self.seq_proj = get_proj(self.C * self.L * self.D + M + P, self.C * self.L * self.D, linear_dropout)
        self.seq = get_att(self.C * self.L * self.D, n_heads, n_layers, att_dropout)

        self.condition_severity = torch.nn.Parameter(
            torch.randn(self.C, self.S, self.D), requires_grad=True
        )

    def forward(self, x: torch.Tensor, meta: torch.Tensor, desc: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, _, _, _ = x.shape
        padding_mask = desc == 3
        true_mask = padding_mask.logical_not()

        filter_x = x[true_mask]
        filter_feats: torch.Tensor = self.backbone(filter_x)
        feats = torch.zeros((B, T, self.F), device=filter_feats.device, dtype=filter_feats.dtype)
        feats[true_mask] = filter_feats # (B, T, F)

        feats = self.img_proj(feats) # (B, T, C * L * D)
        leveled_feats = feats.reshape(B, T, self.C, self.L, self.D) # (B, T, C, L,  D)
        out1 = torch.einsum("BTCLD,CSD->BSTCL", leveled_feats, self.condition_severity)

        desc_emb = self.desc(desc)
        feats = torch.concat([feats, meta, desc_emb], -1)
        seq = self.seq_proj(feats)
        seq = self.seq(seq, src_key_padding_mask=padding_mask)
        seq[padding_mask] = -100
        seq = seq.amax(1)

        leveled_seq = seq.reshape(B, self.C, self.L, self.D)
        out2 = torch.einsum("BCLD,CSD->BCLS", leveled_seq, self.condition_severity[:, :-1])
        outs = {}
        for i, c in enumerate(constants.CONDITIONS_COMPLETE):
            for j, l in enumerate(constants.LEVELS):
                k = f"{c}_{l}"
                outs[k] = out2[:, i, j]
        return outs, out1.reshape(B, self.S, T, self.C * self.L)

    def training_step(self, batch, batch_idx):
        x, meta, desc, img_y, y_true_dict = batch
        y_pred_dict, img_pred = self.forward(x, meta, desc)
        batch_size = y_pred_dict[constants.CONDITION_LEVEL[0]].shape[0]
        losses: Dict[str, torch.Tensor] = self.train_loss.jit_loss(y_true_dict, y_pred_dict)

        for k, v in losses.items():
            self.log(f"train_{k}_loss", v, on_epoch=True, prog_bar=False, on_step=False, batch_size=batch_size)

        seq_loss = sum(losses.values()) / len(losses)
        self.log("train_seq_loss", seq_loss, on_epoch=True, prog_bar=True, on_step=True, batch_size=batch_size)

        img_loss = self.img_loss(img_pred, img_y)
        self.log("train_img_loss", img_loss, on_epoch=True, prog_bar=True, on_step=True, batch_size=batch_size)

        loss = seq_loss + img_loss
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, on_step=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        x, meta, desc, _, y_true_dict = batch
        y_pred_dict, _ = self.forward(x, meta, desc)
        self.val_loss.update(y_true_dict, y_pred_dict)

    def on_validation_epoch_end(self, *args, **kwargs):
        losses = self.val_loss.compute()
        self.val_loss.reset()
        for k, v in losses.items():
            self.log(f"val_{k}_loss", v, on_epoch=True, prog_bar=False, on_step=False)

        loss = sum(losses.values()) / len(losses)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, on_step=False)
