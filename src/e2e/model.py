from typing import Dict

import timm
import torch
import lightning as L

from src import constants, losses, model

M = 103
D = 8


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
        self.backbone = timm.create_model(arch, num_classes=0, in_chans=1, pretrained=pretrained).eval()
        self.F = self.backbone.num_features

        self.desc = torch.nn.Embedding(4, 8, padding_idx=3)

        self.proj = get_proj(self.F + M + D, emb_dim, linear_dropout)

        self.seq = get_att(emb_dim, n_heads, n_layers, att_dropout)

        self.heads = torch.nn.ModuleDict(
            {cl: get_head(emb_dim, linear_dropout) for cl in constants.CONDITION_LEVEL}
        )

    def forward(self, x: torch.Tensor, meta: torch.Tensor, desc: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, L, C, H, W = x.shape
        padding_mask = desc == 3
        true_mask = padding_mask.logical_not()
        filter_x = x[true_mask]
        filter_feats: torch.Tensor = self.backbone(filter_x)
        feats = torch.zeros((B, L, self.F), device=filter_feats.device, dtype=filter_feats.dtype)
        feats[true_mask] = filter_feats
        desc_emb = self.desc(desc)
        feats = torch.concat([feats, meta, desc_emb], -1)
        seq = self.proj(feats)
        seq = self.seq(seq, src_key_padding_mask=padding_mask)
        seq[padding_mask] = -100
        seq = seq.amax(1)
        outs = {k: head(seq) for k, head in self.heads.items()}
        return outs

    def training_step(self, batch, batch_idx):
        x, meta, desc, y_true_dict = batch
        y_pred_dict = self.forward(x, meta, desc)
        batch_size = y_pred_dict[constants.CONDITION_LEVEL[0]].shape[0]
        losses: Dict[str, torch.Tensor] = self.train_loss.jit_loss(y_true_dict, y_pred_dict)

        for k, v in losses.items():
            self.log(f"train_{k}_loss", v, on_epoch=True, prog_bar=False, on_step=False, batch_size=batch_size)

        loss = sum(losses.values()) / len(losses)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, on_step=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        x, meta, desc, y_true_dict = batch
        y_pred_dict = self.forward(x, meta, desc)
        self.val_loss.update(y_true_dict, y_pred_dict)

    def on_validation_epoch_end(self, *args, **kwargs):
        losses = self.val_loss.compute()
        self.val_loss.reset()
        for k, v in losses.items():
            self.log(f"val_{k}_loss", v, on_epoch=True, prog_bar=False, on_step=False)

        loss = sum(losses.values()) / len(losses)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, on_step=False)
