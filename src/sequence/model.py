from typing import Dict

import torch

from src import constants, model, losses
from src.sequence.constants import INPUT_SIZE, D


def get_att(emb_dim, n_heads, n_layers, dropout):
    layer = torch.nn.TransformerEncoderLayer(
        emb_dim, n_heads, dropout=dropout, dim_feedforward=emb_dim * 2, batch_first=True, norm_first=True
    )
    encoder = torch.nn.TransformerEncoder(layer, n_layers, norm=torch.nn.LayerNorm(emb_dim), enable_nested_tensor=False)
    return encoder


def get_proj(emb_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(INPUT_SIZE, emb_dim * 2),
        torch.nn.Linear(emb_dim * 2, emb_dim),
    )



def get_head(in_features, dropout):
    return torch.nn.Sequential(
        torch.nn.Dropout(dropout),
        torch.nn.Linear(in_features, 3)
    )


class LightningModule(model.LightningModule):
    def __init__(self, emb_dim, n_heads, n_layers, att_dropout, linear_dropout, **kwargs):
        super().__init__(**kwargs)
        self.train_loss = losses.LumbarLoss()
        # self.train_loss_2 = losses.FocalLoss(ignore_index=-1, gamma=5, weight=torch.tensor([1., 2., 4.]))
        self.val_loss = losses.LumbarLoss()

        self.proj = get_proj(emb_dim)
        self.seq = get_att(emb_dim, n_heads, n_layers, att_dropout)
        self.emb = torch.nn.Embedding(4, D, padding_idx=3)

        self.heads = torch.nn.ModuleDict(
            {cl: get_head(emb_dim, linear_dropout) for cl in constants.CONDITION_LEVEL}
        )

    def forward(self, x: torch.Tensor, desc: torch.Tensor) -> Dict[str, torch.Tensor]:
        padding_mask = desc == 3

        desc_emb = self.emb(desc) # B, L, 8

        x = torch.concat([x, desc_emb], -1)
        x = self.proj(x)
        x = self.seq(x, src_key_padding_mask=padding_mask)

        x[padding_mask] = -100
        x = x.amax(1)

        outs = {k: head(x) for k, head in self.heads.items()}
        return outs

    def training_step(self, batch, batch_idx):
        x, desc, y_true_dict = batch
        y_pred_dict = self.forward(x, desc)
        batch_size = y_pred_dict[constants.CONDITION_LEVEL[0]].shape[0]
        losses: Dict[str, torch.Tensor] = self.train_loss.jit_loss(y_true_dict, y_pred_dict)
        # losses2 = {k: self.train_loss_2(y_pred_dict[k], y_true_dict[k]) for k in constants.CONDITION_LEVEL}

        for k, v in losses.items():
            self.log(f"train_{k}_loss", v, on_epoch=True, prog_bar=False, on_step=False, batch_size=batch_size)

        loss = sum(losses.values()) / len(losses) # + sum(losses2.values()) / len(losses2)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, on_step=False, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        x, desc, y_true_dict = batch
        y_pred_dict = self.forward(x, desc)
        self.val_loss.update(y_true_dict, y_pred_dict)

    def on_validation_epoch_end(self, *args, **kwargs):
        losses = self.val_loss.compute()
        self.val_loss.reset()
        for k, v in losses.items():
            self.log(f"val_{k}_loss", v, on_epoch=True, prog_bar=False, on_step=False)

        loss = sum(losses.values()) / len(losses)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, on_step=False)
