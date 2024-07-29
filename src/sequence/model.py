from typing import Dict

import torch

from src import constants, model, losses


CONDS = [
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
    "spinal_canal_stenosis",
    "subarticular_stenosis"
]


PLANE2COND = dict(
    zip(
        constants.DESCRIPTIONS,
        (
            "spinal_canal_stenosis",
            "neural_foraminal_narrowing",
            "subarticular_stenosis"
        )
    )
)


def get_proj(in_features, out_features, dropout, activation=None):
    return torch.nn.Sequential(
        torch.nn.Dropout(dropout),
        torch.nn.Linear(in_features, out_features) if in_features is not None else torch.nn.LazyLinear(out_features),
        activation if activation is not None else torch.nn.Identity()
    )


class LightningModule(model.LightningModule):
    def __init__(
        self,
        emb_dim: int,
        n_heads: int,
        n_layers: int,
        att_dropout: float,
        linear_dropout: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.train_loss = losses.LumbarLoss()
        self.val_loss = losses.LumbarLoss()

        self.proj = get_proj(None, emb_dim, linear_dropout)
        self.meta_proj = get_proj(None, emb_dim // 8, 0.0)
        self.transformer = model.get_transformer(emb_dim, n_heads, n_layers, att_dropout)

        self.heads = torch.nn.ModuleDict({k: get_proj(emb_dim, 3, linear_dropout) for k in constants.CONDITIONS_COMPLETE})

    def forward(self, x, meta, mask) -> Dict[str, torch.Tensor]:
        meta = self.meta_proj(meta)
        x = torch.concat([x, meta], -1)
        x = self.proj(x)
        out = self.transformer(x, src_key_padding_mask=mask)
        out[mask] = -100
        out = out.amax(1)
        outs = {k: head(out) for k, head in self.heads.items()}
        return outs

    def training_step(self, batch, batch_idx):
        x, meta, mask, y_true_dict = batch
        y_pred_dict = self.forward(x, meta, mask)
        batch_size = y_pred_dict[list(y_pred_dict)[0]].shape[0]
        losses: Dict[str, torch.Tensor] = self.train_loss.jit_loss(y_true_dict, y_pred_dict)

        for k, v in losses.items():
            self.log(f"train_{k}_loss", v, on_epoch=True, prog_bar=False, on_step=False, batch_size=batch_size)

        loss = sum(losses.values()) / len(losses)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, on_step=False, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        x, meta, mask, y_true_dict = batch
        y_pred_dict = self.forward(x, meta, mask)
        self.val_loss.update(y_true_dict, y_pred_dict)

    def on_validation_epoch_end(self, *args, **kwargs):
        losses = self.val_loss.compute()
        self.val_loss.reset()
        for k, v in losses.items():
            self.log(f"val_{k}_loss", v, on_epoch=True, prog_bar=False, on_step=False)

        loss = sum(losses.values()) / len(losses)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, on_step=False)
