import math
from typing import Dict, Union

import torch

from src import constants, losses, model
from src.patch import model as patch_model


def get_proj(in_dim, out_dim, dropout=0):
    return torch.nn.Sequential(
        torch.nn.Dropout(dropout) if dropout else torch.nn.Identity(),
        torch.nn.Linear(in_dim, out_dim) if in_dim is not None else torch.nn.LazyLinear(out_dim),
    )


class MaskDropout(torch.nn.Module):
    def __init__(self, p: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def forward(self, mask):
        if self.training:
            drop_mask = torch.rand(mask, device=mask.device) < self.p
            return mask & drop_mask
        else:
            return mask


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class LightningModule(model.LightningModule):
    def __init__(
        self,
        image: dict,
        train_any_severe_spinal: bool = True,
        emb_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        add_mid_transformer: bool = False,
        att_dropout: float = 0.1,
        emb_dropout: float = 0.1,
        out_dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_loss = losses.LumbarLoss(train_any_severe_spinal)
        self.val_metric = losses.LumbarMetric(True)
        self.backbone = patch_model.LightningModule(**image)

        self.meta_proj = get_proj(None, emb_dim, emb_dropout)
        self.proj = get_proj(None, emb_dim, emb_dropout)
        self.pos = PositionalEncoding(emb_dim, att_dropout, max_len=5000)

        # self.transformer = model.get_transformer(emb_dim, n_heads, n_layers, att_dropout)
        self.transformer = torch.nn.Transformer(
            emb_dim, n_heads, n_layers, dropout=att_dropout, batch_first=True, norm_first=True
        )

        self.mid_transformer = model.get_transformer(emb_dim, n_heads, 1, att_dropout) if add_mid_transformer else None

        self.head = get_proj(emb_dim, 3, out_dropout)
        self.conditions = torch.nn.Parameter(
            torch.randn(1, len(constants.CONDITIONS_COMPLETE), emb_dim), requires_grad=True
        )
        self.register_buffer("conditions_mask", torch.zeros(1, len(constants.CONDITIONS_COMPLETE), dtype=torch.bool))

        self.maybe_restore_checkpoint()

    def extract_features(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        true_mask = mask.logical_not()
        B, T, C, H, W = x.shape
        filter_x = x[true_mask]
        filter_feats: torch.Tensor = self.backbone(filter_x)
        feats = torch.zeros((B, T, filter_feats.shape[-1]), device=filter_feats.device, dtype=filter_feats.dtype)
        feats[true_mask] = filter_feats
        return feats

    def forward_one(self, x: torch.Tensor, meta: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.extract_features(x, mask)

        feats = self.proj(feats)
        meta = self.meta_proj(meta)

        feats = self.pos(feats)
        meta = self.pos(meta)

        feats = torch.concat([self.conditions.repeat(feats.shape[0], 1, 1), feats], axis=1)
        decoder_mask = torch.concat([self.conditions_mask.repeat(mask.shape[0], 1), mask], axis=1)

        out = self.transformer(src=meta, tgt=feats, src_key_padding_mask=mask, tgt_key_padding_mask=decoder_mask)
        out = out[:, : len(constants.CONDITIONS_COMPLETE)]
        return out

    def forward(
        self,
        x: Union[Dict[str, torch.Tensor], torch.Tensor],
        meta: Union[Dict[str, torch.Tensor], torch.Tensor],
        mask: Union[Dict[str, torch.Tensor], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if isinstance(x, dict):
            feats = []
            levels = constants.LEVELS
            for level in levels:
                feats.append(self.forward_one(x[level], meta[level], mask[level]))

        else:
            levels = ["none"]
            feats = [self.forward_one(x, meta, mask)]

        feats = torch.concatenate(feats, 1)
        if self.mid_transformer is not None:
            feats = self.mid_transformer(feats)

        out_keys = [f"{cond}_{lvl}" for lvl in enumerate(levels) for cond in constants.CONDITIONS_COMPLETE]

        outs = {}
        for i, k in enumerate(out_keys):
            outs[k] = self.head(feats[:, i])
        return outs

    def training_step(self, batch, batch_idx):
        x, meta, mask, y = batch
        pred = self.forward(x, meta, mask)
        losses = self.train_loss(y, pred)
        loss = sum(losses.values()) / len(losses)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, meta, mask, y = batch
        pred = self.forward(x, meta, mask)
        self.val_metric.update(y, pred)

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def do_metric_on_epoch_end(self, prefix):
        losses = self.val_metric.compute()
        self.val_metric.reset()
        for k, v in losses.items():
            self.log(f"{prefix}_{k}_loss", v, on_epoch=True, prog_bar=False, on_step=False)

        loss = sum(losses.values()) / len(losses)
        self.log(f"{prefix}_loss", loss, on_epoch=True, prog_bar=True, on_step=False)

    def on_validation_epoch_end(self, *args, **kwargs):
        self.do_metric_on_epoch_end("val")

    def on_test_epoch_end(self, *args, **kwargs):
        self.do_metric_on_epoch_end("test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, meta, mask = batch
        pred = self.forward(x, meta, mask)
        return pred
