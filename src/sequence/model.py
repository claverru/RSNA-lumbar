import math
from typing import Dict, List

import torch

from src import constants, losses, model
from src.patch import model as patch_model
from src.sequence.data_loading import META_COLS


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


class CLSEmbedding(torch.nn.Module):
    def __init__(self, num_classes: int = constants.CONDITIONS, emb_dim: int = 512):
        super().__init__()
        self.cls_embs = torch.nn.Parameter(torch.randn(1, num_classes, emb_dim), requires_grad=True)
        self.register_buffer("cls_mask", torch.zeros(1, num_classes, dtype=torch.bool))

    def forward(self, x, mask):
        B = x.shape[0]
        x = torch.concat([self.cls_embs.repeat(B, 1, 1), x], axis=1)
        mask = torch.concat([self.cls_mask.repeat(B, 1), mask], axis=1)
        return x, mask


class LightningModule(model.LightningModule):
    def __init__(
        self,
        image: dict,
        train_any_severe_spinal: bool = True,
        conditions: List[str] = constants.CONDITIONS,
        emb_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        att_dropout: float = 0.1,
        emb_dropout: float = 0.1,
        out_dropout: float = 0.3,
        add_mid_attention: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_loss = losses.LumbarLoss(
            train_any_severe_spinal, conditions=conditions, any_severe_spinal_smoothing=0.1
        )
        self.val_metric = losses.LumbarMetric("spinal_canal_stenosis" in conditions, conditions=conditions)
        self.backbone = patch_model.LightningModule(**image)

        self.conditions = conditions

        self.meta_norm = torch.nn.LayerNorm(len(META_COLS))
        self.meta_proj = get_proj(None, emb_dim, emb_dropout)
        self.proj = get_proj(None, emb_dim, emb_dropout)
        self.pos = PositionalEncoding(emb_dim, att_dropout, max_len=5000)

        self.transformer = torch.nn.Transformer(
            emb_dim,
            n_heads,
            n_layers,
            activation="gelu",
            dropout=att_dropout,
            dim_feedforward=emb_dim * 2,
            batch_first=True,
            norm_first=True,
        )

        if add_mid_attention:
            self.mid_attention = model.get_encoder(emb_dim, n_heads, 1, att_dropout)
        else:
            self.mid_attention = None

        self.heads = torch.nn.ModuleDict({k: get_proj(emb_dim, 3, out_dropout) for k in self.conditions})
        # self.heads = get_proj(emb_dim, 3, out_dropout)
        self.cls_emb = CLSEmbedding(len(self.conditions), emb_dim)
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

        meta = self.meta_norm(meta)
        meta = self.meta_proj(meta)

        feats = self.pos(feats)
        meta = self.pos(meta)

        feats, decoder_mask = self.cls_emb(feats, mask)

        out = self.transformer(
            src=meta,
            tgt=feats,
            src_key_padding_mask=mask,
            tgt_key_padding_mask=decoder_mask,
            memory_key_padding_mask=mask,
        )
        out = out[:, : len(self.conditions)]
        return out

    def forward(
        self, x: Dict[str, torch.Tensor], meta: Dict[str, torch.Tensor], mask: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        levels = list(x)
        sides = list(x[levels[0]])
        feats = []
        for level in levels:
            for side in sides:
                feats.append(self.forward_one(x[level][side], meta[level][side], mask[level][side]))

        feats = torch.concatenate(feats, 1)
        if self.mid_attention is not None:
            feats = self.mid_attention(feats)

        outs = {}
        for lvl, level in enumerate(levels):
            for c, cond in enumerate(self.conditions):
                head = self.heads[cond]
                if cond == "spinal_canal_stenosis":
                    spinal_idx = [self.i(lvl, s, c) for s, _ in enumerate(sides)]
                    feat = feats[:, spinal_idx].mean(1)
                    k = "_".join(i for i in (cond, level) if i != "any")
                    outs[k] = head(feat)
                else:
                    for s, side in enumerate(sides):
                        k = "_".join(i for i in (side, cond, level) if i != "any")
                        feat = feats[:, self.i(lvl, s, c)]
                        outs[k] = head(feat)
        return outs

    def i(self, lvl, s, c):
        return lvl * len(self.conditions) * 2 + s * len(self.conditions) + c

    def training_step(self, batch, batch_idx):
        *x, y = batch
        pred = self.forward(*x)
        losses = self.train_loss(y, pred)
        loss = sum(losses.values()) / len(losses)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        *x, y = batch
        pred = self.forward(*x)
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
        x = batch
        pred = self.forward(*x)
        return pred
