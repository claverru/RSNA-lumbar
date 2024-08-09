from typing import Dict

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


def detach_tensors(tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach() for k, v in tensors.items()}


class LightningModule(model.LightningModule):
    def __init__(
        self,
        image: dict,
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
        self.train_loss = losses.LumbarLoss(False, conditions=["subarticular"])
        self.val_metric = losses.LumbarMetric(False, conditions=["subarticular"])
        self.backbone = patch_model.LightningModule(**image)

        self.proj = get_proj(None, emb_dim, emb_dropout)

        self.transformer = model.get_transformer(emb_dim, n_heads, n_layers, att_dropout)
        self.levels = torch.nn.Parameter(torch.randn(1, len(constants.LEVELS), emb_dim), requires_grad=True)
        self.register_buffer("levels_mask", torch.zeros(1, 5, dtype=torch.bool))

        self.mid_transformer = model.get_transformer(emb_dim, n_heads, 1, att_dropout) if add_mid_transformer else None

        self.head = get_proj(emb_dim, 3, out_dropout)

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
        feats = torch.concat([feats, meta], -1)

        feats = self.proj(feats)

        feats = torch.concat([self.levels.repeat(feats.shape[0], 1, 1), feats], axis=1)
        mask = torch.concat([self.levels_mask.repeat(mask.shape[0], 1), mask], axis=1)

        out = self.transformer(feats, src_key_padding_mask=mask)
        out = out[:, : len(constants.LEVELS)]
        return out

    def forward(
        self, x: Dict[str, torch.Tensor], meta: Dict[str, torch.Tensor], mask: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        feats = []
        sides = list(x)
        for side in sides:
            feats.append(self.forward_one(x[side], meta[side], mask[side]))

        feats = torch.stack(feats, 1)
        if self.mid_transformer is not None:
            feats = self.mid_transformer(feats)

        outs = {}
        for i, side in enumerate(sides):
            for j, level in enumerate(constants.LEVELS):
                outs[f"{side}_subarticular_stenosis_{level}"] = self.head(feats[:, i, j])
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
