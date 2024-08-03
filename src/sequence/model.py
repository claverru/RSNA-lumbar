from typing import Dict

import torch

from src import constants, losses, model
from src.patch import model as patch_model


def get_proj(in_dim, out_dim, dropout=0):
    return torch.nn.Sequential(
        torch.nn.Dropout(dropout) if dropout else torch.nn.Identity(),
        torch.nn.Linear(in_dim, out_dim) if in_dim is not None else torch.nn.LazyLinear(out_dim),
    )


class LightningModule(model.LightningModule):
    def __init__(
        self,
        arch,
        emb_dim,
        n_heads,
        n_layers,
        att_dropout,
        linear_dropout,
        pretrained=True,
        eval=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_loss = losses.LumbarLoss()
        self.val_loss = losses.LumbarMetric()
        if isinstance(pretrained, str):
            self.backbone = patch_model.LightningModule.load_from_checkpoint(pretrained, pretrained=False)
        else:
            self.backbone = patch_model.LightningModule(arch, linear_dropout, pretrained, eval)

        self.D = emb_dim
        self.transformer = model.get_transformer(emb_dim, n_heads, n_layers, att_dropout)
        self.proj = get_proj(None, emb_dim, linear_dropout)
        self.meta_proj = get_proj(None, 4, linear_dropout)

        self.heads = torch.nn.ModuleDict(
            {k: get_proj(emb_dim, 3, linear_dropout) for k in constants.CONDITIONS_COMPLETE}
        )

    def extract_features(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        true_mask = mask.logical_not()
        B, T, C, H, W = x.shape
        filter_x = x[true_mask]
        filter_feats: torch.Tensor = self.backbone(filter_x)
        feats = torch.zeros((B, T, filter_feats.shape[-1]), device=filter_feats.device, dtype=filter_feats.dtype)
        feats[true_mask] = filter_feats
        return feats

    def forward(self, x: torch.Tensor, meta: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.extract_features(x, mask)

        meta = self.meta_proj(meta)
        feats = torch.concat([feats, meta], -1)

        feats = self.proj(feats)

        out = self.transformer(feats, src_key_padding_mask=mask)
        out[mask] = -100
        out = out.amax(1)
        outs = {k: head(out) for k, head in self.heads.items()}
        return outs

    def training_step(self, batch, batch_idx):
        x, meta, mask, y = batch
        pred = self.forward(x, meta, mask)
        batch_size = y[list(y)[0]].shape[0]

        losses: Dict[str, torch.Tensor] = self.train_loss(y, pred)

        for k, v in losses.items():
            self.log(f"train_{k}_loss", v, on_epoch=True, prog_bar=False, on_step=False, batch_size=batch_size)

        loss = sum(losses.values()) / len(losses)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, on_step=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        x, meta, mask, y = batch
        pred = self.forward(x, meta, mask)
        self.val_loss.update(y, pred)

    def on_validation_epoch_end(self, *args, **kwargs):
        losses = self.val_loss.compute()
        self.val_loss.reset()
        for k, v in losses.items():
            self.log(f"val_{k}_loss", v, on_epoch=True, prog_bar=False, on_step=False)

        loss = sum(losses.values()) / len(losses)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, on_step=False)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, meta, mask = batch
        pred = self.forward(x, meta, mask)
        return pred
