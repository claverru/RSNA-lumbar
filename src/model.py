from typing import Dict

import torch
import lightning as L
import timm

from src import constants


class Model(torch.nn.Module):
    def __init__(self, arch: str):
        super().__init__()
        self.backbone = timm.create_model(arch, pretrained=True, num_classes=0)
        n_feats = self.backbone.num_features
        self.heads = torch.nn.ModuleDict({out: torch.nn.Linear(n_feats, 3) for out in constants.CONDITION_LEVEL})

    def forward(self, x):
        feats = self.backbone(x)
        outs = {k: head(feats) for k, head in self.heads.items()}
        return outs


class LightningModule(L.LightningModule):
    def __init__(self, arch: str = "resnet34", lr: float = 1e-3):
        super().__init__()
        self.backbone = timm.create_model(arch, pretrained=True, num_classes=0)
        self.backbone.eval()
        n_feats = self.backbone.num_features
        self.heads = torch.nn.ModuleDict({out: torch.nn.Linear(n_feats, 3) for out in constants.CONDITION_LEVEL})
        self.lr = lr
        self.loss_f = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward_train(self, x: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        flat_mask = mask.reshape(-1)
        B, S, C, H, W = x.shape
        flat_x = x.reshape(-1, C, H, W)
        flat_x = flat_x[flat_mask]
        flat_feats = self.backbone(flat_x)
        F = flat_feats.shape[-1]
        feats = torch.ones(B * S, F, dtype=flat_feats.dtype, device=flat_feats.device) * -10
        feats[flat_mask] = flat_feats
        feats = feats.reshape(B, S, F)
        pooled = feats.max(dim=1)[0]
        outs = {k: head(pooled) for k, head in self.heads.items()}
        return outs

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, S, C, H, W = x.shape
        flat_x = x.reshape(-1, C, H, W)
        flat_feats: torch.Tensor = self.backbone(flat_x)
        F = flat_feats.shape[-1]
        feats = flat_feats.reshape(B, S, F)
        pooled = feats.max(dim=1)[0]
        outs = {k: head(pooled) for k, head in self.heads.items()}
        return outs

    def training_step(self, batch, batch_idx):
        x, y_true_dict, mask = batch
        y_pred_dict = self.forward_train(x, mask)
        loss = 0.0
        for k, y_true in y_true_dict.items():
            if (y_true == -1).all():
                continue
            y_pred = y_pred_dict[k]
            loss += self.loss_f(y_pred, y_true)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true_dict, mask = batch
        y_pred_dict = self.forward_train(x, mask)
        loss = 0.0
        for k, y_true in y_true_dict.items():
            if (y_true == -1).all():
                continue
            y_pred = y_pred_dict[k]
            loss += self.loss_f(y_pred, y_true)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
