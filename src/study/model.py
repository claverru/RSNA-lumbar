from typing import Dict

import torch
import lightning as L
import timm

from src import constants
from src.study import constants as study_constants


class LightningModule(L.LightningModule):
    def __init__(self, arch: str = "resnet34"):
        super().__init__()
        self.loss_f = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=torch.tensor([1., 2., 4.]))
        self.spinal_loss_f = torch.nn.NLLLoss(ignore_index=-1, reduction="none")
        self.backbone = timm.create_model(arch, pretrained=True, in_chans=study_constants.IMGS_PER_DESC, num_classes=0)
        n_feats = self.backbone.num_features
        self.heads = torch.nn.ModuleDict({out: torch.nn.Linear(n_feats * 3, 3) for out in constants.CONDITION_LEVEL})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        feats: torch.Tensor = self.backbone(x) # (B * S, n_feats)
        feats = feats.view(B, -1)
        outs = {k: head(feats) for k, head in self.heads.items()}
        return outs

    def compute_severe_spinal_loss(self, y_true_dict, y_pred_dict):
        severe_spinal_true = torch.stack(
            [y_true_dict[k] for k in constants.CONDITION_LEVEL if "spinal" in k], -1
        ).max(-1)[0]
        is_empty_spinal_batch = (severe_spinal_true == -1).all()

        if is_empty_spinal_batch:
            return -1

        severe_spinal_preds = torch.stack(
            [torch.softmax(y_pred_dict[k], -1)[:, -1] for k in constants.CONDITION_LEVEL if "spinal" in k],
            -1
        ).max(-1)[0]

        severe_spinal_preds = torch.stack([1 - severe_spinal_preds, severe_spinal_preds], -1)
        weight = torch.pow(2.0, severe_spinal_true)
        severe_spinal_binary_true = torch.where(
            severe_spinal_true > 0, severe_spinal_true - 1, severe_spinal_true
        )

        severe_spinal_losses = self.spinal_loss_f(
            torch.log(severe_spinal_preds), severe_spinal_binary_true
        ) * weight
        return severe_spinal_losses[severe_spinal_true != -1].mean()


    def compute_condition_loss(self, y_true_dict, y_pred_dict, condition):
        true_batch = [y_true for k, y_true in y_true_dict.items() if condition in k]
        true_batch = torch.concat(true_batch, 0)

        is_empty_batch = (true_batch == -1).all()
        if is_empty_batch:
            return -1

        pred_batch = [y_true for k, y_true in y_pred_dict.items() if condition in k]
        pred_batch = torch.concat(pred_batch, 0)

        return self.loss_f(pred_batch, true_batch)


    def do_loss(self, y_true_dict: Dict[str, torch.Tensor], y_pred_dict: Dict[str, torch.Tensor]):
        losses = {}
        for condition in constants.CONDITIONS:
            losses[condition] = self.compute_condition_loss(y_true_dict, y_pred_dict, condition)
        losses["severe_spinal"] = self.compute_severe_spinal_loss(y_true_dict, y_pred_dict)
        return losses

    def training_step(self, batch, batch_idx):
        x, y_true_dict = batch
        y_pred_dict = self.forward(x)
        losses = self.do_loss(y_true_dict, y_pred_dict)

        for k, v in losses.items():
            self.log(f"train_{k}_loss", v, on_epoch=True, prog_bar=True, on_step=False)

        loss = sum(losses.values()) / len(losses)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true_dict = batch
        y_pred_dict = self.forward(x)
        losses = self.do_loss(y_true_dict, y_pred_dict)

        for k, v in losses.items():
            self.log(f"val_{k}_loss", v, on_epoch=True, prog_bar=True, on_step=False)

        loss = sum(losses.values()) / len(losses)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, on_step=False)

        return loss
