from typing import Dict

import torch
import lightning as L
import timm

from src import constants


class LightningModule(L.LightningModule):
    def __init__(self, arch: str = "resnet34"):
        super().__init__()
        self.loss_f = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=torch.tensor([1., 2., 4.]))
        self.spinal_loss_f = torch.nn.NLLLoss(ignore_index=-1, weight=torch.tensor([1., 4.]))
        self.backbone = timm.create_model(arch, pretrained=True, num_classes=0)
        n_feats = self.backbone.num_features
        self.heads = torch.nn.ModuleDict({out: torch.nn.Linear(n_feats, 3) for out in constants.CONDITION_LEVEL})

    def forward(self, x):
        feats = self.backbone(x)
        outs = {k: head(feats) for k, head in self.heads.items()}
        return outs

    def do_loss(self, y_true_dict: Dict[str, torch.Tensor], y_pred_dict: Dict[str, torch.Tensor]):
        losses = {}
        for condition in constants.CONDITIONS:

            true_batch = [y_true for k, y_true in y_true_dict.items() if condition in k]
            true_batch = torch.concat(true_batch, 0)

            is_empty_batch = (true_batch == -1).all()
            if is_empty_batch:
                losses[condition] = -1
                continue

            pred_batch = [y_true for k, y_true in y_pred_dict.items() if condition in k]
            pred_batch = torch.concat(pred_batch, 0)

            losses[condition] = self.loss_f(pred_batch, true_batch)

            if condition == "spinal" and ~is_empty_batch:
                severe_spinal_true_batch = torch.where(true_batch > 0, true_batch - 1, true_batch)
                pred_batch_probas = torch.nn.functional.softmax(pred_batch, dim=-1)
                severe_spinal_pred_batch = torch.concat(
                    [pred_batch_probas[:, :2].sum(1, keepdims=True), pred_batch_probas[:, 2:]], axis=1
                )
                severe_spinal_pred_batch = torch.log(severe_spinal_pred_batch)
                losses["severe_spinal"] = self.spinal_loss_f(
                    severe_spinal_pred_batch, severe_spinal_true_batch
                )

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
