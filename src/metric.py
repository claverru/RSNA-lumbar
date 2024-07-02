from typing import Dict

import torch
from torchmetrics import Metric

from src import constants, utils


class LumbarLoss(Metric):

    is_differentiable = True
    higher_is_better = False

    def __init__(self):
        super().__init__()
        self.loss_f = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=torch.tensor([1., 2., 4.]))
        self.spinal_loss_f = torch.nn.NLLLoss(ignore_index=-1, reduction="none")
        self.add_state("y_pred_dicts", default=[], dist_reduce_fx="cat")
        self.add_state("y_true_dicts", default=[], dist_reduce_fx="cat")

    def __compute_severe_spinal_loss(
        self, y_true_dict: Dict[str, torch.Tensor], y_pred_dict: Dict[str, torch.Tensor]
    ) -> float:
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

        severe_spinal_preds = torch.stack([1 - severe_spinal_preds, severe_spinal_preds], -1).to(self.dtype)
        weight = torch.pow(2.0, severe_spinal_true)
        severe_spinal_binary_true = torch.where(
            severe_spinal_true > 0, severe_spinal_true - 1, severe_spinal_true
        )

        severe_spinal_losses = self.spinal_loss_f(
            torch.log(severe_spinal_preds), severe_spinal_binary_true
        ) * weight
        return severe_spinal_losses[severe_spinal_true != -1].mean()


    def __compute_condition_loss(
        self, y_true_dict: Dict[str, torch.Tensor], y_pred_dict: Dict[str, torch.Tensor], condition: str
    ) -> float:
        true_batch = [y_true for k, y_true in y_true_dict.items() if condition in k]
        true_batch = torch.concat(true_batch, 0)

        is_empty_batch = (true_batch == -1).all()
        if is_empty_batch:
            return -1

        pred_batch = [y_true for k, y_true in y_pred_dict.items() if condition in k]
        pred_batch = torch.concat(pred_batch, 0).to(self.dtype)

        return self.loss_f(pred_batch, true_batch)

    def update(self, y_true_dict: Dict[str, torch.Tensor], y_pred_dict: Dict[str, torch.Tensor]):
        self.y_pred_dicts.append(y_pred_dict)
        self.y_true_dicts.append(y_true_dict)

    def jit_loss(self, y_true_dict, y_pred_dict):
        losses = {}
        for condition in constants.CONDITIONS:
            losses[condition] = self.__compute_condition_loss(y_true_dict, y_pred_dict, condition)
        losses["severe_spinal"] = self.__compute_severe_spinal_loss(y_true_dict, y_pred_dict)
        return losses

    def compute(self) -> Dict[str, float]:
        y_true_dict = utils.cat_dict_tensor(self.y_true_dicts)
        y_pred_dict = utils.cat_dict_tensor(self.y_pred_dicts)
        return self.jit_loss(y_true_dict, y_pred_dict)
