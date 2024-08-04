from typing import Callable, Dict

import torch
from torchmetrics import Metric

from src import constants, utils


class LumbarLoss(torch.nn.Module):
    def __init__(self, do_any_severe_spinal: bool = True):
        super().__init__()
        self.cond_loss = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=torch.tensor([1.0, 2.0, 4.0]))
        self.any_severe_spinal_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.do_any_severe_spinal = do_any_severe_spinal

    def __compute_severe_spinal_loss(
        self, y_true_dict: Dict[str, torch.Tensor], y_pred_dict: Dict[str, torch.Tensor]
    ) -> float:
        severe_spinal_true = self.get_condition_tensors(y_true_dict, "spinal", torch.stack).amax(0)
        is_valid = severe_spinal_true != -1

        if ~is_valid.any():
            return -1

        severe_spinal_preds = self.get_condition_tensors(y_pred_dict, "spinal", torch.stack)
        lse_first_two = severe_spinal_preds[..., :2].logsumexp(-1)
        severe_spinal_binary_preds = (severe_spinal_preds[..., 2] - lse_first_two).amax(0)

        weight = torch.pow(2.0, severe_spinal_true)
        severe_spinal_binary_true = torch.where(severe_spinal_true == 2, 1.0, 0.0)

        severe_spinal_losses: torch.Tensor = (
            self.any_severe_spinal_loss(
                severe_spinal_binary_preds[is_valid].to(torch.float), severe_spinal_binary_true[is_valid]
            )
            * weight[is_valid]
        )

        return severe_spinal_losses.mean()

    def get_condition_tensors(
        self, tensors: Dict[str, torch.Tensor], condition: str, f: Callable = torch.concat
    ) -> torch.Tensor:
        tensors = [tensor for k, tensor in tensors.items() if condition in k]
        tensors = f(tensors, axis=0)
        return tensors

    def __compute_condition_loss(
        self, y_true_dict: Dict[str, torch.Tensor], y_pred_dict: Dict[str, torch.Tensor], condition: str
    ) -> float:
        true_batch = self.get_condition_tensors(y_true_dict, condition)

        is_empty_batch = (true_batch == -1).all()
        if is_empty_batch:
            return -1.0

        pred_batch = self.get_condition_tensors(y_pred_dict, condition)
        return self.cond_loss(pred_batch.to(torch.float), true_batch)

    def forward(self, y_true_dict: Dict[str, torch.Tensor], y_pred_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        losses = {}
        for condition in constants.CONDITIONS:
            losses[condition] = self.__compute_condition_loss(y_true_dict, y_pred_dict, condition)
        if self.do_any_severe_spinal:
            losses["any_severe_spinal"] = self.__compute_severe_spinal_loss(y_true_dict, y_pred_dict)
        return losses


class LumbarMetric(Metric):
    is_differentiable = True
    higher_is_better = False

    def __init__(self, do_any_severe_spinal: bool = True):
        super().__init__()
        self.loss_f = LumbarLoss(do_any_severe_spinal)
        self.add_state("y_pred_dicts", default=[], dist_reduce_fx="cat")
        self.add_state("y_true_dicts", default=[], dist_reduce_fx="cat")

    def compute(self) -> Dict[str, float]:
        y_true_dict = utils.cat_dict_tensor(self.y_true_dicts)
        y_pred_dict = utils.cat_dict_tensor(self.y_pred_dicts)
        return self.loss_f(y_true_dict, y_pred_dict)

    def update(self, y_true_dict: Dict[str, torch.Tensor], y_pred_dict: Dict[str, torch.Tensor]):
        self.y_pred_dicts.append(y_pred_dict)
        self.y_true_dicts.append(y_true_dict)


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma, weight=None, ignore_index=-100, reduction="mean", from_logits=True):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.from_logits = from_logits
        if from_logits:
            self.cross = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=self.ignore_index, reduction="none")
        else:
            self.cross = torch.nn.NLLLoss(weight=weight, ignore_index=self.ignore_index, reduction="none")

    def forward(self, input_: torch.Tensor, target: torch.Tensor):
        cross_entropy = self.cross(input_, target)
        if self.from_logits:
            input_ = input_.softmax(1)

        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(input_, 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        if self.reduction == "mean":
            return loss.mean()
        return loss
