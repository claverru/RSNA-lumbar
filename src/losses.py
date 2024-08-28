from typing import Callable, Dict, List, Literal

import torch
from torchmetrics import Metric

from src import constants, utils


class LumbarLoss(torch.nn.Module):
    def __init__(
        self,
        do_any_severe_spinal: bool = True,
        conditions: List[str] = constants.CONDITIONS,
        gamma: float = 1.0,
        any_severe_spinal_smoothing: float = 0,
        any_severe_spinal_t: float = 0,
        train_weight_components: bool = False,
    ):
        super().__init__()

        weight = torch.tensor([1.0, 2.0, 4.0])
        if gamma == 1.0:
            self.cond_loss = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=weight)
            self.any_severe_spinal_loss = BCEWithLogitsLossSmoothed(any_severe_spinal_smoothing, reduction="none")
        else:
            self.cond_loss = FocalLoss(gamma, ignore_index=-1, weight=weight)
            self.any_severe_spinal_loss = FocalLoss(gamma, reduction="none", binary=True)

        self.do_any_severe_spinal = do_any_severe_spinal
        self.conditions = conditions

        self.weights = (
            torch.nn.Parameter(torch.zeros(len(conditions) + do_any_severe_spinal)) if train_weight_components else None
        )
        self.any_severe_spinal_t = any_severe_spinal_t

    def __compute_severe_spinal_loss(
        self, y_true_dict: Dict[str, torch.Tensor], y_pred_dict: Dict[str, torch.Tensor]
    ) -> float:
        severe_spinal_true = self.get_condition_tensors(y_true_dict, "spinal_canal_stenosis", torch.stack).amax(0)
        is_valid = severe_spinal_true != -1

        if ~is_valid.any():
            return -1.0

        severe_spinal_preds = self.get_condition_tensors(y_pred_dict, "spinal_canal_stenosis", torch.stack)
        lse_first_two = severe_spinal_preds[..., :2].logsumexp(-1)
        severe_spinal_binary_preds = severe_spinal_preds[..., 2] - lse_first_two

        if self.any_severe_spinal_t > 0:
            att = torch.softmax(severe_spinal_binary_preds / self.any_severe_spinal_t, dim=0)
            severe_spinal_binary_preds = (att * severe_spinal_binary_preds).sum(0)
        else:
            severe_spinal_binary_preds = severe_spinal_binary_preds.amax(0)

        weight = torch.pow(2.0, severe_spinal_true)
        severe_spinal_binary_true = torch.where(severe_spinal_true == 2, 1.0, 0.0)

        pred = severe_spinal_binary_preds[is_valid]
        target = severe_spinal_binary_true[is_valid]
        weight = weight[is_valid]

        severe_spinal_loss = (self.any_severe_spinal_loss(pred, target) * weight).sum() / weight.sum()

        return severe_spinal_loss

    def get_condition_tensors(
        self, tensors: Dict[str, torch.Tensor], condition: str, f: Callable = torch.concat
    ) -> torch.Tensor:
        tensors = [tensors[k] for k in sorted(tensors) if condition in k]
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

    def weight_component(self, loss, weight):
        return torch.exp(-weight) * loss + 0.5 * weight

    def forward(self, y_true_dict: Dict[str, torch.Tensor], y_pred_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        assert all(k in y_true_dict for k in y_pred_dict)
        losses = {}
        for condition in self.conditions:
            losses[condition] = self.__compute_condition_loss(y_true_dict, y_pred_dict, condition)
        if self.do_any_severe_spinal:
            losses["any_severe_spinal"] = self.__compute_severe_spinal_loss(y_true_dict, y_pred_dict)

        if self.weights is not None:
            valid_losses = {
                k: self.weight_component(loss, weight)
                for (k, loss), weight in zip(losses.items(), self.weights)
                if loss != -1
            }
        else:
            valid_losses = {k: loss for k, loss in losses.items() if loss != -1.0}

        total_loss = sum(valid_losses.values()) / len(valid_losses)

        return total_loss, losses


class LumbarMetric(Metric):
    is_differentiable = False
    higher_is_better = False

    def __init__(
        self, do_any_severe_spinal: bool = True, conditions: List[str] = constants.CONDITIONS, gamma: float = 1.0
    ):
        super().__init__()
        self.loss_f = LumbarLoss(do_any_severe_spinal, conditions, gamma)
        self.add_state("y_pred_dicts", default=[], dist_reduce_fx="cat")
        self.add_state("y_true_dicts", default=[], dist_reduce_fx="cat")

    def compute(self) -> Dict[str, float]:
        y_true_dict = utils.cat_tensors(self.y_true_dicts)
        y_pred_dict = utils.cat_tensors(self.y_pred_dicts)
        return self.loss_f(y_true_dict, y_pred_dict)

    def update(self, y_true_dict: Dict[str, torch.Tensor], y_pred_dict: Dict[str, torch.Tensor]):
        self.y_pred_dicts.append(y_pred_dict)
        self.y_true_dicts.append(y_true_dict)


class FocalLoss(torch.nn.Module):
    def __init__(
        self,
        gamma,
        weight=None,
        ignore_index: int = -100,
        reduction: Literal["mean", "sum", "none"] = "mean",
        binary: bool = False,
    ):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.binary = binary
        self.ignore_index = ignore_index
        if binary:
            self.cross = torch.nn.BCEWithLogitsLoss(reduction="none")
        else:
            self.cross = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=self.ignore_index, reduction="none")

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        valid = target != self.ignore_index

        preds = preds[valid]
        target = target[valid]

        cross_entropy = self.cross(preds, target)

        if self.binary:
            positive_class_probas = preds.sigmoid()
        else:
            probas = preds.softmax(1)
            positive_class_probas = torch.gather(probas, 1, target[..., None])

        loss = torch.pow(1 - positive_class_probas, self.gamma) * cross_entropy

        match self.reduction:
            case "mean":
                return loss.mean()
            case "sum":
                return loss.sum()
            case "none":
                return loss


class BCEWithLogitsLossSmoothed(torch.nn.BCEWithLogitsLoss):
    def __init__(self, smoothing: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        target = target * (1 - self.smoothing) + (self.smoothing / 2)
        return super().forward(pred, target)
