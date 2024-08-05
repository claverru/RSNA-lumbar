from typing import Callable, Dict

import numpy as np
import pandas as pd
import sklearn
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

        torch.save(y_true_dict, "true.pt")
        torch.save(y_pred_dict, "pred.pt")
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


class ParticipantVisibleError(Exception):
    pass


def get_condition(full_location: str) -> str:
    for injury_condition in ["spinal", "foraminal", "subarticular"]:
        if injury_condition in full_location:
            return injury_condition
    raise ValueError(f"condition not found in {full_location}")


def score(
    solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str = "row_id", any_severe_scalar: float = 1.0
) -> float:
    target_levels = ["normal_mild", "moderate", "severe"]

    # Run basic QC checks on the inputs
    if not pd.api.types.is_numeric_dtype(submission[target_levels].values):
        raise ParticipantVisibleError("All submission values must be numeric")

    if not np.isfinite(submission[target_levels].values).all():
        raise ParticipantVisibleError("All submission values must be finite")

    if solution[target_levels].min().min() < 0:
        raise ParticipantVisibleError("All labels must be at least zero")
    if submission[target_levels].min().min() < 0:
        raise ParticipantVisibleError("All predictions must be at least zero")

    solution["study_id"] = solution["row_id"].apply(lambda x: x.split("_")[0])
    solution["location"] = solution["row_id"].apply(lambda x: "_".join(x.split("_")[1:]))
    solution["condition"] = solution["row_id"].apply(get_condition)

    del solution[row_id_column_name]
    del submission[row_id_column_name]
    assert sorted(submission.columns) == sorted(target_levels)

    submission["study_id"] = solution["study_id"]
    submission["location"] = solution["location"]
    submission["condition"] = solution["condition"]

    condition_losses = {}
    condition_weights = []
    for condition in ["spinal", "foraminal", "subarticular"]:
        condition_indices = solution.loc[solution["condition"] == condition].index.values
        condition_loss = sklearn.metrics.log_loss(
            y_true=solution.loc[condition_indices, target_levels].values,
            y_pred=submission.loc[condition_indices, target_levels].values,
            sample_weight=solution.loc[condition_indices, "sample_weight"].values,
        )
        condition_losses[condition] = condition_loss
        condition_weights.append(1)

    any_severe_spinal_labels = pd.Series(
        solution.loc[solution["condition"] == "spinal"].groupby("study_id")["severe"].max()
    )
    any_severe_spinal_weights = pd.Series(
        solution.loc[solution["condition"] == "spinal"].groupby("study_id")["sample_weight"].max()
    )
    any_severe_spinal_predictions = pd.Series(
        submission.loc[submission["condition"] == "spinal"].groupby("study_id")["severe"].max()
    )
    any_severe_spinal_loss = sklearn.metrics.log_loss(
        y_true=any_severe_spinal_labels, y_pred=any_severe_spinal_predictions, sample_weight=any_severe_spinal_weights
    )
    condition_losses["any_severe_spinal"] = any_severe_spinal_loss
    condition_weights.append(any_severe_scalar)
    return condition_losses
