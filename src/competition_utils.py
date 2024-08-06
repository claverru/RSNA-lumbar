import numpy as np
import pandas as pd
import scipy
import sklearn

from src import constants

COMP_SEVERITIES = [c.lower().replace("/", "_") for c in constants.SEVERITY2LABEL]
F2SEVERITIES = dict(zip(("f0", "f1", "f2"), COMP_SEVERITIES))


def train2solution(df: pd.DataFrame) -> pd.DataFrame:
    df = df.melt(ignore_index=False, var_name="condition")
    df["sample_weight"] = 2 ** df["value"].clip(lower=0)
    df[["none", "normal_mild", "moderate", "severe"]] = pd.get_dummies(df.pop("value")).astype(int)[[-1, 0, 1, 2]]
    df = df[df["none"] == 0].drop(columns="none").reset_index()
    df["row_id"] = df.pop("study_id").astype(str) + "_" + df.pop("condition")
    df = df.sort_values("row_id").reset_index(drop=True)
    return df


def softmax(df):
    return scipy.special.softmax(df.values, -1)


def studypreds2submission(df: pd.DataFrame) -> pd.DataFrame:
    df = df.melt(id_vars="study_id")
    df[["condition_level", "severity"]] = df.pop("variable").str.rsplit("_", n=1, expand=True)
    df = df.pivot_table(index=["study_id", "condition_level"], columns=["severity"], values="value").reset_index()
    df = df.rename(columns={"f0": "normal_mild", "f1": "moderate", "f2": "severe"})
    df["row_id"] = df.pop("study_id").astype(str) + "_" + df.pop("condition_level")
    df[COMP_SEVERITIES] = softmax(df[COMP_SEVERITIES])
    df = df.sort_values("row_id").reset_index(drop=True)
    return df


def levelpreds2submission(df: pd.DataFrame) -> pd.DataFrame:
    df = df.melt(ignore_index=False, var_name="condition")
    df["sample_weight"] = 2 ** df["value"].clip(lower=0)
    df[["none"] + COMP_SEVERITIES] = pd.get_dummies(df.pop("value")).astype(int)[[-1, 0, 1, 2]]
    df = df[df["none"] == 0].drop(columns="none").reset_index()
    df["row_id"] = df.pop("study_id").astype(str) + "_" + df.pop("condition")
    df = df.sort_values("row_id").reset_index(drop=True)
    return df


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
