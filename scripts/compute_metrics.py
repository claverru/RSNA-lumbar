from pathlib import Path

import pandas as pd
from sklearn.metrics import log_loss
import tyro
import numpy as np


from src import constants


SEVERITY_CLASSES = ["Normal/Mild", "Moderate", "Severe"]

CONDITIONS = ["spinal", "foraminal", "subarticular"]
DEFAULT_CKPT_DIR = Path("checkpoints/effnetb4_380_w/version_1")


def softmax(x, axis=-1, keepdims=True):
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=keepdims)


def merge_index_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index(df.index.map(lambda x: "_".join(str(i) for i in x)))
    return df

def merge_index_series(s: pd.Series) -> pd.Series:
    s = s.set_axis(s.index.map(lambda x: "_".join(str(i) for i in x)))
    return s


def inference_logic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby(["study_id", "series_id", "condition_level"])[SEVERITY_CLASSES].mean()
    df = df.reset_index().groupby(["study_id", "condition_level"])[SEVERITY_CLASSES].mean()
    df = merge_index_df(df)
    return df


def compute_loss(df: pd.DataFrame) -> float:
    losses = []
    for condition in CONDITIONS:
        mask = df.index.str.contains(condition)
        chunk = df.iloc[mask]
        loss = log_loss(
            chunk["label"].values, chunk[SEVERITY_CLASSES].values, sample_weight=chunk["sample_weight"].values
        )
        losses.append(loss)

    spinal_mask = df.index.str.contains("spinal")
    spinal_chunk = df.iloc[spinal_mask]
    study_id = spinal_chunk.index.map(lambda x: x.split("_")[0])

    any_severe_spinal_target = spinal_chunk.groupby(study_id)["label"].max() == 3
    any_severe_spinal_pred = spinal_chunk.groupby(study_id)["Severe"].max()
    any_severe_spinal_weights = spinal_chunk.groupby(study_id)["sample_weight"].max()


    severe_spinal_loss = log_loss(
        any_severe_spinal_target.values, any_severe_spinal_pred, sample_weight=any_severe_spinal_weights, labels=[False, True]
    )

    losses.append(severe_spinal_loss)

    print(CONDITIONS + ["any_severe_spinal"])
    print(losses)
    return sum(losses) / len(losses)

def main(ckpt_dir: Path = Path("checkpoints/effnetb4_380_w/version_1"), train_path: Path = constants.TRAIN_PATH):
    train_df = pd.read_csv(train_path, index_col="study_id").stack().map(constants.SEVERITY2LABEL)
    preds_df = pd.read_csv(ckpt_dir / constants.VAL_PREDS_NAME)

    preds_df = inference_logic(preds_df)
    train_df = merge_index_series(train_df)
    train_df.name = "label"

    df = pd.merge(train_df, preds_df, left_index=True, right_index=True)

    df[SEVERITY_CLASSES] = softmax(df[SEVERITY_CLASSES].values)
    df["sample_weight"] = df["label"].map({0: 1, 1: 2, 2: 4})

    loss = compute_loss(df)

    print(loss)


if __name__ == "__main__":
    tyro.cli(main)
