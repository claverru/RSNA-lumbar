import colorsys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from sklearn.model_selection import StratifiedGroupKFold

from src import constants


def cat_tensors(
    tensors: List[Union[torch.Tensor, dict]], f=lambda x: torch.concat(x, dim=0)
) -> Dict[str, torch.Tensor]:
    if isinstance(tensors[0], torch.Tensor):
        return f(tensors)
    result = {}
    for k in tensors[0]:
        result[k] = cat_tensors([d[k] for d in tensors], f=f)
    return result


# https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection/blob/376afb448852d4c7951458b93e171afc953500c0/2DNet/src/prepare_data.py#L4
def load_dcm_img(path: Path, add_channels: bool = True, size: Optional[int] = None) -> np.ndarray:
    dicom = pydicom.read_file(path)
    img: np.ndarray = dicom.pixel_array

    img = img.clip(np.percentile(img, 1), np.percentile(img, 99))

    img = img - np.min(img)
    img = img / np.max(img)
    img = (img * 255).astype(np.uint8)

    if size is not None:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)

    if add_channels:
        img = img[..., None].repeat(3, -1)

    return img


def load_train(
    train_path: Path = constants.TRAIN_PATH,
    fillna: int = -1,
    per_level: bool = False,
    label_map: Dict[str, int] = constants.SEVERITY2LABEL,
) -> pd.DataFrame:
    df = pd.read_csv(train_path, index_col=0)
    if label_map is not None:
        df = df.map(lambda x: constants.SEVERITY2LABEL.get(x, fillna))
    if per_level:
        df = train_study2level(df)
    return df


def train_study2level(df: pd.DataFrame) -> pd.DataFrame:
    df = df.melt(ignore_index=False, var_name="condition_level", value_name="severity")
    pat = r"^(.*)_(l\d+_[l|s]\d+)$"
    df[["condition", "level"]] = df.pop("condition_level").str.extractall(pat).droplevel(1, axis=0)
    df = df.set_index("level", append=True)
    df = df.pivot(columns="condition", values="severity")
    return df


def train_study2levelside(df: pd.DataFrame) -> pd.DataFrame:
    df = df.melt(ignore_index=False, var_name="condition_level", value_name="severity").reset_index()
    pat = r"^(right|left)?_?(.*)_(l\d+_[l|s]\d+)$"
    extracted = df.pop("condition_level").str.extractall(pat).reset_index(drop=True)
    extracted[0] = extracted[0].fillna("right_left").str.split("_")
    df[["side", "condition", "level"]] = extracted
    df = df.explode("side")
    df = df.pivot(index=["study_id", "level", "side"], columns="condition", values="severity")
    return df


def get_images_df(img_dir: Path = constants.TRAIN_IMG_DIR) -> pd.DataFrame:
    def get_record(img_path):
        return {
            "study_id": int(img_path.parent.parent.stem),
            "series_id": int(img_path.parent.stem),
            "instance_number": int(img_path.stem),
        }

    records = [get_record(path) for path in img_dir.rglob("*.[dcm png]*")]
    sort_cols = ["study_id", "series_id", "instance_number"]
    return pd.DataFrame.from_dict(records).sort_values(sort_cols).reset_index(drop=True)


def get_image_path(study_id, series_id, instance_number, img_dir=constants.TRAIN_IMG_DIR, suffix: str = ".dcm"):
    img_path = img_dir / str(study_id) / str(series_id) / f"{instance_number}{suffix}"
    return img_path


def load_desc(path: Path = constants.DESC_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def pad_sequences(sequences, padding_value=-100):
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=padding_value)


def generate_random_colors(n: int, bright: bool = True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / n, 1, brightness) for i in range(n)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    colors = [tuple([int(v * 255) for v in c]) for c in colors]
    random.shuffle(colors)
    return colors


def load_coor(coor_path: Path = constants.COOR_PATH):
    coor = pd.read_csv(coor_path)
    coor["level"] = coor["level"].str.lower().str.replace("/", "_")
    coor["condition"] = coor["condition"].str.lower().str.replace(" ", "_")
    coor["condition_level"] = coor["condition"] + "_" + coor["level"]
    coor["type"] = coor["level"].where(
        ~coor["condition"].str.contains("subarticular"), coor["condition"].str.split("_", n=1, expand=True)[0]
    )
    return coor


def split(df: pd.DataFrame, n_splits: int, this_split: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assert df.index.names[0] == "study_id"

    train = pd.read_csv(constants.TRAIN_PATH)
    train = train.melt(id_vars="study_id", var_name="condition_level", value_name="severity")

    coor = load_coor()

    train = coor.merge(train, how="inner", on=["study_id", "condition_level"])

    strats = train["condition_level"] + "_" + train["severity"].fillna("Normal/Mild")
    groups = train["study_id"]

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True)
    for i, (train_ids, val_ids) in enumerate(skf.split(strats, strats, groups)):
        if i == this_split:
            break

    train_study_ids, val_study_ids = set(groups[train_ids]), set(groups[val_ids])

    train_study_ids = sorted(train_study_ids.intersection(df.reset_index()["study_id"]))
    val_study_ids = sorted(val_study_ids.intersection(df.reset_index()["study_id"]))

    train_df = df.loc[train_study_ids]
    val_df = df.loc[val_study_ids]

    print(f"Train DataFrame: {len(train_df)}")
    print(f"Val DataFrame  : {len(val_df)}")

    return train_df, val_df


def print_tensor(tensor: Union[dict, torch.Tensor]):
    if isinstance(tensor, torch.Tensor):
        print(tensor.shape, tensor.dtype)
    else:
        for k, v in tensor.items():
            print(k, end="\t")
            print_tensor(v)


def load_meta(meta_path: Path = constants.META_PATH, fillna: float = 0):
    meta = pd.read_csv(meta_path)
    meta = meta.fillna(fillna)
    return meta
