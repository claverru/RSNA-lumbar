from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
import torch
import cv2

from src import constants


def cat_dict_tensor(dicts: List[Dict[str, torch.Tensor]], f=lambda x: torch.concat(x, dim=0)) -> Dict[str, torch.Tensor]:
    result = {}
    for k in dicts[0]:
        result[k] = f([d[k] for d in dicts])
    return result


def load_dcm_img(path: Path, add_channels: bool = True, size: Optional[int] = None) -> np.ndarray:
    dicom = pydicom.read_file(path)
    data: np.ndarray = dicom.pixel_array
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    if size is not None:
        data = cv2.resize(data, (size, size), interpolation=cv2.INTER_CUBIC)

    if add_channels:
        data = data[..., None].repeat(3, -1)
    return data


def load_train(train_path: Path = constants.TRAIN_PATH) -> pd.DataFrame:
    train = pd.read_csv(train_path)
    col_map = {0: "severity", "level_1": "condition_level"}
    train = train.set_index("study_id").stack().reset_index().rename(columns=col_map)
    train.name = "train"
    return train


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


def get_image_path(study_id, series_id, instance_number, img_dir = constants.TRAIN_IMG_DIR, suffix: str = ".dcm"):
    img_path = img_dir / str(study_id) / str(series_id) / f"{instance_number}{suffix}"
    return img_path


def scale_in(s):
    return (s - 1)/(s.max() - 1)


def load_meta(path: Path = constants.META_PATH, normalize: bool = True):
    df = pd.read_csv(path)

    if not normalize:
        return df

    for c in df.columns:
        if c in ("study_id", "series_id"):
            continue
        mean = df.groupby("series_id")[c].transform("mean")
        std = df.groupby("series_id")[c].transform("std")
        # df[f"{c}_mean"] = mean
        # df[f"{c}_std"] = std
        if c == "instance_number":
            new_c = c + "_norm" if c == "instance_number" else c
            norm = scale_in(df[c])
        else:
            new_c = c
            norm = (df[c] - mean) / (std + 1e-7)
        df[new_c] = norm
    return df


def load_desc(path: Path = constants.DESC_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def pad_sequences(sequences, padding_value=-100):
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=padding_value)


def stack(x):
    return torch.stack(x, dim=0)
