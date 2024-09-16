from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
import lightning as L
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2

from src import constants, utils
from src.patch.utils import PLANE2SPACING, Image, Keypoint, Spacing, angle_crop_size


class Dataset(torch.utils.data.Dataset):
    def __init__(self, train: pd.DataFrame, df: pd.DataFrame, img_dir: Path, img_size: int, transforms: A.Compose):
        self.train = train
        self.df = df
        self.img_dir = img_dir
        self.index = self.train.index if train is not None else df.index
        self.transforms = transforms
        self.img_size = img_size

    def __len__(self):
        return len(self.index)

    def get_target(self, idx):
        if self.train is None:
            return None
        return self.train.loc[idx].apply(torch.tensor).to_dict()

    def get_patch_params(self, idx):
        return self.df.loc[idx].values

    def __getitem__(self, index):
        idx = self.index[index]
        study_id, series_id, instance_number, plane, *_ = idx
        x, y, angle, spacing_x, spacing_y = self.get_patch_params(idx)
        spacing = Spacing(spacing_x, spacing_y)
        img = Image.from_params(study_id, series_id, instance_number, self.img_dir, ".png", spacing=spacing)
        kp = Keypoint(x, y)

        target_spacing = Spacing(PLANE2SPACING[plane], PLANE2SPACING[plane])
        img = img.resize_spacing(target_spacing)
        patch = angle_crop_size(img, kp, angle, self.img_size, plane)
        X = self.transforms(image=patch)["image"]
        target = self.get_target(idx)

        if target is None:
            return X

        return X, target


def get_transforms(img_size):
    return A.Compose(
        [
            A.LongestMaxSize(img_size, interpolation=cv2.INTER_CUBIC),
            A.PadIfNeeded(img_size, img_size, position="center", border_mode=cv2.BORDER_CONSTANT, value=0),
            ToTensorV2(),
        ]
    )


def get_angle(x1, y1, x2, y2):
    dy = y2 - y1
    dx = x2 - x1
    angle = np.degrees(np.arctan2(dy, dx))
    return angle


def mid_mask(a):
    mask = np.zeros(len(a), dtype=bool)
    mid = len(a) // 2
    if len(a) % 2 == 0:
        mask[mid - 2 : mid + 2] = True
    else:
        mask[mid - 1 : mid + 2] = True

    return mask


def smooth(s):
    a = s.values
    mask = mid_mask(a)
    value = a[mask].mean()
    a[:] = value
    return a


def load_keypoints(
    keypoints_path: Path = constants.KEYPOINTS_PATH,
    desc_path: Path = constants.DESC_PATH,
    meta_path: Path = constants.META_PATH,
) -> pd.DataFrame:
    df = pd.read_parquet(keypoints_path)
    desc = utils.load_desc(desc_path)
    meta = utils.load_meta(meta_path, with_center=False, with_relative_position=False, with_normal=False)[
        constants.BASIC_COLS + ["Rows", "Columns"]
    ]

    df = df.merge(desc, how="inner", on=constants.BASIC_COLS[:2])

    df = df.melt(id_vars=constants.BASIC_COLS + ["series_description"])

    is_axial = df["series_description"].eq("Axial T2")
    is_sagittal = ~is_axial

    has_axial = df["variable"].str.contains("axial")
    has_sagittal = ~has_axial

    df = df[(is_axial & has_axial) | (is_sagittal & has_sagittal)]

    df[["type", "coor"]] = df.pop("variable").str.rsplit("_", n=1, expand=True)[[0, 1]]
    df["type"] = df["type"].str.split("_", n=1, expand=True)[1]

    df["coor"] = df["coor"].map({"f0": "x", "f1": "y"})

    df = df.pivot(
        index=constants.BASIC_COLS + ["series_description", "type"], columns="coor", values="value"
    ).reset_index()

    df = df.merge(meta, how="inner", on=constants.BASIC_COLS)

    df["x_"] = df["x"] * df.pop("Columns")
    df["y_"] = df["y"] * df.pop("Rows")

    is_axial = df["series_description"].eq("Axial T2")
    is_sagittal = ~is_axial

    right = df["type"].str.contains("right")
    left = ~right
    df["type"] = df["type"].str.rsplit("_", n=1, expand=True)[0]
    df["type"] = df["type"].map(lambda x: {"left": "right", "right": "left"}.get(x, x))

    x1, y1 = df.loc[left, "x_"].values, df.loc[left, "y_"].values
    x2, y2 = df.loc[right, "x_"].values, df.loc[right, "y_"].values

    df["angle"] = get_angle(x1, y1, x2, y2).repeat(2)

    df = df[is_axial | (is_sagittal & right)]

    is_sagittal = ~df["series_description"].eq("Axial T2")

    for c in ("x", "y", "angle"):
        df.loc[is_sagittal, c] = df[is_sagittal].groupby(["series_id", "type"])[c].transform(smooth)

    df = df.drop(columns=["x_", "y_"])

    return df


def load_this_train(train_path: Path = constants.TRAIN_PATH) -> pd.DataFrame:
    df = utils.load_train(train_path)
    df = df.melt(ignore_index=False, var_name="condition_level", value_name="severity").reset_index()
    return df


def load_df(
    keypoints_path: Path = constants.KEYPOINTS_PATH,
    desc_path: Path = constants.DESC_PATH,
    coor_path: Path = constants.COOR_PATH,
    train_path: Path = constants.TRAIN_PATH,
    meta_path: Path = constants.META_PATH,
):
    keypoints = load_keypoints(keypoints_path, desc_path, meta_path)
    coor = utils.load_coor(coor_path).drop(columns=["level", "x", "y"])
    train = load_this_train(train_path)
    meta = utils.load_meta(meta_path, 0, False, False, False)[
        constants.BASIC_COLS + ["PixelSpacing_0", "PixelSpacing_1"]
    ]

    df = coor.merge(keypoints, how="inner", on=constants.BASIC_COLS + ["type"])
    df = df.merge(train, how="inner", on=["study_id", "condition_level"])
    df = df.merge(meta, how="inner", on=constants.BASIC_COLS)

    common_index = constants.BASIC_COLS + ["series_description", "type"]

    # set X
    x = df[common_index + ["x", "y", "angle", "PixelSpacing_0", "PixelSpacing_1"]].copy().drop_duplicates()
    x = x.set_index(common_index)

    # set Y
    y = df[common_index + ["condition", "severity"]].copy().drop_duplicates()
    y = y.set_index(common_index)

    # Remove laterality
    y["condition"] = y["condition"].str.extract("(" + "|".join(constants.CONDITIONS) + ")")[0]
    y = y.groupby(level=[0, 1, 2, 3, 4]).agg({"severity": "max", "condition": "first"})

    y = y.pivot(columns="condition", values="severity").fillna(-1).astype(int)

    return x, y


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        keypoints_path: Path = constants.KEYPOINTS_PATH,
        coor_path: Path = constants.COOR_PATH,
        train_path: Path = constants.TRAIN_PATH,
        desc_path: Path = constants.DESC_PATH,
        meta_path: Path = constants.META_PATH,
        img_size: int = 224,
        img_dir: Path = constants.TRAIN_IMG_DIR,
        n_splits: int = 5,
        this_split: int = 0,
        batch_size: int = 64,
        num_workers: int = 8,
    ):
        super().__init__()
        self.img_size = img_size
        self.n_splits = n_splits
        self.this_split = this_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_dir = Path(img_dir)
        self.df, self.train = load_df(
            Path(keypoints_path), Path(desc_path), Path(coor_path), Path(train_path), Path(meta_path)
        )
        self.train_path = train_path
        self.keypoints_path = Path(keypoints_path)

    def split(self) -> Tuple[List[int], List[int]]:
        return utils.split(self.train, self.n_splits, self.this_split)

    def setup(self, stage: str):
        if stage == "fit":
            train_df, val_df = self.split()
            self.train_ds = Dataset(train_df, self.df, self.img_dir, self.img_size, get_transforms(self.img_size))
            self.val_ds = Dataset(val_df, self.df, self.img_dir, self.img_size, get_transforms(self.img_size))

        if stage == "test":
            pass

        if stage == "predict":
            df = load_keypoints(self.keypoints_path).set_index(constants.BASIC_COLS + ["type"])
            self.predict_ds = Dataset(None, df, self.img_dir, self.img_size, get_transforms(self.img_size))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=lambda wid: np.random.seed(np.random.get_state()[1][0] + wid),
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=lambda wid: np.random.seed(np.random.get_state()[1][0] + wid),
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.predict_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=lambda wid: np.random.seed(np.random.get_state()[1][0] + wid),
        )


if __name__ == "__main__":
    import yaml

    config = yaml.load(open("configs/patch.yaml"), Loader=yaml.FullLoader)
    dm = DataModule(**config["data"]["init_args"])
    dm.setup("fit")
    for X, target in dm.train_dataloader():
        print()
        print(X.shape, X.dtype)
        print({k: (v.shape, v.dtype) for k, v in target.items()})
        print()
        input()
