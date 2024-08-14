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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, train: pd.DataFrame, df: pd.DataFrame, img_dir: Path, size_ratio: int, transforms: A.Compose):
        self.train = train
        self.df = df
        self.img_dir = img_dir
        self.index = self.train.index
        self.transforms = transforms
        self.size_ratio = size_ratio

    def __len__(self):
        return len(self.index)

    def get_target(self, idx):
        if self.train is None:
            return None
        return self.train.loc[idx].apply(torch.tensor).to_dict()

    def get_keypoint(self, idx):
        return self.df.loc[idx].values

    def __getitem__(self, index):
        idx = self.index[index]
        study_id, series_id, instance_number, *_ = idx
        img_path = utils.get_image_path(study_id, series_id, instance_number, self.img_dir, ".png")
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        keypoint = self.get_keypoint(idx)
        patch = get_patch(img, keypoint, self.size_ratio)
        X = self.transforms(image=patch)["image"]
        target = self.get_target(idx)

        return X, target


def get_patch(img, keypoint, size_ratio):
    h, w = img.shape
    half_side = max(h, w) // size_ratio
    x, y = keypoint
    x, y = (int(x * img.shape[1]), int(y * img.shape[0]))
    xmin, ymin, xmax, ymax = x - half_side, y - half_side, x + half_side, y + half_side
    xmin, ymin, xmax, ymax = max(xmin, 0), max(ymin, 0), min(xmax, w), min(ymax, h)
    patch = img[ymin:ymax, xmin:xmax]
    return patch


def get_transforms(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize((0.485,), (0.229,)),
            ToTensorV2(),
        ]
    )


def get_aug_transforms(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(p=0.5),
            A.Affine(rotate=30, shear=25, translate_percent=0.2, scale=(0.8, 1.2), p=0.3),
            A.Perspective(0.2),
            A.GaussNoise(var_limit=0.05, mean=0, p=0.2),
            A.MotionBlur(blur_limit=(3, 7), p=0.2),
            A.Normalize((0.485,), (0.229,)),
            ToTensorV2(),
        ]
    )


def load_keypoints(
    keypoints_path: Path = constants.KEYPOINTS_PATH, desc_path: Path = constants.DESC_PATH
) -> pd.DataFrame:
    # reshape keypoints
    df = pd.read_parquet(keypoints_path)
    levels_sides = constants.LEVELS + ["left", "right"]
    levels_sides_cols = [f"{side}_{x}" for side in levels_sides for x in ("x", "y")]
    feature_cols = [c for c in df.columns if c.startswith("_f")]
    new_cols = dict(zip(feature_cols, levels_sides_cols))
    df = df.rename(columns=new_cols)
    df = df.melt(id_vars=constants.BASIC_COLS)
    df[["type", "xy"]] = df.pop("variable").str.rsplit("_", n=1, expand=True)
    df = df.pivot(values="value", index=constants.BASIC_COLS + ["type"], columns="xy").reset_index()

    # merge desc filter invalid keypoints
    desc = utils.load_desc(desc_path)
    df = df.merge(desc, how="inner", on=constants.BASIC_COLS[:2])
    is_axial = df["series_description"].eq("Axial T2")
    is_sagittal = ~is_axial
    is_levels = df["type"].isin(constants.LEVELS)
    is_sides = ~is_levels
    df = df[(is_axial & is_sides) | (is_sagittal & is_levels)]
    df = df.drop(columns="series_description").reset_index(drop=True)

    return df


def load_this_train(train_path: Path = constants.TRAIN_PATH) -> pd.DataFrame:
    df = utils.load_train(train_path)
    df = df.melt(ignore_index=False, var_name="condition_level", value_name="severity").reset_index()
    # df[["condition", "level"]] = df.pop("condition_level").str.extractall("^(.*)_(l\d_[l|s]\d)$").droplevel(1)
    return df


def load_df(
    keypoints_path: Path = constants.KEYPOINTS_PATH,
    desc_path: Path = constants.DESC_PATH,
    coor_path: Path = constants.COOR_PATH,
    train_path: Path = constants.TRAIN_PATH,
):
    keypoints = load_keypoints(keypoints_path, desc_path)

    coor = utils.load_coor(coor_path).drop(columns=["level", "x", "y"])

    train = load_this_train(train_path)

    df = coor.merge(keypoints, how="inner", on=constants.BASIC_COLS + ["type"])
    df = df.merge(train, how="inner", on=["study_id", "condition_level"])
    df = df[constants.BASIC_COLS + ["x", "y", "condition", "severity", "type"]]

    # Remove laterality
    df["condition"] = df["condition"].str.extract("(" + "|".join(constants.CONDITIONS) + ")")[0]
    df = df.groupby(constants.BASIC_COLS + ["condition", "type"], as_index=False).agg(
        {"x": "mean", "y": "mean", "severity": "max"}
    )

    y = df.pivot(index=constants.BASIC_COLS + ["type"], columns="condition", values="severity").fillna(-1).astype(int)
    x = df[constants.BASIC_COLS + ["type", "x", "y"]].drop_duplicates().set_index(constants.BASIC_COLS + ["type"])

    return x, y


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        keypoints_path: Path = constants.KEYPOINTS_PATH,
        coor_path: Path = constants.COOR_PATH,
        train_path: Path = constants.TRAIN_PATH,
        desc_path: Path = constants.DESC_PATH,
        img_size: int = 224,
        size_ratio: int = 10,
        img_dir: Path = constants.TRAIN_IMG_DIR,
        n_splits: int = 5,
        this_split: int = 0,
        batch_size: int = 64,
        num_workers: int = 8,
    ):
        super().__init__()
        self.img_size = img_size
        self.size_ratio = size_ratio
        self.n_splits = n_splits
        self.this_split = this_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_dir = Path(img_dir)
        self.df, self.train = load_df(Path(keypoints_path), Path(desc_path), Path(coor_path), Path(train_path))
        self.train_path = train_path

    def split(self) -> Tuple[List[int], List[int]]:
        return utils.split(self.train, self.n_splits, self.this_split)

    def setup(self, stage: str):
        if stage == "fit":
            train_df, val_df = self.split()
            self.train_ds = Dataset(train_df, self.df, self.img_dir, self.size_ratio, get_aug_transforms(self.img_size))
            self.val_ds = Dataset(val_df, self.df, self.img_dir, self.size_ratio, get_transforms(self.img_size))

        if stage == "test":
            pass

        if stage == "predict":
            self.predict_ds = Dataset(None, self.df, self.meta, None, self.size_ratio, get_transforms(self.img_size))

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
