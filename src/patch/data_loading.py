from pathlib import Path
from typing import Optional, Tuple

import cv2
from sklearn.model_selection import StratifiedGroupKFold
import albumentations as A
import lightning as L
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2

from src import constants, utils
from src.patch import constants as patch_constants


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path = constants.TRAIN_IMG_DIR,
        size_ratio: int = 10,
        transforms: Optional[A.Compose] = None
    ):
        self.df = df
        self.img_dir = img_dir
        self.size_ratio = size_ratio
        self.transforms = transforms
        self.index2id = {i: idx for i, idx in enumerate(self.df.index.unique())}
        self.dummy_cols = df.columns[~df.columns.isin(["x", "y"])]

    def __len__(self):
        return len(self.index2id)

    def get_target(self, idx):
        return {k: torch.tensor(v) for k, v in self.df.loc[idx, self.dummy_cols].to_dict().items()}

    def get_keypoint(self, idx):
        return self.df.loc[idx, ["x", "y"]].values

    def get_patch(self, img, keypoint):
        h, w = img.shape
        half_side = max(h, w) // self.size_ratio
        x, y = keypoint
        x, y = (int(x * img.shape[1]), int(y * img.shape[0]))
        xmin, ymin, xmax, ymax = x - half_side, y - half_side, x + half_side, y + half_side
        xmin, ymin, xmax, ymax = max(xmin, 0), max(ymin, 0), min(xmax, w) , min(ymax, h)
        patch = img[ymin:ymax, xmin:xmax]
        patch = self.transforms(image=patch)["image"]
        return patch

    def __getitem__(self, index):
        idx = self.index2id[index]
        study_id, series_id, _, instance_number, _ = idx

        img_path = utils.get_image_path(study_id, series_id, instance_number, self.img_dir, suffix=".png")
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        target = self.get_target(idx)
        keypoint = self.get_keypoint(idx)
        x = self.get_patch(img, keypoint)

        return x, target


class PredictDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path = constants.TRAIN_IMG_DIR,
        transforms: Optional[A.Compose] = None
    ):
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        study_id, series_id, instance_number = self.df.iloc[index].to_list()
        img_path = utils.get_image_path(study_id, series_id, instance_number, self.img_dir, suffix=".png")
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)[..., None]
        img = self.transforms(image=img)["image"]
        return img


def get_transforms(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize((0.485, ), (0.229, )),
            ToTensorV2(),
        ]
    )


def get_aug_transforms(img_size):
    return A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=(-0.1, 0.1),
                rotate_limit=(-10, 10),
                interpolation=cv2.INTER_CUBIC,
                scale_limit=0.1,
                # border_mode=cv2.BORDER_CONSTANT,
                # value=0,
                p=0.5
            ),
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.MotionBlur(p=0.1),
            A.GaussNoise(p=0.1),
            A.Normalize((0.485, ), (0.229, )),
            ToTensorV2()
        ]
    )


def load_keypoints(
    keypoints_path: Path = constants.KEYPOINTS_PATH,
    desc_path: Path = constants.DESC_PATH,
) -> pd.DataFrame:
    df = pd.read_parquet(keypoints_path)
    desc = utils.load_desc(desc_path)
    df = desc.merge(df, on=["study_id", "series_id"])
    xy = ("x", "y")
    sagittal_cols = [f"{i}_{x}" for i in constants.LEVELS for x in xy]
    # left subarticular is on the right side of the image
    axial_cols = [f"{i}_{x}" for i in ("right", "left") for x in xy]

    new_cols = sagittal_cols + axial_cols
    index = ["study_id", "series_id", "series_description", "instance_number"]
    df = df.set_index(index)
    old_cols = df.columns
    df = df.rename(columns=dict(zip(old_cols, new_cols)))

    df = df.melt(ignore_index=False).reset_index()

    df = df[
        (df["variable"].isin(axial_cols) & df["series_description"].eq("Axial T2"))
        | (df["variable"].isin(sagittal_cols) & ~df["series_description"].eq("Axial T2"))
    ]

    df[["type", "coor"]] = df.pop("variable").str.rsplit("_", n=1, expand=True)
    df = df.set_index(index + ["type", "coor"]).unstack()["value"]
    return df


def load_df(
    keypoints_path: Path = constants.KEYPOINTS_PATH,
    desc_path: Path = constants.DESC_PATH,
    coor_path: Path = constants.COOR_PATH,
    train_path: Path = constants.TRAIN_PATH
) -> pd.DataFrame:
    coor = utils.load_coor(coor_path).drop(columns=["x", "y"])
    keypoints = load_keypoints(keypoints_path, desc_path)
    train = utils.load_train(train_path)
    desc = utils.load_desc(desc_path)

    coor["condition_level"] = coor["condition"] + "_" + coor["level"]

    df = coor.merge(train, how="inner", on=["study_id", "condition_level"]).drop(columns=["condition_level", "level"])
    df = df.merge(desc, how="inner", on=["study_id", "series_id"])

    # reduce conditions
    df["condition"] = df["condition"].str.replace("right_|left_", "", regex=True)

    severity = df.pop("severity").map(constants.SEVERITY2LABEL) + 1

    dummies = df.pop("condition").str.get_dummies()
    dummy_cols = dummies.columns
    dummies = dummies.where(dummies == 0, severity, axis=0)

    df[dummy_cols] = dummies
    df = df.groupby(["study_id", "series_id", "instance_number", "series_description", "type"], as_index=False).max()
    df[dummy_cols] -= 1
    df[df == -1] = pd.NA
    types = df["type"].unique()
    df = df.set_index(["study_id", "series_id", "series_description", "instance_number", "type"])

    new_index = [i[:-1] for i in df.index]
    new_index = [i + (t, ) for i in set(new_index) for t in types]

    df = df.reindex(new_index).reset_index()

    planes = ["Sagittal T1", "Sagittal T2/STIR", "Axial T2"]
    conds = [
        "neural_foraminal_narrowing",
        "spinal_canal_stenosis",
        "subarticular_stenosis"
    ]
    fill = pd.DataFrame(
        data=[
            [-1,  3,  3],
            [ 3, -1,  3],
            [ 3,  3, -1]
        ],
        columns=conds,
        index=planes
    )

    for plane in planes:
        for cond in dummy_cols:
            this_fill = fill.loc[plane, cond]
            # this_fill = -1
            df.loc[df["series_description"] == plane, cond] = df.loc[df["series_description"] == plane, cond].fillna(this_fill)

    df[conds] = df[dummy_cols].astype(int)
    df.loc[df["type"].isin(["right", "left"]) & ~df["series_description"].eq("Axial T2"), conds] = -1
    df.loc[~df["type"].isin(["right", "left"]) & df["series_description"].str.contains("Axial T2"), conds] = -1

    df = df.set_index(["study_id", "series_id", "series_description", "instance_number", "type"])
    df = df[df.mean(1) != -1]
    df = df.merge(keypoints, how="inner", left_index=True, right_index=True)

    return df



class DataModule(L.LightningDataModule):
    def __init__(
            self,
            keypoints_path: Path = constants.KEYPOINTS_PATH,
            desc_path: Path = constants.DESC_PATH,
            coor_path: Path = constants.COOR_PATH,
            train_path: Path = constants.TRAIN_PATH,
            img_dir: Path = constants.TRAIN_IMG_DIR,
            size_ratio: int = 10,
            n_splits: int = 5,
            this_split: int = 0,
            img_size: int = 256,
            batch_size: int = 64,
            num_workers: int = 8,
        ):
        super().__init__()
        self.img_dir = Path(img_dir)
        self.size_ratio = size_ratio
        self.n_splits = n_splits
        self.this_split = this_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.df = load_df(keypoints_path, desc_path, coor_path, train_path)

    def prepare_data(self):
        pass

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cond_cols = self.df.columns[~self.df.columns.isin(["x", "y"])]
        strats = self.df[cond_cols].astype(str).sum(1)
        groups = self.df.reset_index()["study_id"]
        skf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True)
        for i, (train_ids, val_ids) in enumerate(skf.split(strats, strats, groups)):
            if i == self.this_split:
                break
        return self.df.iloc[train_ids], self.df.iloc[val_ids]

    def setup(self, stage: str):
        if stage == "fit":
            train_df, val_df = self.split()
            self.train_ds = Dataset(train_df, self.img_dir, self.size_ratio, transforms=get_aug_transforms(self.img_size))
            self.val_ds = Dataset(val_df, self.img_dir, self.size_ratio, transforms=get_transforms(self.img_size))

        if stage == "test":
            pass

        if stage == "predict":
            train_df, _ = self.split()
            imgs_df = utils.get_images_df(self.img_dir)
            imgs_df = imgs_df[~imgs_df.set_index(train_df.index.names).index.isin(train_df.index)].reset_index(drop=True)
            self.predict_ds = PredictDataset(imgs_df, self.img_dir, transforms=get_transforms(self.img_size))

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

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.predict_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    import yaml

    config = yaml.load(open("configs/patch.yaml"), Loader=yaml.FullLoader)
    dm = DataModule(**config["data"]["init_args"])
    dm.setup("fit")
    for x, y in dm.train_dataloader():
        print(x.shape, x.dtype)
        print({k: (v.shape, v.dtype) for k, v in y.items()})
        break
