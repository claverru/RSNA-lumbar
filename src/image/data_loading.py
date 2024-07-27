from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from sklearn.model_selection import StratifiedKFold
import albumentations as A
import lightning as L
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2

from src import constants, utils


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path = constants.TRAIN_IMG_DIR,
        train: bool = False,
        transforms: Optional[A.Compose] = None
    ):
        self.df = df
        self.img_dir = img_dir
        self.train = train
        self.transforms = transforms
        self.index2id = {i: idx for i, idx in enumerate(self.df.index.unique())}

    def __len__(self):
        return len(self.index2id)

    def get_target(self, chunk: pd.DataFrame):
        return chunk.set_index("type").map(torch.tensor).to_dict(orient="index")

    def __getitem__(self, index):
        idx = self.index2id[index]
        study_id, series_id, plane, instance_number = idx
        chunk: pd.DataFrame = self.df.loc[idx]
        img_path = utils.get_image_path(study_id, series_id, instance_number, self.img_dir, suffix=".png")
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        x = self.transforms(image=img)["image"]
        target = self.get_target(chunk)
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


def get_train_transforms(img_size):
    return A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=(-0.1, 0.1),
                rotate_limit=(-20, 20),
                scale_limit=0.1,
                interpolation=cv2.INTER_CUBIC,
                # border_mode=cv2.BORDER_CONSTANT,
                # value=0,
                p=0.5
            ),
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.MotionBlur(p=0.2),
            A.GaussNoise(p=0.2),
            A.Normalize((0.485, ), (0.229, )),
            ToTensorV2()
        ]
    )


def load_df(
    coor_path: Path = constants.COOR_PATH,
    train_path: Path = constants.TRAIN_PATH,
    desc_path: Path = constants.DESC_PATH
) -> pd.DataFrame:
    coor = utils.load_coor(coor_path).drop(columns=["x", "y"])
    train = utils.load_train(train_path)
    desc = utils.load_desc(desc_path)
    df = coor.merge(train, how="inner", on=["study_id", "condition_level"]).drop(columns=["condition_level", "level"])
    df = df.merge(desc, how="inner", on=["study_id", "series_id"])

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
    df = df.set_index(["study_id", "series_id", "series_description", "instance_number"])
    df = df.sort_values("type")
    df = df.sort_index()
    return df



class DataModule(L.LightningDataModule):
    def __init__(
            self,
            coor_path: Path = constants.COOR_PATH,
            train_path: Path = constants.TRAIN_PATH,
            desc_path: Path = constants.DESC_PATH,
            img_dir: Path = constants.TRAIN_IMG_DIR,
            n_splits: int = 5,
            this_split: int = 0,
            img_size: int = 256,
            batch_size: int = 64,
            num_workers: int = 8,
        ):
        super().__init__()
        self.img_dir = Path(img_dir)
        self.n_splits = n_splits
        self.this_split = this_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.df = load_df(coor_path, train_path, desc_path)
        self.train = utils.load_train_flatten(train_path)

    def prepare_data(self):
        pass

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        strats = self.train.astype(str).sum(1)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True)
        for i, (train_ids, val_ids) in enumerate(skf.split(strats, strats)):
            if i == self.this_split:
                break
        train_ids, val_ids  = set(self.train.index[train_ids]), set(self.train.index[val_ids])

        study_ids = self.df.reset_index()["study_id"]

        train_df = self.df[study_ids.isin(train_ids).values]
        val_df = self.df[study_ids.isin(val_ids).values]
        return train_df, val_df

    def setup(self, stage: str):
        if stage == "fit":
            train_df, val_df = self.split()
            self.train_ds = Dataset(train_df, self.img_dir, train=True, transforms=get_train_transforms(self.img_size))
            self.val_ds = Dataset(val_df, self.img_dir, train=False, transforms=get_transforms(self.img_size))

        if stage == "test":
            pass

        if stage == "predict":
            # train_df, _ = self.split()
            imgs_df = utils.get_images_df(self.img_dir)
            # imgs_df = imgs_df[~imgs_df.set_index(train_df.index.names).index.isin(train_df.index)].reset_index(drop=True)
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

    config = yaml.load(open("configs/image.yaml"), Loader=yaml.FullLoader)
    dm = DataModule(**config["data"]["init_args"])
    dm.setup("fit")
    for x, y in dm.train_dataloader():
        print(x.shape, x.dtype)
        print(y)
        break
