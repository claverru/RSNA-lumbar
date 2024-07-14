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
from src.image.data_loading import load_df


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

    def get_levels(self, chunk: pd.DataFrame):
        return chunk["level"].values

    def get_keypoints(self, chunk: pd.DataFrame):
        return chunk[["x", "y"]].values

    def get_plane_str(self, chunk: pd.DataFrame):
        return chunk["series_description"].iloc[0]

    def keypoints2xy(self, keypoints, shape):
        C, H, W = shape
        denominator = torch.tensor([[W, H]])
        return torch.tensor(keypoints) / denominator

    def __getitem__(self, index):
        idx = self.index2id[index]
        chunk: pd.DataFrame = self.df.loc[idx]
        img_path = utils.get_image_path(idx[0], idx[1], idx[2], self.img_dir, suffix=".png")
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        plane = self.get_plane_str(chunk)

        levels = self.get_levels(chunk)
        keypoints = self.get_keypoints(chunk)

        transformed = self.transforms(image=img, keypoints=keypoints)
        x = transformed["image"]
        keypoints = transformed["keypoints"] # xy

        if plane == "Axial T2":
            keypoints_axial = sorted(keypoints, key=lambda x: x[0]) # left to right
            xy_axial = self.keypoints2xy(keypoints_axial, x.shape)
            xy_saggital = torch.zeros(5, 2)
            level = torch.tensor(constants.LEVELS.index(levels[0]))
        else:
            keypoints_saggital = sorted(keypoints, key=lambda x: x[1]) # top to botton
            xy_saggital = self.keypoints2xy(keypoints_saggital, x.shape)
            xy_axial = torch.zeros(2, 2)
            level = torch.tensor(-1)

        return x, xy_saggital, xy_axial, level



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
        chunk = self.df.loc[index]
        img_path = utils.get_image_path(
            chunk["study_id"], chunk["series_id"], chunk["instance_number"], self.img_dir, suffix=".png"
        )

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        transformed = self.transforms(image=img)
        x = transformed["image"]

        return x



def get_transforms(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize((0.485, ), (0.229, )),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)
    )


def get_train_transforms(img_size):
    return A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=(-0.3, 0.3),
                rotate_limit=(-10, 10),
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5
            ),
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.1),
            A.MotionBlur(p=0.1),
            A.GaussNoise(p=0.1),
            A.Normalize((0.485, ), (0.229, )),
            ToTensorV2()
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)
    )



def get_predict_transforms(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize((0.485, ), (0.229, )),
            ToTensorV2(),
        ]
    )

def load_this_df(
    coor_path: Path = constants.COOR_PATH,
    train_path: Path = constants.TRAIN_PATH,
    desc_path: Path = constants.DESC_PATH
) -> pd.DataFrame:
    df = load_df(coor_path, train_path, desc_path)

    df["level"] = df["condition_level"].apply(lambda x: "_".join(x.split("_")[-2:]))

    size = df.groupby(df.index)["z"].transform(len)

    axial = df[(df.series_description == "Axial T2") & (size == 2)]
    axial = axial[axial.groupby(axial.index)["level"].transform(lambda x: x.nunique() == 1)]

    sagittal = df[(df.series_description != "Axial T2") & (size == 5)]
    sagittal = sagittal[sagittal.groupby(sagittal.index)["level"].transform(lambda x: x.nunique() == 5)]

    df = pd.concat([axial, sagittal], axis=0)

    return df.sort_index()


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
        self.df = load_this_df(coor_path, train_path, desc_path)

    def prepare_data(self):
        pass

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        strats = self.df["condition_level"] + "_" + self.df["severity"]
        groups = self.df.reset_index()["study_id"]
        skf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True)
        for i, (train_ids, val_ids) in enumerate(skf.split(strats, strats, groups)):
            if i == self.this_split:
                break
        return self.df.iloc[train_ids], self.df.iloc[val_ids]

    def setup(self, stage: str):
        if stage == "fit":
            train_df, val_df = self.split()
            self.train_ds = Dataset(train_df, self.img_dir, train=True, transforms=get_train_transforms(self.img_size))
            self.val_ds = Dataset(val_df, self.img_dir, train=False, transforms=get_transforms(self.img_size))

        if stage == "test":
            pass

        if stage == "predict":
            img_df = utils.get_images_df()
            self.predict_ds = PredictDataset(img_df, self.img_dir, get_predict_transforms(self.img_size))

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
    for x, xy_saggital, xy_axial, level in dm.train_dataloader():
        print(x.shape, x.dtype)
        print(xy_saggital.shape, xy_saggital.dtype)
        print(xy_axial.shape, xy_axial.dtype)
        print(level.shape, level.dtype)
        input()
