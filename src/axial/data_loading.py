from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
import lightning as L
import numpy as np
import pandas as pd
import torch

from src import constants, utils
from src.patch.data_loading import get_aug_transforms, get_patch, get_transforms, load_keypoints
from src.sequence.data_loading import collate_fn_dict, compute_xyz_world, load_meta, predict_collate_fn_dict


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        train: pd.DataFrame,
        df: pd.DataFrame,
        meta: pd.DataFrame,
        size_ratio: int,
        transforms: A.Compose,
    ):
        self.train = train
        self.df = df
        self.meta = meta
        self.index = train.index.unique()
        self.transforms = transforms
        self.size_ratio = size_ratio
        self.study_side_index = set(meta.index)

    def __len__(self):
        return len(self.index)

    def get_target(self, idx):
        if self.train.shape[1] == 0:
            return None
        return self.train.loc[idx].apply(torch.tensor).to_dict()

    def get_patches(self, study_id, side, imgs) -> torch.Tensor:
        if (study_id, side) not in self.study_side_index:
            img = np.zeros((200, 200), dtype=np.uint8)
            patch = self.transforms(image=img)["image"][None, ...]
            return patch

        chunk = self.df.loc[(study_id, side)]
        patches = []
        for img_path, gdf in chunk.groupby(level=0):
            img = imgs[img_path]
            patches += [get_patch(img, keypoint, self.size_ratio) for keypoint in gdf.values]

        patches = [self.transforms(image=patch)["image"] for patch in patches]
        patches = torch.stack(patches, 0)
        return patches

    def get_meta(self, study_id, side):
        if (study_id, side) not in self.study_side_index:
            return torch.zeros((1, self.meta.shape[1]), dtype=torch.float)

        meta = self.meta.loc[(study_id, side)].values
        meta = torch.tensor(meta, dtype=torch.float)
        return meta

    def get_images(self, idx):
        img_paths = set(i[1] if isinstance(i, tuple) else i for i in self.df.loc[idx].index)
        return {img_path: cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in img_paths}

    def __getitem__(self, index):
        idx = self.index[index]
        target = self.get_target(idx)
        imgs = self.get_images(idx)

        study_id = idx
        X = {}
        meta = {}
        mask = {}
        for side in ("left", "right"):
            X[side] = self.get_patches(study_id, side, imgs)
            meta[side] = self.get_meta(study_id, side)
            mask[side] = torch.tensor([False] * len(X[side]))

        if target is None:
            return X, meta, mask

        return X, meta, mask, target


def load_levels(levels_path: Path = constants.LEVELS_PATH):
    df = pd.read_parquet(levels_path)
    probas_cols = [c for c in df.columns if c.startswith("probas")]
    df = df[constants.BASIC_COLS + probas_cols]
    df = df.sort_values(constants.BASIC_COLS)
    return df, probas_cols


def load_df(
    keypoints_path: Path = constants.KEYPOINTS_PATH,
    levels_path: Path = constants.LEVELS_PATH,
    desc_path: Path = constants.DESC_PATH,
    meta_path: Path = constants.META_PATH,
    img_dir: Path = constants.TRAIN_IMG_DIR,
):
    keypoints = load_keypoints(keypoints_path, desc_path)

    levels, probas_cols = load_levels(levels_path)
    meta = load_meta(meta_path)

    # merge levels and filter
    keypoints = keypoints[keypoints["type"].isin(["left", "right"])]
    keypoints["type"] = keypoints["type"].map({"right": "left", "left": "right"})
    df = pd.merge(levels, keypoints, how="left", on=constants.BASIC_COLS)

    # add img path
    df["img_path"] = df[constants.BASIC_COLS].apply(
        lambda x: str(utils.get_image_path(*x, img_dir, suffix=".png")), axis=1
    )

    # merge meta
    df = df.merge(meta, how="inner", on=constants.BASIC_COLS)
    df = compute_xyz_world(df)

    # sort, clean and index
    df = df.sort_values(constants.BASIC_COLS)

    df = df.drop(columns=constants.BASIC_COLS[1:])
    df = df.set_index(["study_id", "type", "img_path"]).sort_index()

    x = df[["x", "y"]]
    new_meta = df[["xx", "yy", "zz", "pos_x", "pos_y", "pos_z"] + probas_cols].droplevel(2)

    print("Data loaded")

    return x, new_meta


def load_train(train_path: Path = constants.TRAIN_PATH) -> pd.DataFrame:
    df = utils.load_train(train_path)
    df = df[[c for c in df.columns if "subarticular" in c]]
    return df


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        keypoints_path: Path = constants.KEYPOINTS_PATH,
        desc_path: Path = constants.DESC_PATH,
        train_path: Path = constants.TRAIN_PATH,
        levels_path: Path = constants.LEVELS_PATH,
        meta_path: Path = constants.META_PATH,
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
        self.df, self.meta = load_df(
            Path(keypoints_path), Path(levels_path), Path(desc_path), Path(meta_path), self.img_dir
        )
        self.train = load_train(train_path)
        self.train_path = train_path

    def split(self) -> Tuple[List[int], List[int]]:
        return utils.split(self.train, self.n_splits, self.this_split)

    def setup(self, stage: str):
        if stage == "fit":
            train_df, val_df = self.split()

            self.train_ds = Dataset(train_df, self.df, self.meta, self.size_ratio, get_aug_transforms(self.img_size))
            self.val_ds = Dataset(val_df, self.df, self.meta, self.size_ratio, get_transforms(self.img_size))

        if stage == "test":
            _, val_df = self.split()

            self.test_ds = Dataset(val_df, self.df, self.meta, self.size_ratio, get_transforms(self.img_size))

        if stage == "predict":
            fake_train = self.df[[]].droplevel([1, 2])
            self.predict_ds = Dataset(fake_train, self.df, self.meta, self.size_ratio, get_transforms(self.img_size))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=lambda wid: np.random.seed(np.random.get_state()[1][0] + wid),
            collate_fn=collate_fn_dict,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=lambda wid: np.random.seed(np.random.get_state()[1][0] + wid),
            collate_fn=collate_fn_dict,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=lambda wid: np.random.seed(np.random.get_state()[1][0] + wid),
            collate_fn=collate_fn_dict,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.predict_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=lambda wid: np.random.seed(np.random.get_state()[1][0] + wid),
            collate_fn=predict_collate_fn_dict,
        )


if __name__ == "__main__":
    import yaml

    config = yaml.load(open("configs/axial.yaml"), Loader=yaml.FullLoader)
    dm = DataModule(**config["data"]["init_args"])

    dm.setup("fit")
    for X, meta, mask, labels in dm.train_dataloader():
        print()
        utils.print_tensor(X)
        utils.print_tensor(meta)
        utils.print_tensor(mask)
        utils.print_tensor(labels)
        print()
        break

    for X, meta, mask, labels in dm.val_dataloader():
        print()
        utils.print_tensor(X)
        utils.print_tensor(meta)
        utils.print_tensor(mask)
        utils.print_tensor(labels)
        print()
        break
