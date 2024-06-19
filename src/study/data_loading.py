from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import lightning as L

from src import constants, utils
from src.study import constants as study_constants


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        study_ids: List[int],
        train_df: pd.DataFrame,
        desc_df: pd.DataFrame,
        img_size: int,
        do_hflip: bool,
        img_dir: Path,
        transforms: A.Compose
    ):
        self.study_ids = study_ids
        self.train_df = train_df
        self.desc_df = desc_df
        self.img_dir = img_dir
        self.img_size = img_size
        self.do_flip = do_hflip
        self.transforms = transforms
        self.desc_index = set(desc_df.index)

    def __len__(self):
        return len(self.study_ids)

    def get_img_paths(self, study_id, series_description):
        index = (study_id, series_description)
        if index in self.desc_index:
            img_paths = []
            for series_id in self.desc_df.loc[index]["series_id"]:
                img_dir = self.img_dir / str(study_id) / str(series_id)
                series_img_paths = sorted(img_dir.glob("*.dcm"), key=lambda x: int(x.stem))
                img_paths.extend(series_img_paths)

            if len(img_paths) <= study_constants.IMGS_PER_DESC:
                return img_paths

            else:
                return [
                    img_paths[int(i)] for i in np.linspace(0, len(img_paths) - 1, study_constants.IMGS_PER_DESC).round()
                ]

        return []

    def get_labels(self, study_id):
        d: dict = self.train_df.loc[study_id].set_index("condition_level")["severity"].to_dict()
        return {k: torch.tensor(constants.SEVERITY2LABEL.get(d.get(k), -1)) for k in constants.CONDITION_LEVEL}

    def hflip(self, X: torch.Tensor, labels: dict):
        if np.random.uniform() < 0.5:
            return X, labels

        new_ks = []
        for k in labels:
            if "right" in k:
                new_k = k.replace("right", "left")
            elif "left" in k:
                new_k = k.replace("left", "right")
            else:
                new_k = k
            new_ks.append(new_k)
        new_labels = {k: v for k, v in zip(new_ks, labels.values())}

        new_X = X.flip(-1)

        return new_X, new_labels

    def __getitem__(self, index):
        study_id = self.study_ids[index]
        target = self.get_labels(study_id)

        series_imgs = []
        for description in constants.DESCRIPTIONS:
            img_paths = self.get_img_paths(study_id, description)

            imgs = np.zeros((self.img_size, self.img_size, study_constants.IMGS_PER_DESC), dtype=np.uint8)
            for i, img_path in enumerate(img_paths):
                img = utils.load_dcm_img(img_path, add_channels=False, size=self.img_size)
                imgs[..., i] = img

            imgs = self.transforms(image=imgs)["image"]
            series_imgs.append(imgs)

        X = torch.stack(series_imgs, dim=0)

        if self.do_flip:
            X, target = self.hflip(X, target)

        return X, target


def get_transforms(img_size):
    return A.Compose(
        [
            # A.Resize(img_size, img_size),
            A.Normalize(
                mean = ((0.485, 0.456, 0.406) * study_constants.IMGS_PER_DESC)[:study_constants.IMGS_PER_DESC],
                std = ((0.229, 0.224, 0.225) * study_constants.IMGS_PER_DESC)[:study_constants.IMGS_PER_DESC],
            ),
            ToTensorV2(),
        ]
    )

A.ElasticTransform

def get_aug_transforms(img_size):
        return A.Compose(
        [
            #
            A.Normalize(
                mean = ((0.485, 0.456, 0.406) * 100)[:study_constants.IMGS_PER_DESC],
                std = ((0.229, 0.224, 0.225) * 100)[:study_constants.IMGS_PER_DESC],
            ),
            ToTensorV2(),
        ]
    )


class DataModule(L.LightningDataModule):
    def __init__(
            self,
            train_path: Path = constants.TRAIN_PATH,
            desc_path: Path = constants.DESC_PATH,
            img_dir: Path = constants.TRAIN_IMG_DIR,
            n_splits: int = 5,
            this_split: int = 0,
            img_size: int = 380,
            batch_size: int = 8,
            num_workers: int = 8,
        ):
        super().__init__()
        self.img_dir = Path(img_dir)
        self.n_splits = n_splits
        self.this_split = this_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.train_df = utils.load_train(train_path).set_index("study_id").sort_index()
        self.desc_df = pd.read_csv(desc_path, index_col=["study_id", "series_description"]).sort_index()

    def split(self) -> Tuple[List[int], List[int]]:
        strats = self.train_df["condition_level"] + "_" + self.train_df["severity"]
        groups = self.train_df.reset_index()["study_id"]
        skf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True)
        for i, (train_ids, val_ids) in enumerate(skf.split(strats, strats, groups)):
            if i == self.this_split:
                break
        return list(set(groups[train_ids])), list(set(groups[val_ids]))

    def setup(self, stage: str):
        if stage == "fit":
            train_study_ids, val_study_ids = self.split()
            self.train_ds = Dataset(
                train_study_ids,
                self.train_df,
                self.desc_df,
                self.img_size,
                True,
                self.img_dir,
                transforms=get_aug_transforms(self.img_size)
            )
            self.val_ds = Dataset(
                val_study_ids,
                self.train_df,
                self.desc_df,
                self.img_size,
                False,
                self.img_dir,
                transforms=get_transforms(self.img_size)
            )

        if stage == "test":
            pass

        if stage == "predict":
            pass

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


if __name__ == "__main__":
    dm = DataModule()
    dm.setup("fit")

    for x, y in dm.train_dataloader():
        print(x.shape)
        print({k: v.shape for k, v in y.items()})
