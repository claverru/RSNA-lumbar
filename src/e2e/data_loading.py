from typing import List, Optional, Tuple

from pathlib import Path
import albumentations as A
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import torch
from albumentations.pytorch import ToTensorV2
import lightning as L

from src import constants, utils
from src.e2e import constants as e2e_constants


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        study_ids: List[int],
        df: pd.DataFrame,
        train: pd.DataFrame,
        img_size: int,
        transforms: A.Compose
    ):
        self.df = df
        self.train = train
        self.study_ids = study_ids
        self.img_size = img_size
        self.transforms = transforms
        self.df_index = set(list(df.index))
        self.df_f = len(df.columns) - 1

    def __len__(self):
        return len(self.study_ids)

    def get_labels(self, study_id):
        d: dict = self.train.loc[study_id].set_index("condition_level")["severity"].to_dict()
        return {k: torch.tensor(constants.SEVERITY2LABEL.get(d.get(k), -1)) for k in constants.CONDITION_LEVEL}

    def get_chunk(self, study_id, description) -> pd.DataFrame:
        index = (study_id, description)
        if index in self.df_index:
            chunk = self.df.loc[index]
            L = len(chunk)
            N = e2e_constants.N_PER_PLANE
            ids = np.linspace(0, L - 1, min(N, L)).round()
            return chunk.iloc[ids]

    def get_images(self, chunk) -> torch.Tensor:
        if chunk is not None:
            img_series = chunk["image_path"].apply(
                lambda x: self.transforms(image=cv2.imread(str(x), cv2.IMREAD_GRAYSCALE))["image"]
            )
            imgs = torch.stack(img_series.to_list(), 0)
        else:
            imgs = torch.zeros((1, 1, self.img_size, self.img_size), dtype=torch.float32)
        return imgs

    def get_metadata(self, chunk) -> np.ndarray:
        if chunk is not None:
            values = chunk.drop(columns="image_path").values.astype(np.float32)
        else:
            values = np.zeros((1, self.df_f), dtype=np.float32)
        return values

    def __getitem__(self, index):
        study_id = self.study_ids[index]
        labels = self.get_labels(study_id)
        X = []
        metadatas = []
        desc = []
        for i, description in enumerate(constants.DESCRIPTIONS):
            chunk = self.get_chunk(study_id, description)
            x = self.get_images(chunk)
            X.append(x)
            metadata = self.get_metadata(chunk)
            metadatas.append(torch.tensor(metadata))
            desc += [i] * len(x)

        desc = torch.tensor(desc)
        X = torch.concat(X, 0)
        metadatas = torch.concat(metadatas)

        return X, metadatas, desc, labels


def get_transforms(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize((0.485, ), (0.229, )),
            ToTensorV2(),
        ]
    )


def get_df(
    meta_path: Path = constants.META_PATH,
    desc_path: Path = constants.DESC_PATH,
    img_dir: Path = constants.TRAIN_IMG_DIR
):
    meta = utils.load_meta(meta_path)
    desc = utils.load_desc(desc_path)
    df = pd.merge(desc, meta, on=["study_id", "series_id"])
    df = df.sort_values(["study_id", "series_id", "instance_number"])
    df["image_path"] = df.apply(
        lambda x: utils.get_image_path(x["study_id"], x["series_id"], x["instance_number"], img_dir, suffix=".png"),
        axis=1
    )
    df = df.drop(columns=["instance_number", "series_id"]).set_index(["study_id", "series_description"]).sort_index()
    return df


def collate_fn(data):
    X, meta, desc, labels = zip(*data)
    X = utils.pad_sequences(X, padding_value=0)
    desc = utils.pad_sequences(desc, 3)
    meta = utils.pad_sequences(meta, 0)
    labels = utils.cat_dict_tensor(labels, torch.stack)
    return X, meta, desc, labels


class DataModule(L.LightningDataModule):
    def __init__(
            self,
            meta_path: Path = constants.META_PATH,
            desc_path: Path = constants.DESC_PATH,
            train_path: Path = constants.TRAIN_PATH,
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
        self.df = get_df(Path(meta_path), Path(desc_path), self.img_dir)
        self.train = utils.load_train(train_path).set_index("study_id").sort_index()

    def split(self) -> Tuple[List[int], List[int]]:
        strats = self.train["condition_level"] + "_" + self.train["severity"]
        groups = self.train.reset_index()["study_id"]
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
                self.df,
                self.train,
                self.img_size,
                get_transforms(self.img_size)
            )
            self.val_ds = Dataset(
                val_study_ids,
                self.df,
                self.train,
                self.img_size,
                get_transforms(self.img_size)
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
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=lambda wid: np.random.seed(np.random.get_state()[1][0] + wid),
            collate_fn=collate_fn,
        )


if __name__ == "__main__":
    import yaml

    config = yaml.load(open("configs/e2e.yaml"), Loader=yaml.FullLoader)
    dm = DataModule(**config["data"]["init_args"])
    dm.setup("fit")
    for X, meta, desc, y in dm.train_dataloader():
        print(X.shape, X.dtype)
        print(meta.shape, meta.dtype)
        print(desc.shape, desc.dtype)
        print({k: (v.shape, v.dtype) for k, v in y.items()})
        input()
