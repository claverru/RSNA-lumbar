from pathlib import Path
from typing import Optional

from sklearn.model_selection import StratifiedKFold
import albumentations as A
import lightning as L
import numpy as np
import pandas as pd
import pydicom
import torch
from albumentations.pytorch import ToTensorV2

from src import constants


CLASS2LABEL = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
# df.reset_index().groupby(["study_id", "series_id"])["instance_number"].nunique().describe()
MAX_SERIES_SIZE = 10


def load_dcm_img(path: Path) -> np.ndarray:
    dicom = pydicom.read_file(path)
    data: np.ndarray = dicom.pixel_array
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    data = data[..., None].repeat(3, -1)
    return data


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, df: pd.DataFrame, img_dir: Path = constants.TRAIN_IMG_DIR, transforms: Optional[A.Compose] = None
    ):
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms
        self.index2id = {i: idx for i, idx in enumerate(self.df.index.unique())}

    def __len__(self):
        return len(self.index2id)

    def __getitem__(self, index):
        idx = self.index2id[index]
        study_id, series_id = idx
        chunk: pd.DataFrame = self.df.loc[idx]
        series_dir = self.img_dir / str(study_id) / str(series_id)
        imgs = []
        for instance_number in chunk["instance_number"].unique():
            img_path = (series_dir / str(instance_number)).with_suffix(".dcm")
            img = load_dcm_img(img_path)
            if self.transforms is not None:
                img = self.transforms(image=img)["image"]
            imgs.append(img)

        imgs = torch.stack(imgs, 0)
        x = torch.nn.functional.pad(imgs, (0, 0, 0, 0, 0, 0, 0, MAX_SERIES_SIZE - imgs.shape[0]))
        mask = torch.tensor([True] * imgs.shape[0] + [False] * (MAX_SERIES_SIZE - imgs.shape[0]))
        d = chunk.set_index("condition_level")["severity"].to_dict()
        y_true = {k: torch.tensor(CLASS2LABEL.get(d.get(k), -1)) for k in constants.CONDITION_LEVEL}
        return x, y_true, mask


def get_transforms(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def load_df(coor_path: Path = constants.COOR_PATH, train_path: Path = constants.TRAIN_PATH) -> pd.DataFrame:
    coor = pd.read_csv(coor_path)
    coor = coor.drop(columns=["x", "y"])
    norm_cond = coor.pop("condition").str.lower().str.replace(" ", "_")
    norm_level = coor.pop("level").str.lower().str.replace("/", "_")
    coor["condition_level"] = norm_cond + "_" + norm_level

    train = pd.read_csv(train_path)
    col_map = {0: "severity", "level_1": "condition_level"}
    train = train.set_index("study_id").stack().reset_index().rename(columns=col_map)
    train.name = "train"

    result = train.merge(coor, how="inner", on=["study_id", "condition_level"])

    return result.set_index(["study_id", "series_id"]).sort_index()


class DataModule(L.LightningDataModule):
    def __init__(
            self,
            coor_path: Path = constants.COOR_PATH,
            train_path: Path = constants.TRAIN_PATH,
            img_dir: Path = constants.TRAIN_IMG_DIR,
            img_size: int = 256,
            batch_size: int = 16,
            num_workers: int = 8,
        ):
        super().__init__()
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.df = load_df(coor_path, train_path)

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if stage == "fit":
            strats = (self.df["condition_level"] + self.df["severity"]).to_numpy()
            skf = StratifiedKFold(n_splits=5, shuffle=True)
            for train_ids, val_ids in skf.split(strats, strats):
                break
            train_df = self.df.iloc[train_ids]
            val_df = self.df.iloc[val_ids]
            self.train_ds = Dataset(train_df, self.img_dir, get_transforms(self.img_size))
            self.val_ds = Dataset(val_df, self.img_dir, get_transforms(self.img_size))

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
    for x, y, mask in dm.val_dataloader():
        print(x.shape)
        print(mask.shape)
        print({k: v.shape for k, v in y.items()})
        break
