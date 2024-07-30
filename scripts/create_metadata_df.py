
from pathlib import Path
import pandas as pd
import pydicom
import tyro

from src import utils, constants


def get_metadata(img_path):
    dicom = pydicom.dcmread(img_path, stop_before_pixels=True)
    result = {}
    for v in dicom.values():
        v = pydicom.dataelem.DataElement_from_raw(v)
        if isinstance(v.value, (pydicom.valuerep.DSfloat, int)):
            result[v.keyword] = float(v.value)
        elif isinstance(v.value, pydicom.multival.MultiValue):
            for i, val in enumerate(v.value):
                result[f"{v.keyword}_{i}"] = float(val)
    return result


def f(study_id, series_id, instance_number, img_dir = constants.TRAIN_IMG_DIR):
    img_path = utils.get_image_path(study_id, series_id, instance_number, img_dir = img_dir)
    return get_metadata(img_path)


def main(out_path: Path, img_dir: Path = constants.TRAIN_IMG_DIR, normalize: bool = True, fillna: bool = True):
    imgs_df = utils.get_images_df()
    a = imgs_df.apply(lambda x: f(x["study_id"], x["series_id"], x["instance_number"], img_dir=img_dir), axis=1)
    meta = pd.DataFrame.from_records(list(a))
    if fillna:
        meta = meta.fillna(0.0) # only affects RescaleSlope
    df = pd.concat([imgs_df, meta], axis=1)
    if normalize:
        df = utils.normalize_meta(df)
    print(df.head())
    print(len([c for c in df.columns if c.endswith("_norm")]))
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    tyro.cli(main)
