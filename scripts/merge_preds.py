from pathlib import Path
import pandas as pd
import tyro

from src import constants



def merge_preds(ckpts_dir: Path, desc_path: Path = constants.DESC_PATH):
    dfs = [pd.read_csv(path) for path in ckpts_dir.rglob("*preds.csv")]
    desc = pd.read_csv(desc_path)
    df = pd.concat(dfs, axis=0)
    sort_cols = ["study_id", "series_id", "series_description", "instance_number"]
    df = pd.merge([desc, df], on=["study_id", "series_id"]).sort_values(sort_cols)
    print(df)


if __name__ == "__main__":
    tyro.cli(merge_preds)
