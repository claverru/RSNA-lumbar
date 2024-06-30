from pathlib import Path
import pandas as pd
import tyro


def load_df(path, i):
    df = pd.read_parquet(path).drop(columns=["study_id"]).set_index(["series_id", "instance_number"])
    rename = {c: f"{c}_v{i}" for c in df.columns}
    df.rename(columns=rename, inplace=True)
    return df


def merge_preds(ckpts_dir: Path):
    dfs = []
    for i, path in enumerate(sorted(ckpts_dir.rglob("*preds.parquet"), key=lambda x: int(x.parent.stem.split("_")[1]))):
        print(path)
        dfs.append(load_df(path, i))

    df = pd.concat(dfs, axis=1).fillna(0)

    print(df.shape)
    df.to_parquet(ckpts_dir / "feats.parquet")


if __name__ == "__main__":
    tyro.cli(merge_preds)
