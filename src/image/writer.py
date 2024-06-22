from pathlib import Path

from lightning.pytorch.callbacks import BasePredictionWriter
import pandas as pd
import torch


class Writer(BasePredictionWriter):
    def __init__(self):
        super().__init__("epoch")

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        out_dir = Path(trainer.logger.log_dir)
        out_path = out_dir / "preds.parquet"

        preds = torch.cat(predictions).numpy()
        cols = [f"f{i}" for i in range(preds.shape[1])]

        # NOTE: reduces significantly memory size, maybe try without it in the future
        preds_df = pd.DataFrame(preds, columns=cols).round(2)

        df = trainer.predict_dataloaders.dataset.df

        df["version"] = int(out_dir.stem.split("_")[1])

        out_df = pd.concat([df, preds_df], axis=1)

        out_df.to_parquet(out_path, index=False)
