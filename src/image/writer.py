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
        preds_df = pd.DataFrame(preds, columns=cols)#.round(2)
        print(f"Preds shape: {preds.shape}")

        df = trainer.predict_dataloaders.dataset.df

        out_df = pd.concat([df, preds_df], axis=1)

        out_df.to_parquet(out_path, index=False)
