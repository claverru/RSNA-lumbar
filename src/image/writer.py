from pathlib import Path

from lightning.pytorch.callbacks import BasePredictionWriter
import pandas as pd
import torch

from src import utils


class Writer(BasePredictionWriter):
    def __init__(self):
        super().__init__("epoch")

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        out_dir = Path(trainer.logger.log_dir)
        out_path = out_dir / "preds.parquet"

        preds = utils.cat_dict_tensor(predictions)
        cols = []
        for k in preds:
            for i in range(preds[k].shape[1]):
                cols.append(f"{k}_f{i}")
        data = torch.concat(list(preds.values()), 1)

        # NOTE: reduces significantly memory size, maybe try without it in the future
        preds_df = pd.DataFrame(data, columns=cols)#.round(2)
        print(preds_df.head())
        print(preds_df.shape)

        df = trainer.predict_dataloaders.dataset.df

        out_df = pd.concat([df, preds_df], axis=1)

        out_df.to_parquet(out_path, index=False)
