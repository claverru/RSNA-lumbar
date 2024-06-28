import random

from lightning.pytorch.cli import LightningCLI
import lightning as L
import numpy as np

import torch


torch.set_float32_matmul_precision("high")


class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument('--monitor', type=str)
        parser.add_argument('--monitor_mode', type=str)


def get_random_seed():
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min
    return random.randint(min_seed_value, max_seed_value)


if __name__ == '__main__':
    cli = CustomLightningCLI(
        L.LightningModule,
        L.LightningDataModule,
        seed_everything_default=get_random_seed(),
        subclass_mode_model=True,
        subclass_mode_data=True,
        parser_kwargs={'parser_mode': 'omegaconf'},
    )
