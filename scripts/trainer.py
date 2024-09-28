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
        parser.add_argument('--strict_load', type=bool, default=False, help="Whether to strictly enforce that the keys in checkpoint match the model's state dict.")

    def before_instantiate_classes(self) -> None:
        self.strict_load = self.config.get('strict_load', False)
        if 'strict_load' in self.config:
            del self.config['strict_load']

    def instantiate_classes(self) -> None:
        super().instantiate_classes()
        if hasattr(self.model, 'load_state_dict'):
            original_load_state_dict = self.model.load_state_dict
            def new_load_state_dict(state_dict, *args, **kwargs):
                kwargs.pop('strict', None)
                return original_load_state_dict(state_dict, strict=self.strict_load, *args, **kwargs)
            self.model.load_state_dict = new_load_state_dict


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
        auto_configure_optimizers=False
    )
