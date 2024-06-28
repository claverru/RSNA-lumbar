from typing import Optional
from lightning.pytorch.callbacks import BackboneFinetuning


class CustomBackboneFinetuning(BackboneFinetuning):
    def __init__(
        self,
        unfreeze_backbone_at_epoch: int = 10,
        multiplicative: int = 1,
        backbone_initial_ratio_lr: float = 10e-2,
        backbone_initial_lr: Optional[float] = None,
        should_align: bool = True,
        initial_denom_lr: float = 10.0,
        train_bn: bool = True,
        verbose: bool = False,
        rounding: int = 12,
    ) -> None:
        super().__init__(
            unfreeze_backbone_at_epoch,
            lambda x: multiplicative,
            backbone_initial_ratio_lr,
            backbone_initial_lr,
            should_align,
            initial_denom_lr,
            train_bn,
            verbose,
            rounding
        )
