from lightning.pytorch.callbacks import BaseFinetuning
from torch.nn.modules.batchnorm import _BatchNorm


class CustomBackboneFinetuning(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10, unfreeze_bn=False):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self._unfreeze_bn = unfreeze_bn

    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.backbone, train_bn=False)

    @staticmethod
    def make_trainable(modules, unfreeze_backbone) -> None:
        modules = BaseFinetuning.flatten_modules(modules)
        for module in modules:
            if isinstance(module, _BatchNorm):
                if unfreeze_backbone:
                    module.track_running_stats = True
                    for param in module.parameters(recurse=False):
                        param.requires_grad = True
            else:
                for param in module.parameters(recurse=False):
                    param.requires_grad = True

    def finetune_function(self, pl_module, current_epoch, optimizer):
        if current_epoch == self._unfreeze_at_epoch:
            self.make_trainable(modules=pl_module.backbone, unfreeze_backbone=self._unfreeze_bn)
