from typing import Dict

import kornia.augmentation as K
import timm
import torch
from kornia.contrib import Lambda

from src import constants, losses, model


class Apply2DTransformsTo3D(torch.nn.Module):
    def __init__(self, transforms_2d):
        super().__init__()
        self.transforms_2d = transforms_2d

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.transforms_2d(x)
        return x.view(B, T, *x.shape[1:])


def get_transforms(is_3d=True):
    transforms_2d = torch.nn.Sequential(
        Lambda(lambda x: x.float() / 255.0),
        K.Normalize(mean=torch.tensor([0.485]), std=torch.tensor([0.229])),
    )
    if is_3d:
        return Apply2DTransformsTo3D(transforms_2d)
    return transforms_2d


def get_aug_transforms(is_3d=True, tta=False):
    transforms_2d = torch.nn.Sequential(
        Lambda(lambda x: x.float() / 255.0),
        K.RandomHorizontalFlip(p=0.5),
        # K.RandomVerticalFlip(p=0.5),
        K.RandomAffine(
            degrees=(-20, 20),
            translate=(0.12, 0.12),
            scale=(0.8, 1.2),
            shear=(-15, 15),
            p=1.0,
            padding_mode="zeros",
        ),
        K.RandomPerspective(
            distortion_scale=0.2,
            p=0.5,
        ),
        K.RandomSharpness(sharpness=0.3, p=0.5 * (not tta)),
        K.RandomEqualize(p=0.2 * (not tta)),
        K.RandomContrast(contrast=0.3, p=0.2 * (not tta)),
        K.RandomBrightness(brightness=0.3, p=0.2 * (not tta)),
        K.RandomGaussianNoise(mean=0.0, std=0.1, p=0.2 * (not tta)),
        K.RandomMotionBlur(kernel_size=(3, 7), angle=(-45.0, 45.0), direction=(-1.0, 1.0), p=0.2 * (not tta)),
        K.Normalize(mean=torch.tensor([0.485]), std=torch.tensor([0.229])),
    )
    if is_3d:
        return Apply2DTransformsTo3D(transforms_2d)
    return transforms_2d


class LightningModule(model.LightningModule):
    def __init__(
        self,
        arch,
        linear_dropout=0.2,
        pretrained=True,
        eval=True,
        do_any_severe_spinal: bool = False,
        gamma: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_loss = losses.LumbarLoss(do_any_severe_spinal, gamma=gamma)
        self.val_metric = losses.LumbarMetric(do_any_severe_spinal, gamma=gamma)
        self.norm = torch.nn.InstanceNorm2d(1)
        self.backbone = timm.create_model(arch, num_classes=0, in_chans=1, pretrained=pretrained)
        if eval:
            self.backbone = self.backbone.eval()

        self.heads = torch.nn.ModuleDict({k: model.get_proj(None, 3, linear_dropout) for k in constants.CONDITIONS})
        self.train_tf = get_aug_transforms(is_3d=False)
        self.val_tf = get_transforms(is_3d=False)

        self.maybe_restore_checkpoint()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.norm(x)
        x = self.backbone(x)
        return x

    def forward_train(self, x):
        x = self.forward(x)
        outs = {k: head(x) for k, head in self.heads.items()}
        return outs

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.train_tf(x)
        pred = self.forward_train(x)
        loss, _ = self.train_loss(y, pred)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.val_tf(x)
        pred = self.forward_train(x)
        self.val_metric.update(y, pred)

    def do_metric_on_epoch_end(self, prefix):
        loss, losses = self.val_metric.compute()
        self.val_metric.reset()
        for k, v in losses.items():
            self.log(f"{prefix}_{k}_loss", v, on_epoch=True, prog_bar=False, on_step=False)
        self.log(f"{prefix}_loss", loss, on_epoch=True, prog_bar=True, on_step=False)

    def on_validation_epoch_end(self, *args, **kwargs):
        self.do_metric_on_epoch_end("val")

    def on_test_epoch_end(self, *args, **kwargs):
        self.do_metric_on_epoch_end("test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        pred = self.forward_train(x)
        return pred
