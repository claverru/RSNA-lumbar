import torch
import timm

from src import model


class DistanceLoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.distance = torch.nn.PairwiseDistance()

    def forward(self, pred, true):
        result = self.distance(pred, true)
        if self.reduction == "mean":
            result = result.mean()
        return result


class PositionEncoding(torch.nn.Module):
    def __init__(self, n_feats: int, depth: int = 100):
        super().__init__()
        self.encoding = torch.nn.Parameter(
            torch.randn(1, n_feats, depth) / 100, requires_grad=True
        )

    def forward(self, x):
        return x + self.encoding


def get_pool(n_feats, size):
    return torch.nn.Sequential(
        torch.nn.AdaptiveMaxPool2d(size),
        torch.nn.Flatten(start_dim=2),
        PositionEncoding(n_feats, size * size),
        torch.nn.Linear(size * size, 1),
        torch.nn.Flatten()
    )


class LightningModule(model.LightningModule):
    def __init__(self, arch: str = "resnet34", img_size: int = 448, pretrained: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.in_channels = 1

        self.loss_xy = DistanceLoss(reduction="mean")

        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0, in_chans=self.in_channels).eval()
        n_feats = self.backbone.num_features

        depth = img_size // self.backbone.feature_info[-1]["reduction"]

        self.norm = torch.nn.InstanceNorm2d(self.in_channels)
        self.pool = get_pool(n_feats, depth)

        self.xy_sagittal = torch.nn.Sequential(
            torch.nn.Linear(n_feats, 5 * 2),
            torch.nn.Sigmoid()
        )

        self.xy_axial = torch.nn.Sequential(
            torch.nn.Linear(n_feats, 2 * 2),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        B = x.shape[0]
        norm_x = self.norm(x)
        feats = self.backbone.forward_features(norm_x)
        flatten_feats = self.pool(feats)

        xy_sagittal = self.xy_sagittal(flatten_feats).reshape(B, -1, 2)
        xy_axial = self.xy_axial(flatten_feats).reshape(B, -1, 2)

        return xy_sagittal, xy_axial

    def do_xy_loss(self, pred, target):
        loss = 0.0
        mask = ((target >= 0) & (target <= 1)).all(-1)
        if mask.any():
            target = target[mask]
            pred = pred[mask]
            loss = self.loss_xy(pred, target)
        return loss

    def do_step(self, batch, prefix="train"):
        x, xy_sagittal, xy_axial = batch
        pred_xy_sagittal, pred_xy_axial = self.forward(x)

        xy_sagittal_loss = self.do_xy_loss(pred_xy_sagittal, xy_sagittal)
        xy_axial_loss = self.do_xy_loss(pred_xy_axial, xy_axial)

        loss = xy_sagittal_loss + xy_axial_loss

        self.log(prefix + "_sag_loss", xy_sagittal_loss, on_epoch=True, prog_bar=True, on_step=False)
        self.log(prefix + "_ax_loss", xy_axial_loss, on_epoch=True, prog_bar=True, on_step=False)
        self.log(prefix + "_loss", loss, on_epoch=True, prog_bar=True, on_step=False)

        return loss

    def training_step(self, batch, batch_idx):
        return self.do_step(batch)

    def validation_step(self, batch, batch_idx):
        return self.do_step(batch, "val")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        B = x.shape[0]
        pred_xy_sagittal, pred_xy_axial = self.forward(x)
        result = torch.concat([pred_xy_sagittal.reshape(B, -1), pred_xy_axial.reshape(B, -1)], axis=1)
        return {"": result}
