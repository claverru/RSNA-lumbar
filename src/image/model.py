from typing import Dict, Tuple
import torch
import timm

from src import constants, model


def get_proj(in_dim, out_dim, dropout):
    return torch.nn.Sequential(
        torch.nn.Dropout(dropout),
        torch.nn.Linear(in_dim, out_dim)
    )


def get_block(in_dim, out_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, in_dim // 16),
        torch.nn.ReLU(),
        torch.nn.Linear(in_dim // 16, in_dim // 64),
        torch.nn.ReLU(),
        torch.nn.Linear(in_dim // 64, out_dim)
    )



def get_pool(size):
    return torch.nn.Sequential(
        torch.nn.AdaptiveMaxPool2d(size),
        torch.nn.Flatten(start_dim=2),
        torch.nn.Linear(size * size, 1),
        torch.nn.Flatten()
    )


class DistanceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.distance = torch.nn.PairwiseDistance()

    def forward(self, pred, true):
        return self.distance(pred, true)[true[:, 0] != -1].mean()


class HierarchyHead(torch.nn.Module):
    def __init__(
        self,
        n_feats: int,
        emb_dim: int,
        dropout: float,
        from_levels: bool = False,
        with_unseen: bool = True,
        levels_and_conditions: bool = False
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_levels = len(constants.LEVELS)
        self.n_conditions = len(constants.CONDITIONS_COMPLETE)
        self.n_severities = len(constants.SEVERITY2LABEL)
        if with_unseen:
            self.n_severities += 1

        self.from_levels = from_levels
        if not from_levels:
            if levels_and_conditions:
                self.levels = get_proj(n_feats, self.n_levels * self.n_conditions * emb_dim, dropout)
            else:
                self.levels = get_proj(n_feats, self.n_levels * emb_dim, dropout)

        if levels_and_conditions:
            self.op = "BLCD,CSD->BLCS"
            self.reshape = (-1, self.n_levels, self.n_conditions, self.emb_dim)
        else:
            self.op = "BLD,CSD->BLCS"
            self.reshape = (-1, self.n_levels, self.emb_dim)

        self.condition_severity = torch.nn.Parameter(
            torch.randn(self.n_conditions, self.n_severities, emb_dim), requires_grad=True
        )

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = feats.shape[0]
        if self.from_levels:
            levels = feats
        else:
            levels = self.levels(feats).reshape(self.reshape)
        out = torch.einsum(self.op, levels, self.condition_severity)
        return levels, out


class CoorsHead(torch.nn.Module):
    def __init__(self, n_feats: int, n_coors: int = 2):
        super().__init__()
        self.n_coors = n_coors
        self.block = torch.nn.Sequential(
            get_block(n_feats, len(constants.CONDITION_LEVEL) * n_coors),
            torch.nn.Sigmoid()
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        B = feats.shape[0]
        return self.block(feats).reshape(B, len(constants.CONDITION_LEVEL), self.n_coors) # (B, CL, XYZ)


class LightningModule(model.LightningModule):
    def __init__(self, arch: str = "resnet34", emb_dim: int = 32, dropout: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.loss_f = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 2., 4., 0.2]))
        self.loss_coors_f = DistanceLoss()
        self.loss_plane_f = torch.nn.CrossEntropyLoss()
        backbone = timm.create_model(arch, pretrained=True, num_classes=0, in_chans=1).eval()
        n_feats = backbone.num_features


        self.backbone = torch.nn.Sequential(torch.nn.InstanceNorm2d(1), backbone)

        # not being used, just for state dict compat
        self.norm = torch.nn.InstanceNorm2d(1)
        self.pool = get_pool(8)
        #

        self.head = HierarchyHead(n_feats, emb_dim, dropout)
        self.coors_head = CoorsHead(n_feats, n_coors=3)
        self.plane = torch.nn.Linear(emb_dim * len(constants.LEVELS), len(constants.DESCRIPTIONS))

    def forward_train(self, x):
        _, _, out, coor, plane = self.forward_features(x)
        outs = {}
        for i, level in enumerate(constants.LEVELS):
            for j, condition in enumerate(constants.CONDITIONS_COMPLETE):
                k = f"{condition}_{level}"
                outs[k] = out[:, i, j]
        coors = {}
        for i, cl in enumerate(constants.CONDITION_LEVEL):
            coors[cl] = coor[:, i]
        return outs, coors, plane

    def forward_features(self, x):
        B = x.shape[0]
        feats = self.backbone(x)
        levels, out = self.head(feats)
        coors = self.coors_head(feats)
        plane = self.plane(levels.reshape(B, -1))
        return feats, levels, out, coors, plane

    def forward(self, x):
        feats, levels, out, coors, plane = self.forward_features(x)
        B = x.shape[0]
        levels = levels.reshape(B, -1)
        out = out.reshape(B, -1)
        coors = coors.reshape(B, -1)
        return torch.concat([feats, levels, out, coors, plane], 1)

    def do_cls_loss(self, y_true_dict, y_pred_dict):
        return sum(self.loss_f(y_pred_dict[k], y_true_dict[k]) for k in constants.CONDITION_LEVEL) / len(y_true_dict)

    def do_coors_loss(self, y_coors, pred_coors):
        y = torch.concat([y_coors[k] for k in constants.CONDITION_LEVEL], 0)
        pred = torch.concat([pred_coors[k] for k in constants.CONDITION_LEVEL], 0)
        loss = self.loss_coors_f(pred, y)
        return loss

    def do_loss(self, y_cls, y_coors, y_plane, pred_cls, pred_coors, pred_plane, prefix="", log=True):
        cls_loss = self.do_cls_loss(y_cls, pred_cls)
        coor_loss = self.do_coors_loss(y_coors, pred_coors)
        plane_loss = self.loss_plane_f(pred_plane, y_plane)
        total_loss = cls_loss + coor_loss + plane_loss
        if log:
            self.log(prefix + "cls_loss", cls_loss, on_epoch=True, prog_bar=True, on_step=False)
            self.log(prefix + "coor_loss", coor_loss, on_epoch=True, prog_bar=True, on_step=False)
            self.log(prefix + "plane_loss", plane_loss, on_epoch=True, prog_bar=True, on_step=False)
            self.log(prefix + "loss", total_loss, on_epoch=True, prog_bar=True, on_step=False)
        return total_loss

    def training_step(self, batch, batch_idx):
        x, y_cls, y_coors, y_plane = batch
        pred_cls, pred_coors, pred_plane = self.forward_train(x)
        return self.do_loss(y_cls, y_coors, y_plane, pred_cls, pred_coors, pred_plane)

    def validation_step(self, batch, batch_idx):
        x, y_cls, y_coors, y_plane = batch
        pred_cls, pred_coors, pred_plane = self.forward_train(x)
        return self.do_loss(y_cls, y_coors, y_plane, pred_cls, pred_coors, pred_plane, "val_")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X = batch
        return self.forward(X)
