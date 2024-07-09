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



class HierarchyHead(torch.nn.Module):
    def __init__(self, n_feats: int, emb_dim: int, dropout: float, from_levels: bool = False, with_unseen: bool = True):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_levels = len(constants.LEVELS)
        self.n_conditions = len(constants.CONDITIONS_COMPLETE)
        self.n_severities = len(constants.SEVERITY2LABEL)
        if with_unseen:
            self.n_severities += 1

        self.from_levels = from_levels
        if not from_levels:
            self.levels = get_proj(n_feats, self.n_conditions * self.n_levels * emb_dim, dropout)
        self.condition_severity = torch.nn.Parameter(
            torch.randn(self.n_conditions, self.n_severities, emb_dim), requires_grad=True
        )

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = feats.shape[0]
        if self.from_levels:
            levels = feats
        else:
            levels = self.levels(feats).reshape(B, self.n_conditions, self.n_levels, self.emb_dim)
        out = torch.einsum("BCLD,CSD->BCLS", levels, self.condition_severity)
        return levels, out

    def forward_train(self, feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, out = self.forward(feats)
        outs = {}
        for i, c in enumerate(constants.CONDITIONS_COMPLETE):
            for j, l in enumerate(constants.LEVELS):
                k = f"{c}_{l}"
                outs[k] = out[:, i, j]
        return outs


class CoorsHead(torch.nn.Module):
    def __init__(self, n_feats: int):
        super().__init__()
        self.block = torch.nn.Sequential(get_block(n_feats, len(constants.CONDITION_LEVEL) * 2), torch.nn.Sigmoid())

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.block(feats)

    def forward_train(self, feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = feats.shape[0]
        out = self(feats).reshape(B, -1, 2)
        outs = {}
        for i, cl in enumerate(constants.CONDITION_LEVEL):
            outs[cl] = out[:, i]
        return outs


class LightningModule(model.LightningModule):
    def __init__(self, arch: str = "resnet34", emb_dim: int = 32, dropout: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.loss_f = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 2., 4., 0.2]))
        self.loss_coors_f = torch.nn.MSELoss(reduction="none")
        self.loss_plane_f = torch.nn.CrossEntropyLoss()
        backbone = timm.create_model(arch, pretrained=True, num_classes=0, in_chans=1).eval()
        n_feats = backbone.num_features
        self.backbone = torch.nn.Sequential(torch.nn.InstanceNorm2d(1), backbone)
        self.head = HierarchyHead(n_feats, emb_dim, dropout)
        self.coors_head = CoorsHead(n_feats)
        self.plane = torch.nn.Linear(n_feats, len(constants.DESCRIPTIONS))

    def forward_train(self, x):
        feats = self.backbone(x)
        outs = self.head.forward_train(feats)
        coors = self.coors_head.forward_train(feats)
        plane = self.plane(feats)
        return outs, coors, plane

    def forward(self, x): # todo coors
        feats = self.backbone(x)
        levels, out = self.head.forward(feats)
        B = feats.shape[0]
        levels = levels.reshape(B, -1)
        out = out.reshape(B, -1)
        coors = self.coors_head.forward(feats)
        plane = self.plane.forward(feats)
        result = torch.concat([feats, levels, out, coors, plane], -1)
        return result

    def do_cls_loss(self, y_true_dict, y_pred_dict):
        return sum(self.loss_f(y_pred_dict[k], y_true_dict[k]) for k in constants.CONDITION_LEVEL) / len(y_true_dict)

    def do_coors_loss(self, y_coors, pred_coors):
        y = torch.concat([y_coors[k] for k in constants.CONDITION_LEVEL], 0)
        pred = torch.concat([pred_coors[k] for k in constants.CONDITION_LEVEL], 0)
        loss = self.loss_coors_f(pred, y)[y[:, 0] != -1].mean()
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
        return self(X)
