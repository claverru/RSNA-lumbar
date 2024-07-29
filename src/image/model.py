import torch
import timm

from src import model, constants


def get_proj(in_dim, out_dim, dropout):
    return torch.nn.Sequential(
        torch.nn.Dropout(dropout),
        torch.nn.Linear(in_dim, out_dim)
    )


class LightningModule(model.LightningModule):
    def __init__(self, arch: str = "resnet34", emb_dim: int = 32, dropout: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = 1
        self.loss_f = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 2., 4., 0.2]), ignore_index=-1)
        # self.loss_f = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 2., 4.]), ignore_index=-1)
        self.backbone = timm.create_model(arch, pretrained=True, num_classes=0, in_chans=self.in_channels).eval()
        n_feats = self.backbone.num_features

        self.levels = get_proj(n_feats, emb_dim * len(constants.LEVELS), dropout)
        self.heads = torch.nn.ModuleDict({k: get_proj(emb_dim, 4, dropout) for k in constants.CONDITIONS_COMPLETE})

        self.norm = torch.nn.InstanceNorm2d(self.in_channels)

    def forward_train(self, x):
        levels = self.forward(x)
        levels = levels.reshape(levels.shape[0], len(constants.LEVELS), -1)
        levels = {k: levels[:, i] for i, k in enumerate(constants.LEVELS)}
        outs = {}
        for level_k, level in levels.items():
            out = {}
            for cond_k, head in self.heads.items():
                out[cond_k] = head(level)
            outs[level_k] = out

        return outs

    def forward(self, x):
        x_norm = self.norm(x)
        feats = self.backbone(x_norm)
        levels = self.levels(feats)
        return levels

    def do_step(self, batch, prefix="train"):
        x, y_dict = batch
        preds_dict = self.forward_train(x)
        losses = {}
        for level in constants.LEVELS:
            for cond in constants.CONDITIONS_COMPLETE:
                k = f"{cond}_{level}"
                preds = preds_dict[level][cond]
                y = y_dict[level][cond]
                loss = torch.tensor(0.)
                if (y != -1).any():
                    loss = self.loss_f(preds, y)
                self.log(f"{prefix}_{k}", loss, on_epoch=True, prog_bar=False, on_step=False)
                losses[k] = loss
        loss = sum(losses.values()) / len(losses)
        self.log(f"{prefix}_loss", loss, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self.do_step(batch)

    def validation_step(self, batch, batch_idx):
        return self.do_step(batch, "val")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X = batch
        return self.forward(X)
