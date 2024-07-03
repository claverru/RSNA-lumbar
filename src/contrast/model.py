import torch
import timm

from src import model


def criterion(x1, x2, label, margin: float = 1.0):
    dist = torch.nn.functional.pairwise_distance(x1, x2)

    loss = (1 - label) * torch.pow(dist, 2) \
        + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    loss = torch.mean(loss)

    return loss


class LightningModule(model.LightningModule):
    def __init__(self, arch: str = "resnet34", dropout: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.loss_f = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=torch.tensor([1., 2., 4.]))
        self.backbone = timm.create_model(arch, pretrained=True, num_classes=0, in_chans=1).eval()

    def forward(self, x1, x2):
        B = x1.shape[0]
        x = torch.concat([x1, x2], 0)
        feats = self.backbone(x)
        feats1 = feats[:B]
        feats2 = feats[B:]
        return feats1, feats2

    def training_step(self, batch, batch_idx):
        x1, x2, target = batch
        pred1, pred2 = self.forward(x1, x2)
        loss = criterion(pred1, pred2, target)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, target = batch
        pred1, pred2 = self.forward(x1, x2)
        loss = criterion(pred1, pred2, target)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X = batch
        return self.backbone(X)
