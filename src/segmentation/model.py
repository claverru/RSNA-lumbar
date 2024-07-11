import torch
import segmentation_models_pytorch as smp

from src import model


class LightningModule(model.LightningModule):
    def __init__(
            self,
            arch: str = "unetplusplus",
            encoder_name: str = "tu-densenet201",
            encoder_weights: str = "imagenet",
            **kwargs
        ):
        super().__init__(**kwargs)
        self.in_channels = 1
        self.loss_f = torch.nn.CrossEntropyLoss()
        self.model = smp.create_model(
            arch=arch, encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=1, classes=6
        )
        self.model.encoder.eval()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, mask = batch
        pred = self.forward(x)
        loss = self.loss_f(pred, mask)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, mask = batch
        pred = self.forward(x)
        loss = self.loss_f(pred, mask)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, on_step=False)
        return loss
