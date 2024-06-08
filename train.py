import lightning as L
from src import model, data_loading
import torch


torch.set_float32_matmul_precision("high")


if __name__ == "__main__":

    module = model.LightningModule()
    data_module = data_loading.DataModule()

    trainer = L.Trainer(
        max_epochs=30,
        accelerator="gpu",
        accumulate_grad_batches=32,
        benchmark=True,
        precision="16-mixed",
        callbacks=[
            L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            L.pytorch.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", filename="{epoch}_{val_loss:.4f}")
        ]
    )

    trainer.fit(module, data_module)
