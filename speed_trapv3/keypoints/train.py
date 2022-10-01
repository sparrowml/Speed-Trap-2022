"""Model training."""
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping

from .config import Config
from .dataset import SegmentationDataset
from .model import SegmentationModel
from .utils import Holdout


class SegmentationLightning(pl.LightningModule):
    """Trains Segmentation model."""

    def __init__(self) -> None:
        super().__init__()
        self.model = SegmentationModel()
        self.train_dataset = SegmentationDataset(Holdout.TRAIN)
        self.dev_dataset = SegmentationDataset(Holdout.DEV)

    def custom_histogram_adder(self):
        # iterating through all parameters
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Return train dataloader."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=Config.batch_size,
            num_workers=Config.num_workers,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the dev dataloader."""
        return torch.utils.data.DataLoader(
            self.dev_dataset,
            batch_size=Config.batch_size,
            num_workers=Config.num_workers,
        )

    def training_step(self, batch, batch_idx):
        """Perform training step."""
        if batch_idx == 0:
            self.reference_image = (batch["image"][0]).unsqueeze(0)
        result = self.model(batch["image"])
        loss = F.binary_cross_entropy(result["heatmaps"], batch["heatmaps"])
        relative_error = torch.norm(
            batch["keypoints"] - result["keypoints"]
        ) / torch.norm(batch["keypoints"])
        logger_kwargs = dict(
            on_step=True, prog_bar=True, logger=True, batch_size=len(batch)
        )
        self.log("train_loss", loss, **logger_kwargs)
        self.log("train_rel_error", relative_error, **logger_kwargs)
        return loss

    def makegrid(output, numrows):
        outer = torch.Tensor.cpu(output).detach()
        plt.figure(figsize=(20, 5))
        b = np.array([]).reshape(0, outer.shape[2])
        c = np.array([]).reshape(numrows * outer.shape[2], 0)
        i = 0
        j = 0
        while i < outer.shape[1]:
            img = outer[0][i]
            b = np.concatenate((img, b), axis=0)
            j += 1
            if j == numrows:
                c = np.concatenate((c, b), axis=1)
                b = np.array([]).reshape(0, outer.shape[2])
                j = 0

            i += 1
        return c

    def showActivations(self, x):
        # logging reference image
        self.logger.experiment.add_image(
            "input", torch.Tensor.cpu(x[0][0]), self.current_epoch, dataformats="HW"
        )

    def training_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average loss
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        # creating log dictionary
        # tensorboard_logs = {"loss": avg_loss}
        self.log(
            "dev_rel_error",
            avg_loss,
            prog_bar=True,
            on_epoch=True,
            logger=True,
        )
        # logging histograms
        self.custom_histogram_adder()
        self.logger.experiment.add_scalar(
            "Loss/Train", avg_loss, self.current_epoch
        )  # scalar name, y_coordinate, x_coordinate
        self.showActivations(self.reference_image)

    def validation_step(self, batch, _):
        """Perform validation step."""
        result = self.model(batch["image"])
        relative_error = torch.norm(
            batch["keypoints"] - result["keypoints"]
        ) / torch.norm(batch["keypoints"])
        return relative_error

    def validation_epoch_end(self, outputs: list[torch.Tensor]) -> None:
        """Log average relative error."""
        relative_error = sum(outputs) / max(len(outputs), 1)
        self.log(
            "dev_rel_error",
            relative_error,
            prog_bar=True,
            on_epoch=True,
            logger=True,
        )

    def configure_optimizers(self):
        """Return optimizer."""
        return torch.optim.Adagrad(self.parameters(), lr=Config.learning_rate)


def train_model(checkpoint_path: Optional[str] = None) -> None:
    """Train model command."""
    pl_model = SegmentationLightning()
    log_dir = "./"
    logger = pl.loggers.TensorBoardLogger(log_dir)
    trainer = pl.Trainer(
        resume_from_checkpoint=checkpoint_path,
        log_every_n_steps=5,
        max_epochs=Config.max_epochs,
        gpus=Config.gpus,
        callbacks=[
            EarlyStopping("dev_rel_error", patience=3),
        ],
        # profiler="pytorch",
        profiler="simple",
        logger=logger,
        # callbacks=None,
        # fast_dev_run=True,  # turn this on to verify that the training code is working with CPU!!! (including validation)
        # overfit_batches=1,  # turn this on to check if the loss is working properly.
    )
    trainer.fit(pl_model)


def save_checkpoint(checkpoint_path: str) -> None:
    """Save checkpoint."""
    lightning = SegmentationLightning()
    lightning = lightning.load_from_checkpoint(checkpoint_path)
    torch.save(lightning.model.state_dict(), Config.trained_model_path)
