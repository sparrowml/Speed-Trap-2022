"""Model training."""
from typing import Optional

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

    def training_step(self, batch, _):
        """Perform training step."""
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
        )

    def configure_optimizers(self):
        """Return optimizer."""
        return torch.optim.Adagrad(self.parameters(), lr=Config.learning_rate)


def train_model(checkpoint_path: Optional[str] = None) -> None:
    """Train model command."""
    pl_model = SegmentationLightning()
    trainer = pl.Trainer(
        callbacks=[EarlyStopping("dev_rel_error", patience=Config.patience)],
        # callbacks=None,
        # resume_from_checkpoint=checkpoint_path,
        log_every_n_steps=5,
        max_epochs=Config.max_epochs,
        gpus=Config.gpus,
        # overfit_batches=1,
    )
    trainer.fit(pl_model)


def save_checkpoint(checkpoint_path: str) -> None:
    """Save checkpoint."""
    lightning = SegmentationLightning()
    lightning = lightning.load_from_checkpoint(checkpoint_path)
    torch.save(lightning.model.state_dict(), Config.trained_model_path)
