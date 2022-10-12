from operator import itemgetter
from typing import no_type_check

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from sparrow_tracky import MODA

from speed_trapv3.detection.config import Config
from speed_trapv3.detection.dataset import RetinaNetDataset
from speed_trapv3.detection.model import RetinaNet
from speed_trapv3.utils import Holdout, batch_moda


class RetinaNetTrainer(pl.LightningModule):
    """PyTorch Lightning training class."""

    def __init__(self) -> None:
        super().__init__()
        self.model = RetinaNet(n_classes=Config.n_classes)
        self.learning_rate = Config.learning_rate
        self.train_dataset = RetinaNetDataset()
        self.dev_dataset = RetinaNetDataset(Holdout.DEV)
        self.test_dataset = RetinaNetDataset(Holdout.TEST)
        self.batch_size = Config.batch_size
        self.n_workers = Config.n_workers

    @no_type_check
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Set up train dataloader."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            collate_fn=lambda x: x,
        )

    @no_type_check
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Set up dev dataloader."""
        return torch.utils.data.DataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            collate_fn=lambda x: x,
        )

    # @no_type_check
    # def test_dataloader(self) -> torch.utils.data.DataLoader:
    #     """Set up test dataloader."""
    #     return torch.utils.data.DataLoader(
    #         self.test_dataset,
    #         batch_size=self.batch_size,
    #         num_workers=self.n_workers,
    #         collate_fn=lambda x: x,
    #     )

    @no_type_check
    def training_step(self, batch: list[dict[str, torch.Tensor]], _) -> torch.Tensor:
        """Take a training step."""
        images = list(map(itemgetter("image"), batch))
        loss = self.model(images, batch)
        cls_loss, box_loss = itemgetter("classification", "bbox_regression")(loss)
        total_loss = cls_loss + box_loss
        logger_kwargs = dict(
            on_step=True, prog_bar=True, logger=True, batch_size=len(batch)
        )
        self.log("box_loss", box_loss, **logger_kwargs)
        self.log("class_loss", cls_loss, **logger_kwargs)
        self.log("total_loss", total_loss, **logger_kwargs)
        return total_loss

    @no_type_check
    def validation_step(self, batch: list[dict[str, torch.Tensor]], _) -> MODA:
        """Take a validation step."""
        images = list(map(itemgetter("image"), batch))
        results = self.model(images)
        moda = batch_moda(results, batch)
        self.log(
            "dev_moda",
            moda.value,
            prog_bar=True,
            on_epoch=True,
            batch_size=len(batch),
        )
        return moda

    @no_type_check
    def validation_epoch_end(self, outputs: list[MODA]) -> float:
        """Process validation steps."""
        moda = MODA()
        for moda_batch in outputs:
            moda += moda_batch
        return moda.value

    # @no_type_check
    # def test_step(self, batch: list[dict[str, torch.Tensor]], _) -> MODA:
    #     """Take a test step."""
    #     images = list(map(itemgetter("image"), batch))
    #     results = self.model(images)
    #     moda = batch_moda(results, batch, self.n_classes)
    #     self.log(
    #         "test_moda",
    #         moda.value,
    #         prog_bar=True,
    #         on_epoch=True,
    #         batch_size=len(batch),
    #     )
    #     return moda

    # @no_type_check
    # def test_epoch_end(self, outputs: list[MODA]) -> float:
    #     """Process test steps."""
    #     return self.validation_epoch_end(outputs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers."""
        return torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)


def train_model(
    max_epochs: int = Config.max_epochs,
) -> None:
    """Run train model command."""
    early_stop = EarlyStopping(
        "box_loss", mode="min", patience=Config.early_stopping_patience
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=Config.gpus,
        # callbacks=[early_stop],
        # overfit_batches=1,
    )
    lightning = RetinaNetTrainer()
    if Config.trained_model_path.exists():
        lightning.model.load(str(Config.trained_model_path))
    # else:
    #     lightning.model.load(str(Config.pretrained_model_path), skip_classes=True)
    trainer.fit(lightning)


def save_checkpoint(checkpoint_path: str) -> None:
    """Save a checkpoint."""
    lightning = RetinaNetTrainer()
    lightning = lightning.load_from_checkpoint(checkpoint_path)
    torch.save(lightning.model.state_dict(), Config.trained_model_path)
    print("Version trained model:")
    print(f"dvc add {Config.trained_model_path}")
