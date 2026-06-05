"""This module contains the training and testing logic for image-based disaster response models."""

import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

import wandb
from config.settings import settings
from signal_task.models.uq_model_components.density_head import (
    LikelihoodLoss,
)
from signal_task.utils.data_processing import process_input_bands
from signal_task.utils.wandb_utils import log_sample_image


def my_f1(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculate F1 score for binary classification.

    Args:
        y_pred: Predicted labels (0 or 1).
        y_true: True labels (0 or 1).

    Returns:
        F1 score as a float.
    """

    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    print(TP, FP, FN, TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    # Calculate F1 Score
    my_f1_score = 2 * (precision * recall) / (precision + recall)
    return my_f1_score


class IMGTrain:
    """
    Trainer for image-based disaster response models.

    Args:
        disaster: str: The disaster type.
    """

    def __init__(self, disaster: str) -> None:
        self.disaster = disaster
        self.loss_mapping = {
            "l1": nn.L1Loss(reduction="mean"),
            "dice": smp.losses.DiceLoss(mode="multiclass", from_logits=True),
            "density_likelihood": LikelihoodLoss(dice=False),
        }

    def _process_batch(
        self,
        model: nn.Module,
        batch_data: Dict[str, Any],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process a single batch of data.

        Args:
            model: The model to use for inference
            batch_data: Dictionary containing batch data
            device: Device to run computation on

        Returns:
            Tuple containing processed input, target, and model output
        """
        x_hls = batch_data["x"].to(device)
        x = process_input_bands(
            x_original=x_hls, model_type=model.model_type, disaster=self.disaster
        )

        mask = batch_data.get("mask")
        x_processed = torch.cat([x, mask.to(device)], dim=1) if mask is not None else x
        y = batch_data["y"].to(device)

        # Handle special case for certain models
        coord_val = batch_data.get("spatial_coords")
        if (
            coord_val is not None
            and getattr(model, "model_name", None) == "ibm-nasa-geospatial/prithvi-2_upernet"
        ):
            logits = model((x_processed, None, coord_val))
        else:
            logits = model(x_processed)

        # Ensure output size matches target
        if logits.size()[-2:] != y.size()[-2:]:
            logits = F.interpolate(
                logits,
                size=y.size()[-2:],
                # check the note at
                # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
                mode="nearest-exact",
            )

        return x_processed, y, logits

    def _train_epoch(
        self,
        *,
        model: nn.Module,
        train_loader: Any,
        criterion: Any,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        logger: Any,
        image_logging_interval: int = 20,
    ) -> float:
        """Run one epoch of training.

        Args:
            model: Model to train
            train_loader: DataLoader for training data
            criterion: Loss function
            optimizer: Optimizer
            device: Device to run on
            logger: Logger for metrics,
            image_logging_interval: Interval for logging images

        Returns:
            Average epoch loss
        """
        model.train()
        epoch_loss = 0.0

        for batch_id, train_data in enumerate(tqdm(train_loader)):
            x, y, logits = self._process_batch(model, train_data, device)

            y_for_loss, logits_for_loss = y, logits
            if self.disaster == "flood":
                # Remove invalid pixels from training
                y_for_loss = y[y < 2]
                logits_for_loss = logits.permute(0, 2, 3, 1)[y < 2]

            optimizer.zero_grad()
            loss = criterion(logits_for_loss, y_for_loss)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if not settings.default.debug:
                wandb.log(
                    {
                        "batch loss": loss.item(),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )
                if batch_id % image_logging_interval == 0:
                    log_sample_image(x, y, logits)

            if settings.default.debug:
                logger.info("Stopping training after 1 batch in debug mode")
                break

        return epoch_loss / len(train_loader)

    def _validate(
        self,
        model: nn.Module,
        val_loader: Any,
        criterion: Any,
        device: torch.device,
        logger: Any,
    ) -> float:
        """Run validation.

        Args:
            model: Model to validate
            val_loader: DataLoader for validation data
            criterion: Loss function
            device: Device to run on

        Returns:
            Validation loss
        """
        model.eval()
        loss_val = 0.0

        with torch.no_grad():
            for val_data in val_loader:
                _, y, logits = self._process_batch(model, val_data, device)

                y_for_loss, logits_for_loss = y, logits
                if self.disaster == "flood":
                    # Remove invalid pixels from val
                    y_for_loss = y[y < 2]
                    logits_for_loss = logits.permute(0, 2, 3, 1)[y < 2]

                loss = criterion(logits_for_loss, y_for_loss)
                loss_val += loss.item()

                if settings.default.debug:
                    logger.info("Stopping validation after 1 batch in debug mode")
                    break

        return loss_val / len(val_loader)

    def _save_checkpoint(
        self,
        *,
        model: nn.Module,
        ckp_path: Path,
        epoch: int,
        is_best: bool = False,
        logger: Any = None,
    ) -> Dict[str, Any]:
        """Save model checkpoint.

        Args:
            model: Model to save
            ckp_path: Path to save checkpoint
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            logger: Logger for messages

        Returns:
            Model state dictionary
        """
        state_dict = {key: value.cpu() for key, value in model.state_dict().items()}
        prefix = "best" if is_best else "last"
        file_path = os.path.join(ckp_path, f"{prefix}_model.pth")

        with open(file_path, "wb") as f:
            torch.save(state_dict, f)
            if logger and is_best:
                logger.info(f"Saving the best model at epoch {epoch} to {file_path}....")

        return state_dict

    # pylint: disable=too-many-locals
    def train(
        self,
        *,
        model: nn.Module,
        data: Any,
        device: torch.device,
        ckp_path: Path,
        num_epochs: int,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        loss: str,
        patience: int = 20,
        logger: Any = None,
        image_logging_interval: int = 20,
    ) -> Tuple[Dict[str, Any], int, Dict[str, Any], int]:
        """Training code.

        Args:
            model: nn.Module: The model to train.
            data: Any: The data to train on.
            device: torch.device: The device to train on.
            ckp_path: Path: The path to save the model.
            num_epochs: int: The number of epochs to train for.
            optimizer: torch.optim.Optimizer: The optimizer to use.
            lr_scheduler: Any: The learning rate scheduler to use.
            loss: str: The loss function to use.
            patience: int: The number of epochs to wait before early stopping.
            logger: Any: The logger to use.
            image_logging_interval: Interval for logging images

        Returns:
            Tuple[Dict[str, Any], int, Dict[str, Any], int]: The best model state dictionary,
                the best epoch, the last model state dictionary, and the number of epochs trained.
        """
        train_loader, _ = data.train_dataloader()
        val_loader, _ = data.val_dataloader()
        criterion = self.loss_mapping[loss]

        best_loss = np.inf
        best_epoch = 0
        best_state = None
        last_state = None

        for epoch in tqdm(range(num_epochs)):
            # Training phase
            epoch_loss = self._train_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                image_logging_interval=image_logging_interval,
                logger=logger,
            )
            lr_scheduler.step()

            if not settings.default.debug:
                wandb.log(
                    {
                        "train loss": epoch_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )

            logger.info(f"Epoch {epoch} : {epoch_loss:.3f}")

            # Validation phase
            loss_val = self._validate(model, val_loader, criterion, device, logger)
            logger.info(f"Val loss {epoch} : {loss_val:.3f}")

            if not settings.default.debug:
                wandb.log({"validation loss": loss_val})

            # Save checkpoints
            last_state = self._save_checkpoint(
                model=model,
                ckp_path=ckp_path,
                epoch=epoch,
            )

            if loss_val < best_loss:
                best_loss = loss_val
                best_epoch = epoch
                best_state = self._save_checkpoint(
                    model=model,
                    ckp_path=ckp_path,
                    epoch=epoch,
                    is_best=True,
                    logger=logger,
                )
            elif epoch >= best_epoch + patience:
                logger.info("Early stopping")
                break

        return best_state, best_epoch, last_state, epoch


if __name__ == "__main__":
    from torchgeo.models import ViTSmall16_Weights

    from signal_task.dataset.image_dataloader import IMGDataloader
    from signal_task.models.model_components.ssl4eo import SSL4EO

    weights = ViTSmall16_Weights.SENTINEL2_ALL_MAE

    test_model = SSL4EO(
        output_dim=1,
        num_frames=1,
        decoder_norm="batch",
        decoder_padding="same",
        decoder_activation="relu",
        decoder_depths=[2, 2, 8, 2],
        decoder_dims=[160, 320, 640, 1280],
        embed_dim=384,
        patch_size=16,
        tubelet_size=1,
        weights=weights,
    )

    # model = BaselineNet(input_dim=4, output_dim=1, model_name=model_name)
    test_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_model = test_model.to(test_device)
    batch = {"x_local": torch.randn((2, 3, 512, 512)).to(test_device)}

    output = test_model(batch)
    print(output)
    # load data
    disaster_data = IMGDataloader(disaster="fire")

    # load trainer
    trainer = IMGTrain(disaster="fire")
