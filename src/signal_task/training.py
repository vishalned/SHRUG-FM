"""Main script for training and evaluating the model."""

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import torch
from ml4floods.data.worldfloods.configs import CHANNELS_CONFIGURATIONS
from torch import nn
from torch.optim.lr_scheduler import ConstantLR, LinearLR, SequentialLR

import wandb
from signal_task.config.settings import settings
from signal_task.dataset.floods.dataset_setup import get_dataset
from signal_task.dataset.image_dataloader import IMGDataloader
from signal_task.dataset.landslides.dataloader import LandslidesDataloader
from signal_task.models.model_ssl4eo import BaselineNet
from signal_task.testers.fire_tester import FireTester
from signal_task.trainers.img_trainer import IMGTrain
from signal_task.utils.logging_utils import get_logger
from signal_task.utils.misc import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to a config file. Specify relative to "
        "./src/signal_task/config/, e.g. dataset_vertex.toml",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
        default=1,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the raw_downstream/ directory in the mounted bucket, "
        "e.g. /gcs/2025-esl-extreme-environments-raw-data/raw_downstream/. "
        "Set to None to use the value in dataset.toml or the specified config file.",
        default=None,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to the checkpoints",
        default=None,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model",
        default="vit_small_patch16_224_unet_segmentation",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        help="Run ID for logging",
        default=None,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="Type of model to use",
        default="ssl4eo",
        choices=["ssl4eo"],
    )
    parser.add_argument(
        "--training_mode",
        type=str,
        help="",
        default="frozen_body",
        choices=["fully_finetune", "frozen_body", "random"],
    )
    parser.add_argument("--ssl_method", type=str, help="Pre-training method", default="MAE")
    parser.add_argument(
        "--disaster",
        type=str,
        help="",
        default="fire",
        choices=["fire", "flood", "landslide"],
    )
    return parser.parse_args()


@dataclass
class ModelConfig:
    """Configuration class for model training and evaluation."""

    disaster: str
    model_name: str
    mode: str
    seed: int
    stage: str
    model_path: Optional[str]
    run_id: Optional[str]
    data_path: Optional[str]
    checkpoint_path: Optional[str]
    config_file: Optional[str]
    ssl_method: Optional[str]
    model_type: str

    def __post_init__(self) -> None:
        """Initialize derived configuration parameters."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.config_file is not None:
            print(f"Loading config from {self.config_file}")
            settings.load_file(path=self.config_file, silent=False)

        self.settings = settings[self.disaster]
        print(self.settings)
        self.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.run_id = self.run_id or self.timestamp

        if self.checkpoint_path is None:
            self.checkpoint_path = self.settings["checkpoint_path"]
        else:
            self.settings["checkpoint_path"] = self.checkpoint_path

        if self.data_path is not None:
            self.settings["data_path"] = self.data_path

        if self.ssl_method is not None:
            self.settings["ssl_method"] = self.ssl_method

        if self.disaster == "flood":
            self._set_defaults()

        self.save_path = (
            Path(self.checkpoint_path)
            / self.mode
            / self.model_name.split("/")[-1]
            / self.disaster
            / self.run_id
        )

    def _set_defaults(self) -> None:
        """
        Set default values for the flood disaster settings.
        From https://github.com/spaceml-org/ml4floods/blob/e37b1a5336655244c2818e7d4a80cb7ada90e34a/ml4floods/models/config_setup.py
        """
        self.settings["model"]["input_dim"] = len(
            CHANNELS_CONFIGURATIONS[self.settings["dataloader"]["channel_configuration"]]
        )
        self.settings["dataloader"]["add_mndwi_input"] = self.settings["dataloader"].get(
            "add_mndwi_input", False
        )
        if self.settings["dataloader"]["add_mndwi_input"]:
            self.settings["model"]["input_dim"] += 1


class ModelFactory:
    """Factory class for creating models."""

    @staticmethod
    def create_model(config: ModelConfig, logger: Any) -> nn.Module:
        """Create and initialize model based on configuration.

        Args:
            config: ModelConfig: The configuration for the model.
            logger: Any: The logger to use.

        Returns:
            nn.Module: The created and initialized model.

        Raises:
            ValueError: If the model type is not supported.
        """
        if config.model_type == "ssl4eo":
            model = BaselineNet(
                disaster=config.disaster,
                model_name=config.model_name,
                ssl_method=config.ssl_method,
                freezing_body=config.mode == "frozen_body",
                logger=logger,
        )
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

        if config.mode == "random":
            model.initialize_weights()
            for param in model.parameters():
                param.requires_grad = True
            logger.info("random initialized model....")

        return cast(nn.Module, model.to(config.device))


def setup_training(
    config: ModelConfig,
    model: nn.Module,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Setup training components including optimizer and scheduler.

    Args:
        config: ModelConfig: The configuration for the model.
        model: nn.Module: The model to train.

    Returns:
        Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
            The optimizer and the learning rate scheduler.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.settings["train"]["lr"],
        weight_decay=config.settings["train"]["weight_decay"],
    )

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=config.settings["train"]["lr"] / 100,
        end_factor=1.0,
        total_iters=config.settings["train"]["warmup_epochs"],
    )
    main_scheduler = ConstantLR(
        optimizer, factor=1.0, total_iters=config.settings["train"]["num_epochs"]
    )
    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[config.settings["train"]["warmup_epochs"]],
    )

    return optimizer, lr_scheduler


def load_data(config: ModelConfig) -> IMGDataloader:
    """Load data based on configuration.

    Args:
        config: ModelConfig: The configuration for the model.

    Returns:
        IMGDataloader: The loaded data.
    """
    if config.disaster == "flood":
        dm = get_dataset(
            config.settings.dataloader,
            config.settings.data_path,
            config.seed if config.settings.dataloader.bootstrapping else None,
        )
        dm.prepare_data()
        return dm

    elif config.disaster == "fire":

        if settings.fire.dataloader.bootstrapping:
            return IMGDataloader(
                disaster="fire", data_path=config.settings.data_path, seed=str(config.seed)
            )
        return IMGDataloader(disaster="fire", data_path=config.settings.data_path)
    
    elif config.disaster == "landslide":
        
        if settings.landslide.dataloader.bootstrapping:
            return LandslidesDataloader(seed=str(config.seed))
        return LandslidesDataloader()
    else:
        raise ValueError(f"Unknown disaster type: {config.disaster}")

def setup_wandb(config: ModelConfig) -> None:
    """Initialize Weights & Biases logging if not in debug mode.

    Args:
        config: ModelConfig: The configuration for the model.
    """
    if not settings.default.debug:
        run = wandb.init(
            project="uncertainty",
            entity="fdl-extremes-25",
            name=f"{config.disaster}_{config.model_name}_{config.mode}_{config.seed}_{config.timestamp}",
            config=vars(config),
        )
        run.define_metric("validation loss", summary="min")


def train_model(
    *,
    config: ModelConfig,
    model: nn.Module,
    data: IMGDataloader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    logger: Any,
) -> Tuple[Dict, int, Path]:
    """Handle model training process.

    Args:
        config: ModelConfig: The configuration for the model.
        model: nn.Module: The model to train.
        data: IMGDataloader: The data to train on.
        optimizer: torch.optim.Optimizer: The optimizer to use.
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler: The learning rate scheduler to use.
        logger: Any: The logger to use.

    Returns:
        Tuple[Dict, int, Path]: The best model state dictionary, the best epoch, and the path to the best model.
    """
    num_epochs = 1 if settings.default.debug else config.settings["train"]["num_epochs"]
    logger.info(f"Training for {num_epochs} epochs...")
    trainer = IMGTrain(disaster=config.disaster)

    best_model_state_dict, best_epoch, _, _ = trainer.train(
        model=model,
        data=data,
        device=config.device,
        ckp_path=config.save_path,
        num_epochs=num_epochs,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss=config.settings["train"]["loss"],
        patience=config.settings["train"]["patience"],
        logger=logger,
    )

    logger.info(f"The best model is saved at epoch {best_epoch}")
    return best_model_state_dict, best_epoch, config.save_path


def test_model(
    *,
    config: ModelConfig,
    model: nn.Module,
    data: IMGDataloader,
    backup_model_path: Path,
    model_id: str,
    logger: Any,
    debug: bool = False,
) -> None:
    """Handle model testing process.

    Args:
        config: ModelConfig: The configuration for the model.
        model: nn.Module: The model to test.
        data: IMGDataloader: The data to test on.
        backup_model_path: Path: The path to the model checkpoint.
        model_id: str: The model ID.
        logger: Any: The logger to use.
        debug: Whether or not to run in debug mode. Defaults to False.
    """
    tester_classes = {"fire": FireTester, "flood": FireTester, 'landslide': FireTester}

    best_model_state_dict = torch.load(backup_model_path / f"{model_id}.pth")
    logger.info(f"Begin testing {backup_model_path}...")

    msg = model.load_state_dict(best_model_state_dict)
    logger.info(msg)

    tester = tester_classes[config.disaster](disaster=config.disaster)
    tester.test(
        model=model,
        data=data,
        uncertainty=False,
        device=config.device,
        save_path=backup_model_path,
        debug=debug,
    )
    logger.info("Finish testing!")


# pylint: disable=too-many-locals
def train_and_evaluate(
    *,
    disaster: str = "fire",
    model_name: str = "vit_small_patch16_224_unet_segmentation",
    ssl_method: str = "MAE",
    mode: str = "frozen_body",
    seed: int = 1,
    stage: str = "train_test",
    debug: Optional[bool] = True,
    model_path: Optional[str] = None,
    run_id: Optional[str] = None,
    data_path: Optional[str] = None,
    config_file: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    model_type: str = "ssl4eo",
) -> None:
    """
    This function trains and evaluates a model for a specified disaster.

    Args:
        disaster (str): The type of disaster.
        model_name (str): The name of the model to be trained.
        ssl_method (str): The method of self-supervised pre-training.
        mode (str): The training mode ("fully_finetune", "frozen_body", "random").
        seed (int): The random seed for reproducibility.
        stage (str): The stage of the training process ("train_test" or "test").
        debug (Optional[bool]): Whether to run in debug mode. Defaults to True.
        model_path (Optional[str]): The path to the model for testing.
        run_id (Optional[str]): The run ID for logging. If None, a timestamp will be used.
        data_path (Optional[str]): The path to the dataset. If None, it will use the default path from settings.
        config_file (Optional[str]): The path to the configuration file.
        checkpoint_path (Optional[str]): The path to the model checkpoint for testing.
        model_type (str): The type of model to use ("ssl4eo").

    Raises:
        ValueError: If the stage is not supported.
    """
    # Initialize configuration
    config = ModelConfig(
        disaster=disaster,
        model_name=model_name,
        ssl_method=ssl_method,
        mode=mode,
        seed=seed,
        stage=stage,
        model_path=model_path,
        run_id=run_id,
        data_path=data_path,
        checkpoint_path=checkpoint_path,
        config_file=config_file,
        model_type=model_type,
    )

    # Setup environment
    torch.set_num_threads(8)
    set_seed(config.seed)
    os.makedirs(config.save_path, exist_ok=True)
    #setup_wandb(config)

    save_path = (
        Path(config.checkpoint_path) / mode / model_name.split("/")[-1] / disaster / config.run_id
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    log_path = save_path / "training.log"
    if os.path.exists(log_path):
        os.remove(log_path)
    logger = get_logger(log_dir=save_path)

    if settings.default.debug:
        logger.info("Running in DEBUG mode!")
    else:
        wandb.init(
            project="uncertainty",  # your project name
            entity="fdl-extremes-25",
            name=f"{disaster}_{model_name}_{mode}_{config.seed}_{config.run_id}",
        )

    logger.info(f"{config.settings}")
    logger.info(f"Seed: {config.seed}")
    logger.info(f"Using device: {config.device}")
    logger.info(f"Using model type: {config.model_type}")

    # Create model and setup training components
    model = ModelFactory.create_model(config, logger)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total number of parameters in the model: {total_params}")

    # Load data
    data = load_data(config)

    if config.stage == "train_test":
        logger.info("Setting up training optimizer & LR scheduler")
        optimizer, lr_scheduler = setup_training(
            config=config,
            model=model,
        )
        logger.info("Starting training...")
        _, best_epoch, backup_model_path = train_model(
            config=config,
            model=model,
            data=data,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            logger=logger,
        )

        logger.info(f"The best model is saved at epoch {best_epoch}")
        # Rename the best_model_state_dict with its epoch and the current time
        now = datetime.now()
        # Format the date and time as a string: "year-month-day-hour-minutes"
        now.strftime("%Y-%m-%d-%H-%M")
        model_id = "best_model"  # current
        backup_model_path = save_path

    elif stage == "test":
        backup_model_path = Path(model_path)
        model_id = [
            file_name for file_name in os.listdir(backup_model_path) if "best_model" in file_name
        ][0][:-4]
    else:
        raise ValueError(f"Unknown stage: {config.stage}")

    # Test the model
    logger.info("Running evaluation...")
    test_model(
        config=config,
        model=model,
        data=data,
        backup_model_path=backup_model_path,
        model_id=model_id,
        logger=logger,
        debug=debug,
    )

    if not settings.default.debug:
        wandb.finish()


if __name__ == "__main__":
    parsed_args = parse_args()
    MODEL_PATH = None  # pylint: disable=invalid-name
    train_and_evaluate(
        disaster=parsed_args.disaster,
        seed=parsed_args.seed,
        model_path=MODEL_PATH,
        data_path=parsed_args.data_path,
        config_file=parsed_args.config_file,
        checkpoint_path=parsed_args.checkpoint_path,
        model_name=parsed_args.model_name,
        ssl_method=parsed_args.ssl_method,
        run_id=parsed_args.run_id,
        model_type=parsed_args.model_type,
        mode=parsed_args.training_mode,
    )

    if not settings.default.debug:
        wandb.finish()
