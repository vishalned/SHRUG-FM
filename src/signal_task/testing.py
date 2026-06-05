"""Main script for training and evaluating the model."""

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import toml
import torch
from ml4floods.data.worldfloods.configs import CHANNELS_CONFIGURATIONS

import wandb
from signal_task.config.settings import settings
from signal_task.dataset.floods.dataset_setup import get_dataset
from signal_task.dataset.image_dataloader import IMGDataloader
from signal_task.dataset.landslides.dataloader import LandslidesDataloader
from signal_task.models.model_ssl4eo import BaselineNet
from signal_task.models.uq_model_components.ensemble_model import (
    EnsembleModel,
)
from signal_task.testers.tester import Tester
from signal_task.utils.logging_utils import get_logger
from signal_task.utils.misc import set_seed


@dataclass
class ModelConfig:
    """Configuration class for model training and evaluation."""

    disaster: str
    model_name: str
    mode: str
    seed: int
    model_path: Optional[str]
    eval_id: Optional[str]
    data_path: Optional[str]
    model_type: str

    def __post_init__(self) -> None:
        """Initialize derived configuration parameters."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.settings = settings[self.disaster]
        self.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.eval_id = self.eval_id or self.timestamp

        if self.data_path is not None:
            self.settings["data_path"] = self.data_path
        self.save_path = Path() / settings[self.disaster]["results_path"] / self.eval_id
        if self.disaster == "flood":
            self._set_defaults()

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


def parse_args() -> Any:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--disaster", type=str, choices=["fire", "flood", "landslide"], default="flood")
    parser.add_argument("--model_id", type=str, help="Model ID", default="bootstrap")
    parser.add_argument("--model_type", type=str, help="Model type", default="ensemble")
    parser.add_argument("--fm_type", type=str, help="Foundation model type", default="ssl4eo_MOCO")
    parser.add_argument("--seed", type=int, help="Random seed", default=1)
    parser.add_argument("--data_path", type=str, help="Path to the data", default=None)
    parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoints", default=None)
    parser.add_argument("--debug", action="store_true", help="Set to debug mode.")
    args = parser.parse_args()
    return args


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
            None,  # config.seed if config.settings.dataloader.bootstrapping else None,
        )
        dm.prepare_data()
        return dm
    if config.disaster == "fire":
        if settings.fire.dataloader.bootstrapping:
            return IMGDataloader(
                disaster="fire", data_path=config.settings.data_path, seed=str(config.seed)
            )
        return IMGDataloader(disaster="fire", data_path=config.settings.data_path)

    elif config.disaster == "landslide":
        
        if settings.landslide.dataloader.bootstrapping:
            pass
        return LandslidesDataloader()
    else:
        raise ValueError(f"Unknown disaster type: {config.disaster}")
    return None  # This line will never be reached due to the assert above


# pylint: disable=too-many-locals, too-many-statements
def evaluation(
    *,
    disaster: str = "flood",
    model_name: str = "vit_small_patch16_224_unet_segmentation",
    mode: str = "frozen_body",
    ssl_method: str = "MAE",
    seed: int = 1,
    eval_id: Optional[str] = None,
    model_path: Optional[str] = None,
    data_path: Optional[str] = None,
    debug: Optional[bool] = False,
    model_type: str = "ensemble",
) -> None:
    """
    This function trains and evaluates a model for a specified disaster.

    Args:
        disaster (str): The type of disaster.
        model_name (str): The name of the model to be trained.
        mode (str): The training mode ("fully_finetune", "frozen_body", "random").
        ssl_method (str): The SSL method to use.
        seed (int): The random seed for reproducibility.
        eval_id (Optional[str]): The unique identifier for the run, used for logging and saving results.
        model_path (Optional[str]): The path to the model checkpoint for testing. If None,
            the best model will be loaded from the training process.
        data_path (Optional[str]): The path to the data.
        debug (Optional[bool]): Whether to run in debug mode.
        model_type (str): The type of model to use.
    """
    config = ModelConfig(
        disaster=disaster,
        model_name=model_name,
        mode=mode,
        seed=seed,
        model_path=model_path,
        eval_id=eval_id,
        data_path=data_path,
        model_type=model_type,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(8)

    set_seed(seed)
    # config = settings[disaster]

    if eval_id is None:
        eval_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    results_directory = Path()
    save_path = results_directory / config.settings["results_path"] / eval_id
    Path(save_path).mkdir(parents=True, exist_ok=True)

    log_path = save_path / "test.log"
    if os.path.exists(log_path):
        os.remove(log_path)
    logger = get_logger(log_dir=save_path)
    logger.info(f"{config.settings}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Using device: {device}")

    if isinstance(model_path, list):
        model = EnsembleModel(
            checkpoint_paths=model_path,
            disaster=disaster,
            model_name=model_name,
            freezing_body=True if mode == "frozen_body" else False,
            ssl_method=ssl_method,
            logger=logger,
            device=device,
        )
    else:
        # TODO Add checkpoint path for loading and device. Then run script
        # Model selection and loading
        model = BaselineNet(
            checkpoint_path=model_path,
            disaster=disaster,
            model_name=model_name,
            freezing_body=True if mode == "frozen_body" else False,
            ssl_method=ssl_method,
            logger=logger,
            device=device,
        )
        model = model.model

    target_size = config.settings["dataloader"].target_size
    model.target_size = [target_size, target_size]

    logger.info(model)
    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Log the total number of parameters
    logger.info(f"Total number of parameters in the model: {total_params}")

    model = model.to(device)

    # load data
    # disaster_data = IMGDataloader(disaster=disaster, data_path=data_path)
    disaster_data = load_data(config)

    # load trainer
    tester = Tester(disaster=disaster)

    # Testing
    tester.test(
        model=model,
        data=disaster_data,
        uncertainty=True,
        save_samples=False,
        device=device,
        save_path=save_path,
        debug=debug,
    )
    logger.info("Finish evaluating!")

    if isinstance(model_path, list):
        configuration = {"id": eval_id, "models": [str(m) for m in model_path]}
    else:
        configuration = {"id": eval_id, "model": str(model_path)}
    with open(save_path / "eval_config.toml", "w", encoding="utf-8") as f:
        toml.dump(configuration, f)


if __name__ == "__main__":
    parsed_args = parse_args()
    FM_TYPE = parsed_args.fm_type
    MODEL_TYPE = parsed_args.model_type
    MODEL_ID = parsed_args.model_id
    DISASTER = parsed_args.disaster

    model_settings = settings.trained_models[DISASTER][FM_TYPE][MODEL_TYPE]
    model_chkpts = model_settings[MODEL_ID].model_chkpt

    if isinstance(model_chkpts, list):
        model_path = [
            Path(model_settings.checkpoint_path) / model_chkpt for model_chkpt in model_chkpts
        ]
    else:
        model_path = Path(model_settings.checkpoint_path) / model_chkpts

    evaluation(
        disaster=DISASTER,
        seed=parsed_args.seed,
        model_name=model_settings.model_name,
        mode=model_settings.mode,
        ssl_method=model_settings.ssl_method,
        model_path=model_path,
        data_path=parsed_args.data_path,
        eval_id=model_settings[MODEL_ID].id,
        debug=parsed_args.debug,
    )

    if not settings.default.debug:
        wandb.finish()
