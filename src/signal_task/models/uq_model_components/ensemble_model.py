"""Ensemble model for uncertainty quantification."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import torch
from torch import nn

from signal_task.models.model_ssl4eo import BaselineNet
from signal_task.models.model_terramind import TerraMindNet
from signal_task.models.uq_model_components.ensemble_uq_model import (
    EnsembleUQModel,
)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model initialization."""

    checkpoint_paths: List[str]
    disaster: str
    model_name: str
    model_type: str
    freezing_body: bool
    logger: Optional[object]
    device: torch.device
    ssl_method: str = "MAE"

    @classmethod
    def create(
        cls,
        *,
        checkpoint_paths: List[str],
        disaster: str = "fire",
        model_name: str = "vit_small_patch16_224_unet_segmentation",
        model_type: str = "ssl4eo",
        freezing_body: bool = True,
        ssl_method: str = "MAE",
        logger: Any = None,
        device: Optional[torch.device] = None,
    ) -> "EnsembleConfig":
        """Create an EnsembleConfig instance with default values.

        Args:
            cls: Class reference
            checkpoint_paths: List of paths to model checkpoint files (.pth files)
            disaster: The disaster type (e.g., "fire")
            model_name: The model name
            model_type: Type of model to use ("ssl4eo" or "terramind")
            freezing_body: Whether to freeze the encoder body
            ssl_method: The SSL method to use ("MAE", "DINO", "MOCO", "FGMAE")
            logger: Logger instance for logging
            device: Device to load models on (if None, uses CUDA if available)

        Returns:
            EnsembleConfig instance with the provided parameters
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls(
            checkpoint_paths=checkpoint_paths,
            disaster=disaster,
            model_name=model_name,
            model_type=model_type,
            freezing_body=freezing_body,
            ssl_method=ssl_method,
            logger=logger,
            device=device,
        )


class EnsembleModel(EnsembleUQModel):
    """
    Concrete implementation of EnsembleUQModel for downstream models.

    This class builds an ensemble from multiple model checkpoints.
    It loads each checkpoint, creates the appropriate model, and combines them
    into an ensemble for uncertainty quantification.
    """

    def __init__(
        self,
        *,
        checkpoint_paths: List[str],
        disaster: str = "fire",
        model_name: str = "vit_small_patch16_224_unet_segmentation",
        model_type: str = "ssl4eo",
        freezing_body: bool = True,
        ssl_method: str = "MAE",
        logger: Any = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the Ensemble model.

        Args:
            checkpoint_paths: List of paths to model checkpoint files (.pth files)
            disaster: The disaster type (e.g., "fire")
            model_name: The model name
            model_type: Type of model to use ("ssl4eo" or "terramind")
            freezing_body: Whether to freeze the encoder body
            ssl_method: The SSL method to use ("MAE", "DINO", "MOCO", "FGMAE")
            logger: Logger instance for logging
            device: Device to load models on (if None, uses CUDA if available)
        """
        super().__init__(ensemble_members=[])

        # Create configuration
        self.config = EnsembleConfig.create(
            checkpoint_paths=checkpoint_paths,
            disaster=disaster,
            model_name=model_name,
            model_type=model_type,
            freezing_body=freezing_body,
            ssl_method=ssl_method,
            logger=logger,
            device=device,
        )

        # Initialize shared_encoder to None. It will be loaded only once if freezing_body is True.
        self.shared_encoder = None

        # Load ensemble members from checkpoints
        ensemble_members = self._load_ensemble_members()

        # Initialize the parent EnsembleUQModel
        self.ensemble_members = nn.ModuleList(ensemble_members)

        if self.config.logger:
            self.config.logger.info(f"Initialized Ensemble with {len(ensemble_members)} members")
            self.config.logger.info(f"Checkpoint paths: {checkpoint_paths}")

    def _create_model(self) -> nn.Module:
        """Create a new model instance based on configuration.

        Returns:
            nn.Module: The created model

        Raises:
            ValueError: If the model type is invalid
        """
        if self.config.model_type == "ssl4eo":
            return BaselineNet(
                disaster=self.config.disaster,
                model_name=self.config.model_name,
                freezing_body=self.config.freezing_body,
                ssl_method=self.config.ssl_method,
                logger=self.config.logger,
            )
        if self.config.model_type == "terramind":
            return TerraMindNet(
                disaster=self.config.disaster,
                model_name=self.config.model_name,
                freezing_body=self.config.freezing_body,
                logger=self.config.logger,
            )

        raise ValueError(f"Invalid model type: {self.config.model_type}")

    def _load_checkpoint(self, checkpoint_path: str) -> dict:
        """Load and process checkpoint file."""
        checkpoint = torch.load(Path(checkpoint_path), map_location=self.config.device)

        if isinstance(checkpoint, dict):
            return checkpoint.get("state_dict", checkpoint)
        return checkpoint

    def _initialize_shared_encoder(self) -> None:
        """Initialize shared encoder if using frozen body."""
        if not self.config.freezing_body or not self.config.checkpoint_paths:
            return

        if self.config.logger:
            self.config.logger.info("Freezing body is True. Loading shared encoder once.")

        # Create a dummy model and load first checkpoint
        dummy_model = self._create_model()
        state_dict = self._load_checkpoint(self.config.checkpoint_paths[0])
        dummy_model.load_state_dict(state_dict, strict=False)

        # Get encoder based on model type
        if self.config.model_type == "ssl4eo":
            self.shared_encoder = dummy_model.model.vit_encoder.to(self.config.device)
        else:  # terramind
            self.shared_encoder = dummy_model.model.encoder.to(self.config.device)

        self.shared_encoder.eval()

        if self.config.logger:
            self.config.logger.info(f"Shared encoder loaded from {self.config.checkpoint_paths[0]}")

    def _setup_model_encoder(self, model: nn.Module) -> None:
        """Set up model encoder (shared or individual)."""
        if not self.config.freezing_body:
            return

        if self.config.model_type == "ssl4eo":
            del model.model.vit_encoder
            model.model.vit_encoder = self.shared_encoder
        else:  # terramind
            del model.model.encoder
            model.model.encoder = self.shared_encoder

    def _load_model_weights(self, model: nn.Module, state_dict: dict, member_idx: int) -> None:
        """Load weights into model."""
        if self.config.freezing_body:
            # Filter out encoder keys
            prefix = "vit_encoder." if self.config.model_type == "ssl4eo" else "encoder."
            decoder_state_dict = {k: v for k, v in state_dict.items() if not k.startswith(prefix)}
            model.load_state_dict(decoder_state_dict, strict=False)

            if self.config.logger:
                self.config.logger.info(f"Loaded only decoder weights for member {member_idx+1}")
        else:
            model.load_state_dict(state_dict, strict=False)

    def _load_ensemble_members(self) -> List[nn.Module]:
        """
        Load ensemble members from checkpoint paths.

        Returns:
            List of loaded models
        """
        ensemble_members = []

        # Initialize shared encoder if needed
        self._initialize_shared_encoder()

        # Load each ensemble member
        for i, checkpoint_path in enumerate(self.config.checkpoint_paths):
            try:
                # Create and setup model
                model = self._create_model()
                self._setup_model_encoder(model)

                # Load weights
                state_dict = self._load_checkpoint(checkpoint_path)
                self._load_model_weights(model, state_dict, i)

                # Finalize model setup
                model.to(self.config.device)
                model.eval()
                ensemble_members.append(model)

                if self.config.logger:
                    self.config.logger.info(
                        f"Successfully loaded checkpoint {i+1}/{len(self.config.checkpoint_paths)}: {checkpoint_path}"
                    )

            except Exception as e:
                if self.config.logger:
                    self.config.logger.error(
                        f"Failed to load checkpoint {checkpoint_path}: {str(e)}"
                    )
                raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {str(e)}") from e

        if not ensemble_members:
            raise ValueError("No ensemble members were successfully loaded")

        return ensemble_members

    def _get_raw_ensemble_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get raw predictions from all ensemble members.

        We:
        1. Get logits from each model
        2. Apply softmax to get probabilities
        3. Extract the positive class probability (class 1)

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Tensor of shape (batch_size, num_ensemble_members, height, width)
            containing probabilities for the positive class
        """
        with torch.no_grad():
            if self.config.freezing_body and self.shared_encoder:
                # Encode once using the shared encoder
                encoded_input = self.encode(x)
                return self._get_raw_ensemble_predictions_from_embeddings(encoded_input)

            # No shared encoder, process each member independently
            all_member_predictions = []
            for member in self.ensemble_members:
                logits = member(x)
                # Scale to target size
                logits = self.scale_to_target_size(logits)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                all_member_predictions.append(probabilities)

            # Stack predictions to get (num_ensemble_members, batch_size, num_classes, height, width)
            return torch.stack(all_member_predictions, dim=0)

    def _get_raw_ensemble_predictions_from_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get raw predictions from all ensemble members.
        """
        all_member_predictions = []
        with torch.no_grad():
            for member in self.ensemble_members:
                logits = member.decode(x)
                logits = self.scale_to_target_size(logits)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                all_member_predictions.append(probabilities)
        return torch.stack(all_member_predictions, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input tensor.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            The embedding of the input tensor of shape (batch_size, num_patches, embedding_dim)
        """
        return self.ensemble_members[0].encode(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode the input tensor.

        Args:
            x: Input tensor of shape (batch_size, num_patches, embedding_dim)

        Returns:
            The decoded tensor of shape (batch_size, num_classes, height, width)
        """
        ensemble_predictions = self._get_raw_ensemble_predictions_from_embeddings(x)
        return ensemble_predictions.mean(dim=0)

    def variance_from_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the variance of the ensemble predictions from embeddings.

        Args:
            x: Input tensor of shape (batch_size, num_patches, embedding_dim)

        Returns:
            The variance of the ensemble predictions of shape (batch_size, num_classes, height, width)
        """
        ensemble_predictions = self._get_raw_ensemble_predictions_from_embeddings(x)
        return ensemble_predictions.var(dim=0)


if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_checkpoint_paths = [
        "results/frozen_body/vit_small_patch16_224_unet_segmentation/fire/"
        "best_model_25_2025-07-09-11-53/best_model_25_2025-07-09-11-53.pth",
        "results/frozen_body/vit_small_patch16_224_unet_segmentation/fire/"
        "best_model_25_2025-07-09-11-53/last_epoch35.pth",
        "results/frozen_body/vit_small_patch16_224_unet_segmentation/fire/"
        "best_model_46_2025-07-08-21-42/best_model_46_2025-07-08-21-42.pth",
        "results/frozen_body/vit_small_patch16_224_unet_segmentation/fire/"
        "best_model_46_2025-07-08-21-42/last_epoch49.pth",
    ]

    class DummyLogger:
        """Dummy logger for testing."""

        def info(self, msg: str) -> None:
            """Info method.

            Args:
                msg: Message to print
            """
            print(f"[INFO] {msg}")

        def error(self, msg: str) -> None:
            """Error method.

            Args:
                msg: Message to print
            """
            print(f"[ERROR] {msg}")

    dummy_logger = DummyLogger()

    ensemble_model = EnsembleModel(
        checkpoint_paths=model_checkpoint_paths,
        disaster="fire",
        model_name="vit_small_patch16_224_unet_segmentation",
        model_type="ssl4eo",  # or "terramind"
        freezing_body=True,
        logger=dummy_logger,
        device=test_device,
    )
    test_x = torch.randn(1, 13, 224, 224)
    test_x = test_x.to(test_device)
    test_y = ensemble_model.forward(test_x)

    # Mean Probability Map (Segmentation Output)
    mean_prob_map = ensemble_model.predict_proba(test_x)
    print(f"\nMean Probability Map (shape {mean_prob_map.shape}):")
    # Print a small section or summary to avoid overwhelming output
    print(mean_prob_map[0, 0:5, 0:5])  # Print top-left 5x5 pixels of the first image

    # Variance Map
    variance_map = ensemble_model.variance(test_x)
    print(f"\nVariance Map (shape {variance_map.shape}):")
    print(variance_map[0, 0:5, 0:5])

    # Entropy Map
    entropy_map = ensemble_model.entropy(test_x)
    print(f"\nEntropy Map (shape {entropy_map.shape}):")
    print(entropy_map[0, 0:5, 0:5])

    # Mutual Information Map
    mutual_info_map = ensemble_model.mutual_information(test_x)
    print(f"\nMutual Information Map (shape {mutual_info_map.shape}):")
    print(mutual_info_map[0, 0:5, 0:5])

    # Percentiles Map
    custom_percentiles_q = torch.tensor([0.1, 0.5, 0.9]).to(
        test_device
    )  # 10th, 50th, 90th percentiles
    percentile_maps = ensemble_model.percentiles(test_x, custom_percentiles_q)
    print(f"\nPercentiles Map ({custom_percentiles_q.tolist()}) (shape {percentile_maps.shape}):")
    # Print top-left 5x5 pixels for each percentile
    print(percentile_maps[0, 0:5, 0:5, :])

    print("output", test_y.shape)
