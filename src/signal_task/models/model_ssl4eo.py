"""This file contains the definition of the SSL4EO model."""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn
from torchgeo.models import ViTSmall16_Weights

from signal_task.config.settings import settings
from signal_task.models.model_components.ssl4eo import SSL4EO
from signal_task.utils.data_processing import process_input_bands


class BaselineNet(nn.Module):
    """Baseline model using SSL4EO."""

    def __init__(
        self,
        disaster: str,
        model_name: str,
        checkpoint_path: str = None,
        freezing_body: bool = True,
        logger: Any | None = None,
        freezing_neck: bool = True,
        ssl_method: str = "MAE",
        device: Optional[torch.device] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__()
        # Define model
        self.num_bands = 13
        self.disaster = disaster
        self.model_type = "ssl4eo"
        self.model_name = model_name
        self.ssl_method = ssl_method
        if ssl_method == "MAE":
            weights = ViTSmall16_Weights.SENTINEL2_ALL_MAE
        elif ssl_method == "DINO":
            weights = ViTSmall16_Weights.SENTINEL2_ALL_DINO
        elif ssl_method == "MOCO":
            weights = ViTSmall16_Weights.SENTINEL2_ALL_MOCO
        elif ssl_method == "FGMAE":
            weights = ViTSmall16_Weights.SENTINEL2_ALL_FGMAE
        else:
            raise ValueError(f"Can't find matched pre-training {ssl_method}.")
        if logger:
            logger.info(f"Using {ssl_method} pre-training")

        if model_name == "vit_small_patch16_224_unet_segmentation":
            # If segmentation task, to use encoder+semantic segmentation head
            model = SSL4EO(
                output_dim=2,
                num_frames=1,
                decoder_norm="batch",
                decoder_padding="same",
                decoder_activation="relu",
                decoder_depths=[2, 2, 8, 2],
                decoder_dims=kwargs.get("decoder_dims", [160, 320, 640, 1280]),
                embed_dim=kwargs.get("embed_dim", 384),
                patch_size=16,
                tubelet_size=1,
                weights=weights,
            )

            # Freeze the encoder and finetune the semantic segmentation head
            if freezing_body:
                if logger:
                    logger.info("Freeze the encoder")
                for _, param in model.vit_encoder.named_parameters():
                    param.requires_grad = False

            if checkpoint_path is not None:
                model_state_dict = torch.load(Path(checkpoint_path), map_location=device)
                model.model.load_state_dict(model_state_dict)
                logger.info(f"Loaded trained model weights from {checkpoint_path}")

        elif model_name == "vit_small_patch16_224_unet_segmentation_density_head":
            from signal_task.models.uq_model_components.dirichlet_ssl4eo import (
                DirichletSSL4EO,
            )

            neck_checkpoint = (
                settings[disaster]["checkpoint_path"]
                + "/frozen_body/vit_small_patch16_224_unet_segmentation/fire"
                + "/best_model_38_2025-07-13-01-13/best_model_38_2025-07-13-01-13.pth"
            )

            model = DirichletSSL4EO(num_frames=1, output_dim=2, encoder_weights=weights)

            if neck_checkpoint is not None:
                model_state_dict = torch.load(neck_checkpoint)
                model.load_state_dict(model_state_dict, strict=False)
                if logger:
                    logger.info(f"Loaded neck weights from {neck_checkpoint}")

            if freezing_body:
                if logger:
                    logger.info("Freeze the encoder")
                for _, param in model.model.vit_encoder.named_parameters():
                    param.requires_grad = False

            if freezing_neck:
                if logger:
                    logger.info("Freeze the neck")
                for _, param in model.model.decoder_head.decoder_bridge.named_parameters():
                    param.requires_grad = False
                for _, param in model.model.decoder_head.decoder_blocks.named_parameters():
                    param.requires_grad = False

            if checkpoint_path is not None:
                new_state_dict = {}
                model_state_dict = torch.load(Path(checkpoint_path), map_location=device)
                for k, v in model_state_dict.items():
                    new_key = k.replace("model.model.", "model.")  # remove the extra 'model.'
                    new_state_dict[new_key] = v

                model.load_state_dict(new_state_dict)
                if logger:
                    logger.info(f"Loaded trained model weights from {checkpoint_path}")
        elif model_name == "vit_small_patch16_224_unet_segmentation_mc_dropout":
            from signal_task.models.uq_model_components.mcdropout_uq_model import (
                MCDropoutUQModel,
            )

            # hard-coded for now, clarify with distribution team how to clean up args
            dropout, num_passes, num_classes, fully_finetune = 0.5, 10, 2, True

            # initialize base model
            model = SSL4EO(
                output_dim=2,
                num_frames=1,
                decoder_norm="batch",
                decoder_padding="same",
                decoder_activation="relu",
                decoder_depths=[2, 2, 8, 2],
                decoder_dims=kwargs.get("decoder_dims", [160, 320, 640, 1280]),
                embed_dim=kwargs.get("embed_dim", 384),
                patch_size=16,
                tubelet_size=1,
                weights=weights,
                dropout=dropout,
            )

            # Freeze the encoder and finetune the semantic segmentation head
            if freezing_body:
                if logger:
                    logger.info("Freeze the encoder")
                for _, param in model.vit_encoder.named_parameters():
                    param.requires_grad = False

            # initialize MC Dropout model wrapper
            model = MCDropoutUQModel(
                model=model,
                num_passes=num_passes,
                num_classes=num_classes,
                dropout=dropout,
                fully_finetune=fully_finetune,
                disaster=disaster,
            )

            if checkpoint_path is not None:
                new_state_dict = {}
                model_state_dict = torch.load(Path(checkpoint_path), map_location=device)
                for k, v in model_state_dict.items():
                    k_temp = k.replace("base_model.", "temp.")
                    k_temp = k_temp.replace("model.", "")  # remove the extra 'model.'
                    new_key = k_temp.replace("temp.", "base_model.")
                    new_state_dict[new_key] = v
                model.load_state_dict(new_state_dict)
                if logger:
                    logger.info(f"Loaded trained model weights from {checkpoint_path}")

        else:
            raise ValueError(f"Can't find matched model {model_name}.")

        model.float()
        self.model = model

    def initialize_weights(self, std: float = 0.02) -> None:
        """
        Initialize the weights of the model.

        Args:
            std: float: The standard deviation of the weights.
        """
        # for m in self.decoder:
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: torch.Tensor: The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        raw_size = x.size() if isinstance(x, torch.Tensor) else None

        if x.shape[1] != 13:
            x = process_input_bands(x, model_type="ssl4eo", disaster=self.disaster)

        x = self.model(x)
        x = getattr(x, "logits", x)
        # Interpolate if the output size doesn't match the input size
        if isinstance(x, torch.Tensor) and raw_size is not None and x.size()[-2:] != raw_size[-2:]:
            x = nn.functional.interpolate(
                x, size=raw_size[-2:], mode="bilinear", align_corners=False
            )
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input tensor.

        Args:
            x: torch.Tensor: The input tensor.

        Returns:
            torch.Tensor: The encoded tensor.
        """
        if x.shape[1] != 13:
            x = process_input_bands(x, model_type="ssl4eo", disaster=self.disaster)

        x = self.model.encode(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode the features that the encoder outputs into the original input shape.

        Args:
            x: torch.Tensor: The input tensor.

        Returns:
            torch.Tensor: The decoded tensor.
        """
        x = self.model.decode(x)
        # Assuming x is already in the shape (batch_size, channels, height, width)
        return x


if __name__ == "__main__":
    MODEL_ID = "baseline"
    MODEL_PATH = Path(settings.density.checkpoint_path) / settings.density[MODEL_ID].model

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_model = BaselineNet(
        disaster="fire",
        model_name="vit_small_patch16_224_unet_segmentation",
        ssl_method="MAE",
        checkpoint_path=MODEL_PATH,
        device=DEVICE,
        logger=None,
    ).to(DEVICE)

    # The model accepts remote sensing data in a video format (B, C, H, W)
    x_sample = torch.randn(1, 13, 224, 224)
    x_sample = x_sample.to(DEVICE)
    y = test_model.forward(x_sample)
    print("output", y.shape)  # (1,2,224,224)
