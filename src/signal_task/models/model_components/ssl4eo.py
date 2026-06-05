"""Module containing SSL4EO model implementation for extreme environment monitoring."""

from typing import Any, List, Optional

import torch
import torchgeo
from torch import nn
from torchgeo.models import ViTSmall16_Weights

from .decoder import CoreDecoder


class SSL4EO(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        *,
        patch_size: int = 16,
        num_frames: int = 1,
        tubelet_size: int = 1,
        embed_dim: int = 384,
        output_dim: int = 2,
        decoder_norm: str = "batch",
        decoder_padding: str = "same",
        decoder_activation: str = "relu",
        decoder_depths: Optional[List[int]] = None,
        decoder_dims: Optional[List[int]] = None,
        weights: Optional[Any] = None,
        dropout: Optional[float] = 0.0,
    ) -> None:
        """Initialize the SSL4EO model.

        Args:
            patch_size: Size of patches for ViT
            num_frames: Number of frames to process
            tubelet_size: Size of tubelets
            embed_dim: Embedding dimension
            output_dim: Output dimension
            decoder_norm: Type of normalization for decoder
            decoder_padding: Type of padding for decoder
            decoder_activation: Activation function for decoder
            decoder_depths: List of depths for decoder layers
            decoder_dims: List of dimensions for decoder layers
            weights: Pre-trained weights for the model
            dropout: Probability of dropout. Defaults to 0.0, no dropout
        """
        super().__init__()

        # Define default values here to be safe
        if decoder_depths is None:
            decoder_depths = [2, 2, 8, 2]
        if decoder_dims is None:
            decoder_dims = [160, 320, 640, 1280]

        # --------------------------------------------------------------------------
        # encoder specifics
        if weights is None:
            self.vit_encoder = torchgeo.models.vit_small_patch16_224()
        else:
            self.vit_encoder = torchgeo.models.vit_small_patch16_224(weights=weights)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # --------------------------------------------------------------------------
        # CNN Decoder Blocks:
        self.depths = decoder_depths
        self.dims = decoder_dims
        self.tubelet_size = tubelet_size
        self.output_dim = output_dim
        self.num_frames = num_frames
        self.dropout = dropout

        self.decoder_head = CoreDecoder(
            embedding_dim=embed_dim * num_frames,
            output_dim=output_dim,
            depths=decoder_depths,
            dims=decoder_dims,
            activation=decoder_activation,
            padding=decoder_padding,
            norm=decoder_norm,
            dropout=dropout,
        )

        self.decoder_downsample_block = nn.Identity()

    def reshape(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape the input tensor for processing.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim)

        Returns:
            torch.Tensor: Reshaped tensor
        """
        # Separate channel axis
        batch_size, seq_len, embed_dim = x.shape
        x = x.permute(0, 2, 1)
        # Non-temporal version commented out for reference
        # x = x.view(batch_size, embed_dim, int(seq_len**0.5), int(seq_len**0.5))
        x = x.reshape(
            batch_size,
            embed_dim * self.num_frames,
            int((seq_len / self.num_frames) ** 0.5),
            int((seq_len / self.num_frames) ** 0.5),
        )

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        x = self.vit_encoder.forward_features(x)
        # remove cls token
        x = x[:, 1:, :]  # 1, 196, 384
        # reshape into 2d features
        x = self.reshape(x)  # 1, 384, 14, 14
        x = self.decoder_downsample_block(x)  # [1, 384, 14, 14]
        x = self.decoder_head(x)  # [1, 2, 512, 512]
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input tensor using the VisionTransformer encoder.

        Args:
            x: Input tensor to encode

        Returns:
            torch.Tensor: Encoded features
        """
        # 1, 197, 384 (b, sequence length (flattened_patches)+cls, embed_dim)
        x = self.vit_encoder.forward_features(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode the features into the original input shape.

        Args:
            x: Input tensor of encoded features

        Returns:
            torch.Tensor: Decoded output
        """
        # remove cls token
        x = x[:, 1:, :]  # 1, 196, 384
        # reshape into 2d features
        x = self.reshape(x)  # 1, 384, 14, 14
        x = self.decoder_downsample_block(x)  # [1, 384, 14, 14]
        x = self.decoder_head(x)  # [1, 2, 512, 512]
        # Assuming x is already in the shape (batch_size, channels, height, width)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_weights = ViTSmall16_Weights.SENTINEL2_ALL_MAE

    model = SSL4EO(
        output_dim=2,
        num_frames=1,
        decoder_norm="batch",
        decoder_padding="same",
        decoder_activation="relu",
        decoder_depths=[2, 2, 8, 2],
        decoder_dims=[160, 320, 640, 1280],
        embed_dim=384,
        patch_size=16,
        tubelet_size=1,
        weights=model_weights,
        dropout=0.0,
    )

    model = model.to(device)
    sample_input = torch.randn((1, 13, 224, 224)).to(device)
    model.forward(sample_input)
