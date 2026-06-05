"""
Create embeddings for the data.
"""

from pathlib import Path

import numpy as np
import torch
import torchgeo
from torch.utils.data import DataLoader
from torchgeo.models import ViTSmall16_Weights
from tqdm import tqdm


class CreateEmbeddings:
    """
    Class for creating embeddings using pre-trained vision transformers.
    """

    def __init__(
        self,
        weights="pretrained",
        save_path=None,
        num_channels=3,
        dataset_name="ptdata",
        split="",
        prefix="",
    ) -> None:
        """
        Initialize the CreateEmbeddings class with a pre-trained model.
        Args:
            weights: Pre-trained weights for the model.
            save_path: Path to save the embeddings.
            num_channels: Number of channels in the input images.
            data: Type of data being processed
                - ptdata (pretraining)
                - dtdata (downstream - hls burnscars data)
                (default is 'ptdata').
            split: Split of the data.
            prefix: Prefix of the data.
        """
        if weights == "pretrained":
            weights = ViTSmall16_Weights.SENTINEL2_ALL_MAE
        # TODO add support for random weights
        self.vit_encoder = torchgeo.models.vit_small_patch16_224(weights=weights)
        Path(save_path).mkdir(parents=True, exist_ok=True)
        self.root_path = Path(save_path)
        self.n_channels = num_channels
        self.dataset_name = dataset_name
        self.embeddings = []
        if prefix == "":
            self.results_file_name = f"embeddings_{self.dataset_name}_{self.n_channels}_{split}.npz"
        else:
            self.results_file_name = f"{prefix}.npz"

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings from the input tensor using the pre-trained model.

        Parameters:
            x (torch.Tensor): Input tensor to get embeddings for.

        Returns:
            torch.Tensor: Embeddings tensor.
        """
        B, _, H, W = x.shape
        x_final = torch.zeros((B, 13, H, W), dtype=x.dtype)
        if self.n_channels == 6:
            if self.dataset_name == "burnscar":
                x_final[:, 1] = x[:, 0]  # B02 (Blue)
                x_final[:, 2] = x[:, 1]  # B03 (Green)
                x_final[:, 3] = x[:, 2]  # B04 (Red)
                x_final[:, 8] = x[:, 3]  # B08A (NIR narrow)
                x_final[:, 11] = x[:, 4]  # B11 (SWIR 1)
                x_final[:, 12] = x[:, 5]  # B12 (SWIR 2)
            elif self.dataset_name == "ssl4eo":
                x_final[:, 1] = x[:, 0]  # B02 (Blue)
                x_final[:, 2] = x[:, 1]  # B03 (Green)
                x_final[:, 3] = x[:, 2]  # B04 (Red)
                x_final[:, 8] = x[:, 3]  # B08A (NIR narrow)
                x_final[:, 11] = x[:, 4]  # B11 (SWIR 1)
                x_final[:, 12] = x[:, 5]  # B12 (SWIR 2)
            elif self.dataset_name == "flood":
                x_final = x
            elif self.dataset_name == "landslide":        
                x_final[:, 1] = x[:, 0]  # B02 (Blue)
                x_final[:, 2] = x[:, 1]  # B03 (Green)
                x_final[:, 3] = x[:, 2]  # B04 (Red)
                x_final[:, 8] = x[:, 9]  # B08A (NIR narrow)
                x_final[:, 11] = x[:, 7]  # B11 (SWIR 1)
                x_final[:, 12] = x[:, 8]  # B12 (SWIR 2)
        elif self.n_channels == 3:
            # our data loader loads RGB, vit expect bgr
            x_final[:, 1] = x[:, 2]  # Blue
            x_final[:, 2] = x[:, 1]  # Green
            x_final[:, 3] = x[:, 0]  # Red
        with torch.no_grad():
            x_emb = self.vit_encoder.forward_features(
                x_final
            )  # b, 197, 384 (b, sequence length (flattened_patches)+cls, embed_dim )
            # remove cls token
        x_emb = x_emb[:, 0, :]  # b, 196, 384
        return x_emb

    def get_embeddings_dl(self, dataloader: DataLoader, save: bool = True) -> None:
        """
        Get embeddings from a DataLoader.

        Parameters:
            dataloader: DataLoader containing the input data.
            save (bool): Whether to save the embeddings.

        Raises:
            ValueError: If the data type is invalid.
        """
        for batch in tqdm(dataloader):
            if self.dataset_name == "ssl4eo":
                x = batch["image"]
            elif self.dataset_name == "burnscar":
                x = batch["x"]
            elif self.dataset_name == "floods":
                x = batch["x"]
            elif self.dataset_name == "landslide":
                x = batch["x"]
            else:
                raise ValueError(f"Invalid dataset name: {self.dataset_name}")
            emb_i = self.get_embeddings(x)
            self.embeddings.append(emb_i)
            
        self.embeddings = torch.cat(self.embeddings, dim=0)
        print(f"Embeddings shape: {self.embeddings.shape}")
        if save:
            self.save()

    def save(self) -> None:
        """
        Save the embeddings to a file.
        """
        np.savez_compressed(
            self.root_path / self.results_file_name, embeddings=self.embeddings.detach().numpy()
        )
        print(f"Embeddings saved to {self.root_path / self.results_file_name}")