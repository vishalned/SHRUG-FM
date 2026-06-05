"""
Multispectral dataset class from ExE-Bench.
"""

import os
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import pandas as pd
import rasterio
import torch
from PIL import Image

from ..config.settings import settings


class BaseMultispectralDataset(torch.utils.data.Dataset):
    """
    1, Blue, B02
    2, Green, B03
    3, Red, B04
    4, NIR, B8A
    5, SW 1, B11
    6, SW 2, B12
    Further details in https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars
    """

    def __init__(
        self,
        *,
        split: str = "train",
        bands: str | None = None,
        disaster: str = "fire",
        data_path: str | None = None,
        bootstrapping_seed: Union[str, None] = None,
    ):
        """
        Initialize the dataset.

        Args:
            split: str: The split of the dataset.
            bands: str | None: The bands to use.
            disaster: str: The disaster.
            data_path: str | None: The path to the data.
            bootstrapping_seed: Union[str, None]: The seed for bootstrapping.
        """
        assert split in {"train", "val", "test"}
        self.split = split
        self.folder = "training" if split in {"train", "val"} else "validation"
        self.settings = settings[disaster]
        self.chip_size = self.settings["dataloader"]["img_size"]
        if data_path is None:
            data_path = self.settings["data_path"]
        self.data_path = Path(data_path) / disaster
        self.disaster = disaster
        self.bands = bands
        self.val_ratio = self.settings["dataloader"]["val_ratio"]
        self.bootstrapping_seed = bootstrapping_seed

        self.filenames = self._read_split()  # read train/valid/test splits

    def _normalization(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize the image by subtracting the mean and dividing by the standard deviation.

        Args:
            image: np.ndarray: The image to normalize.

        Returns:
            np.ndarray: The normalized image.
        """
        means = np.array(self.settings["normalization"]["means"])
        stds = np.array(self.settings["normalization"]["stds"])
        image = (image - means[:, None, None]) / stds[:, None, None]
        return image

    def _transform(self, x: dict) -> dict:
        """
        Transform the image to the desired size.

        Args:
            x: dict: The sample.

        Returns:
            dict: The transformed sample.
        """
        image = x["image"]  # CHW
        new_chips = np.zeros((image.shape[0], self.chip_size, self.chip_size), dtype=np.float32)

        for i in range(image.shape[0]):
            # cv2.resize dsize receive the parameter(width, height), different from the img size of (H, W)
            new_slice = cv2.resize(  # pylint: disable=no-member
                image[i],
                (self.chip_size, self.chip_size),
                interpolation=cv2.INTER_NEAREST,  # pylint: disable=no-member
            )
            new_chips[i, :, :] = new_slice

        x["image"] = new_chips
        return x

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample from the dataset.

        Args:
            idx: int: The index of the sample.

        Returns:
            dict: The sample.

        Raises:
            BaseException: If coords cannot be read.
            ValueError: If the disaster is not a valid disaster.
        """
        if self.disaster not in settings:
            raise ValueError(f"{self.disaster} is not a valid disaster")
        filename = self.filenames[idx]
        base_path = Path(self.data_path) / self.folder
        image_path = base_path / (filename + settings[self.disaster]["image_file_suffix"])
        label_path = base_path / (filename + settings[self.disaster]["label_file_suffix"])
        with rasterio.open(image_path) as dataset:
            if self.bands is None:
                image = dataset.read()  # CHW
                image = self._normalization(image)

            elif self.bands == "rgb":
                red = dataset.read(3)
                green = dataset.read(2)
                blue = dataset.read(1)
                # Stack the bands into a single array with shape (height, width, 3)
                image = np.stack((red, green, blue), axis=-1)
            try:
                spatial_coords = dataset.lnglat()
            # pylint: disable=broad-exception-caught
            except BaseException as e:
                print(f"Error reading coords: {e}")
                spatial_coords = None

        label = np.array(Image.open(label_path))  # (512,512)

        sample = {
            "image": image,
            "label": label,
            "spatial_coords": spatial_coords,
            "meta_info": filename,
        }

        if self.chip_size != 512:
            sample = self._transform(sample)

        return sample

    def _read_split(self):
        split_filename = (
            "validation_index.csv" if self.split == "test" else "training_index.csv"
        )
        print(f"Using {split_filename} for training or testing")
        split_folder = "validation" if self.split == "test" else "training"
        split_filepath = os.path.join(self.data_path, split_folder, split_filename)
        split_data = pd.read_csv(split_filepath)

        if self.bootstrapping_seed is not None and self.split in {"train", "val"}:
            print("Bootstrapping detected, generating samples ...")
            bootstrapping_split_filename = f"training_index_seed-{self.bootstrapping_seed}.csv"
            bootstrapping_split_filepath = os.path.join(
                self.data_path, split_folder, bootstrapping_split_filename
            )
            if not Path(bootstrapping_split_filepath).exists():
                split_data = pd.read_csv(split_filepath)
                split_data = split_data.sample(
                    n=len(split_data), replace=True, random_state=int(self.bootstrapping_seed)
                )
                split_data.to_csv(
                    Path(self.data_path)
                    / split_folder
                    / f"training_index_seed-{self.bootstrapping_seed}.csv",
                    index=False,
                )
                print("Bootstrapping file does not exist, generated")
            else:
                split_data = pd.read_csv(bootstrapping_split_filepath)
                print("Bootstrapping file does exist, loaded")

        filenames = split_data.iloc[:, 0]
        split = round(1 / self.val_ratio)

        prefix = settings[self.disaster]["prefix"]  # prefix of filename
        if self.split == "train":  # 90% for train
            filenames = [x[:prefix] for i, x in enumerate(filenames) if i % split != 0]
        elif self.split == "val":  # 10% for validation
            filenames = [x[:prefix] for i, x in enumerate(filenames) if i % split == 0]
        elif self.split == "test":
            filenames = [x[:prefix] for i, x in enumerate(filenames)]

        return filenames


class MultispectralDataset(BaseMultispectralDataset):
    """Multispectral dataset class from ExE-Bench."""

    def __getitem__(self, *args, **kwargs):
        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = sample["image"]
        spatial_coords = sample["spatial_coords"]

        label = np.array(Image.fromarray(sample["label"]))
        mask = label == -1
        # convert missing data to nonfire
        label[label == -1] = 0
        # CHW
        tensor_x = torch.tensor(image, dtype=torch.float32)
        sample["x"] = torch.nan_to_num(tensor_x, nan=0.0)
        # raise ValueError(f"Input tensor contains NaN values")
        # y shape 1HW
        sample["y"] = torch.tensor(np.expand_dims(label, 0), dtype=torch.float32).long()
        # To do: if the noise mask not impact the final performance, then remove it.
        sample["spatial_coords"] = torch.tensor(spatial_coords, dtype=torch.float32)
        sample["noise_mask"] = torch.tensor(np.expand_dims(~mask, 0))
        return sample


class Sentinel1Flood(MultispectralDataset):
    """Sentinel-1 flood dataset class from ExE-Bench."""

    def __init__(
        self,
        split: str = "train",
        data_path: str | None = None,
        bootstrapping_seed: Union[str, None] = None,
    ):
        super().__init__(
            disaster="flood",
            split=split,
            data_path=data_path,
        )
        self.meta_info = {"disaster": "flood"}


class HlsFire(MultispectralDataset):
    """HLS fire dataset class from ExE-Bench."""

    def __init__(
        self,
        split: str = "train",
        data_path: str | None = None,
        bootstrapping_seed: Union[str, None] = None,
    ):
        super().__init__(
            disaster="fire", split=split, data_path=data_path, bootstrapping_seed=bootstrapping_seed
        )
        self.meta_info = {"disaster": "fire"}


if __name__ == "__main__":
    test_dataset = MultispectralDataset(
        split="train",
        bands=None,
        disaster="flood",
    )
    x_sample = test_dataset[1]
    img_sample = x_sample["x"]
    mask_sample = x_sample["y"]
