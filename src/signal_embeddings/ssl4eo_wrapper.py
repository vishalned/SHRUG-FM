"""
SSL4EO S12 dataset wrapper for loading specific bands.
"""

import os
import re
from typing import Any, Optional
import numpy as np
import rasterio
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import default_collate
from torchgeo.datasets import SSL4EOS12, Sentinel1, Sentinel2
from torchgeo.datasets.utils import disambiguate_timestamp

def collate_skip_none(batch: list[Any]) -> Optional[Any]:
    """
    Collate function that skips None entries in a batch.

    Parameters:
        batch (list): List of items, some of which may be None.

    Returns:
        Collated batch with None entries removed, or None if batch is empty.
    """
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    return default_collate(batch)



class SSL4EOS12Wrapper(SSL4EOS12):
    """
    Wrapper for the SSL4EOS12 dataset to load the bands.
    Args:
        root: str,
        split: str,
        seasons: int,
        transforms: dict,
        dim_reduc_transforms: dict,
        download: bool,
        checksum: bool,
        bands: str,
    """

    def __init__(self, *args, dim_reduc_transforms=None, bands="rgb", **kwargs):
        self.bands_ = bands
        self.dim_reduc_transforms = dim_reduc_transforms
        super().__init__(*args, **kwargs)

    def __getitem_orig__(self, index: int) -> dict[str, Tensor]:  # pylint: disable=too-many-locals
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            image sample
        """
        root = os.path.join(self.root, self.split, f"{index:07}")
        subdirs = os.listdir(root)
        # subdirs = random.sample(subdirs, self.seasons)
        subdirs = subdirs[: self.seasons]
        filename_regex = self.metadata[self.split]["filename_regex"]

        images = []
        xs = []
        ys = []
        wavelengths: list[float] = []
        for subdir in subdirs:
            directory = os.path.join(root, subdir)
            if match := re.match(filename_regex, subdir):
                date_str = match.group("date")
                match self.split:
                    case "s1":
                        date_format = Sentinel1.date_format
                    case "s2c" | "s2a":
                        date_format = Sentinel2.date_format
                _, _ = disambiguate_timestamp(date_str, date_format)
                for band in self.bands:
                    match self.split:
                        case "s1":
                            wavelengths.append(Sentinel1.wavelength)
                        case "s2c" | "s2a":
                            wavelengths.append(Sentinel2.wavelengths[band])

                    filename = os.path.join(directory, f"{band}.tif")
                    with rasterio.open(filename) as f:
                        minx, maxx = f.bounds.left, f.bounds.right
                        miny, maxy = f.bounds.bottom, f.bounds.top
                        image = f.read(out_shape=(1, self.size, self.size))
                        images.append(torch.from_numpy(image.astype(np.float32)))
                xs.append((minx + maxx) / 2)
                ys.append((miny + maxy) / 2)

        sample = {
            "image": torch.cat(images),
            "x": torch.tensor(xs),
            "y": torch.tensor(ys),
            # 't': torch.tensor(ts),
            "wavelength": torch.tensor(wavelengths),
            "res": torch.tensor(10),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __getitem__(self, index):
        try:
            item = self.__getitem_orig__(index)
        except (IndexError, KeyError, OSError) as e:
            print(f"Error getting item {index}: {e}")
            return None
        if self.bands_ == "rgb":
            item["image"] = item["image"][[3, 2, 1]]
        elif self.bands_ == "fire":
            # get similar bands as the hls fire
            item["image"] = item["image"][[1, 2, 3, 8, 11, 12]]
        elif self.bands_ == "all":
            item["image"] = item["image"]
        else:
            raise ValueError(f"Unknown bands: {self.bands_}")
        if self.dim_reduc_transforms is not None:
            item = self.dim_reduc_transforms(item)
        return item


