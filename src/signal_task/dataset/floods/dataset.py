# type: ignore
# pylint: skip-file

import contextlib
import numbers
import os
import random
import threading
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pyproj
import rasterio
import rasterio as rio
import rasterio.windows
import torch
from ml4floods.data import utils
from ml4floods.data.worldfloods.configs import BANDS_S2
from ml4floods.preprocess.tiling import WindowSlices
from shapely import Polygon
from shapely.geometry import box
from shapely.ops import transform
from torch.utils.data import Dataset

from fm_extremes_uncertainty.utils.logging_utils import get_logger_initialized


def resize_img(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Transform the image to the desired size.

    Args:
        x: Tensor: The image, shape C,H,W.

    Returns:
        Tensor: The transformed sample.
    """
    out: torch.Tensor = torch.nn.functional.interpolate(
        x.unsqueeze(0),  # Add batch dimension
        size=(window_size, window_size),
        mode="nearest-exact",
    )
    return out.squeeze(0)  # Remove batch dimension


def reproject_bbox(bbox: Polygon, crs_source: str, crs_target: str = "4326") -> Polygon:
    """
    Credits to Patrick
    https://github.com/FrontierDevelopmentLab/2025-ESL-Extreme-Environments/blob/bca63e14b52bf7861fe96634feb2195e615e4132/src/fm_extremes_distribution/data_processing/fetch_hydroatlas.py#L123

    Function to map from arbitrary CRS (e.g. UTM) to WGS84,
    needed as a helper function to parse WorldFloods metadata

    Args:
        bounds_str (str): String of bounding box
        crs_source (str): String of source CRS
        crs_target (str): String of target CRS

    Returns:
        Any: depends on val's proper type
    """
    try:
        # define source and target CRS
        src_crs = pyproj.CRS(crs_source)
        tgt_crs = pyproj.CRS(crs_target)  # ("EPSG:4326")

        # define transformer and do the actual mapping
        project = pyproj.Transformer.from_crs(src_crs, tgt_crs, always_xy=True).transform
        return transform(project, bbox)

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error reprojecting row from CRS {crs_source} to {crs_target}: {e}")
        return None


def window_to_coordinates(window: WindowSlices, window_filename: str) -> Polygon:
    with rio.open(window_filename) as ds:
        # compute window bounds in dataset CRS
        xs, ys = rio.transform.xy(
            ds.transform,
            [window[1].row_off, window[1].height + window[1].row_off],
            [window[1].col_off, window[1].col_off + window[1].width],
        )
        window_box = box(xs[0], ys[0], xs[1], ys[1])
        return reproject_bbox(window_box, ds.crs.to_string(), "EPSG:4326")


"""
The remainder was copy-pasted from ml4floods, then adjusted for our needs.
"""


class WorldFloodsDatasetTiled(Dataset):
    """A prepackaged WorldFloods PyTorch Dataset
    This initializes the dataset given a set a set of image files with a
    subdirectory for the training and testing data ("image_prefix" and "gt_prefix").
    This also does the tiling under the hood given the windowsize.

    Args:
        list_of_windows (List[WindowSlices]):  a list of
            namedtuples each consisting of a filename and a rasterio.window
        path_to_splits: path to worldfloods_v2/ folder in mounted data bucket,
            e.g. /gcs/2025-esl-extreme-environments-raw-data/raw_downstream/floods/worldfloods_v2
        image_prefix (str): the input folder sub_directory
        gt_prefix (str): the target folder sub directory
        transforms (Callable): the transformations used within the
            training data module

    Attributes:
        list_of_windows (List[WindowSlices]):  a list of
            namedtuples each consisting of a filename and a rasterio.window
        image_prefix (str): the input folder sub_directory
        gt_prefix (str): the target folder sub directory
        transforms (Callable): the transformations used within the
            training data module
        bands: List[int]
            0-based list of bands to read from BANDS_S2
    """

    def __init__(
        self,
        list_of_windows: List[WindowSlices],
        path_to_splits: str,
        image_prefix: str = "/image_files/",
        gt_prefix: str = "/gt_files/",
        transforms: Optional[Callable] = None,
        bands: List[int] = list(range(len(BANDS_S2))),
        mndwi_indices: List[int] = None,
        lock_read: bool = False,
        model_input_size: int = 224,
        bootstrap_seed: Optional[int] = None,
    ) -> None:
        self.image_prefix = image_prefix
        self.gt_prefix = gt_prefix
        self.transforms = transforms
        self.channels_read = bands
        self.mndwi_indices = mndwi_indices
        self.path_to_splits = path_to_splits
        self.model_input_size = model_input_size
        self.bootstrap_seed = bootstrap_seed
        self.logger = get_logger_initialized()

        if lock_read:
            # Useful when reading from bucket
            self._lock = threading.Lock()
        else:
            self._lock = contextlib.nullcontext()

        self.list_of_windows = list_of_windows
        self.meta = {"disaster": "flood"}

        # use bootstrap_seed to use a random sample of the training data
        if bootstrap_seed is not None:
            self.list_of_windows = self.bootstrap(self.list_of_windows, bootstrap_seed)

    def bootstrap(self, list_of_windows: List[WindowSlices], seed: int) -> List[WindowSlices]:
        self.logger.info(f"Using bootstrap seed {seed} to sample training data.")
        bootstrap_size = round(len(list_of_windows))
        random.seed(seed)
        return random.choices(list_of_windows, k=bootstrap_size)

    def set_filtered_windows(self, filtered_windows: List[WindowSlices]) -> None:
        """
        Set the filtered windows for the dataset.

        Args:
            filtered_windows (List[WindowSlices]): List of filtered windows.
        """
        self.list_of_windows = filtered_windows
        if self.bootstrap_seed is not None:
            self.list_of_windows = self.bootstrap(self.list_of_windows, self.bootstrap_seed)

    def get_label(self, idx: int) -> np.ndarray:
        """
        Method to read only the label. This function is useful for filtering the patches of the Dataset
        Args:
            idx:

        Returns:
        """
        sub_window = self.list_of_windows[idx]
        y_name = sub_window.file_name.replace(self.image_prefix, self.gt_prefix, 1)
        y_name = os.path.join(self.path_to_splits, y_name)
        return rasterio_read(
            y_name,
            self._lock,
            channels=None,
            kwargs_rasterio={
                "window": sub_window.window,
                "boundless": True,
                "fill_value": 0,
            },
        )

    def __getitem__(self, idx: int) -> Dict:
        """Index to select an image tile

        Args:
            idx (int): index

        Returns:
            a dictionary with the keys to the image tiles and mask tiles
            {"image", "mask"}
        """
        # get filenames from named tuple
        sub_window = self.list_of_windows[idx]

        # get filename
        image_name = sub_window.file_name

        # replace string for image_prefix
        image_name = image_name.replace(self.gt_prefix, self.image_prefix, 1)

        # replace string for gt_prefix
        y_name = image_name.replace(self.image_prefix, self.gt_prefix, 1)

        image_name = os.path.join(self.path_to_splits, image_name)

        y_name = os.path.join(self.path_to_splits, y_name)

        # Open Image File
        image_tif = rasterio_read(
            image_name,
            self._lock,
            channels=[c + 1 for c in self.channels_read],
            kwargs_rasterio={
                "window": sub_window.window,
                "boundless": True,
                "fill_value": 0,
            },
        )
        mask_tif = rasterio_read(
            y_name,
            self._lock,
            channels=None,
            kwargs_rasterio={
                "window": sub_window.window,
                "boundless": True,
                "fill_value": 0,
            },
        )

        # get rid of nan, convert to float
        image = np.nan_to_num(image_tif).astype(np.float32)

        if self.mndwi_indices is not None:
            mndwi = (image[self.mndwi_indices][0] - image[self.mndwi_indices][1]) / (
                image[self.mndwi_indices][0] + image[self.mndwi_indices][1] + 1e-6
            )
            image = np.concatenate([image, mndwi[np.newaxis]], axis=0)
        mask = np.nan_to_num(mask_tif).astype(int)

        # Apply transformation
        if self.transforms is not None:
            data = self.transforms(image=image, mask=mask)
        else:
            data = {"image": image, "mask": mask}

        if (
            sub_window.window.width != self.model_input_size
            or sub_window.window.height != self.model_input_size
        ):
            data["image"] = resize_img(data["image"], self.model_input_size)

        window_filename = os.path.join(self.path_to_splits, sub_window.file_name)
        proj_window_box = window_to_coordinates(sub_window, window_filename)

        # Dropping the cloud mask (dim 0) and return {0: 'land', 1: 'water', 2: 'invalid'} mask
        # labels: {'gtversion': 'v2',
        # 'encoding_values': [{0: 'invalid', 1: 'clear', 2: 'cloud'},
        # {0: 'invalid', 1: 'land', 2: 'water'}],
        invalid_mask = data["mask"][1] == 0
        water_mask = data["mask"][1] == 2
        data["mask"][1][invalid_mask] = 2
        data["mask"][1][water_mask] = 1
        data["mask"][1][~(water_mask | invalid_mask)] = 0
        return {
            "x": data["image"],
            "y": data["mask"][1],
            # lon, lat, lon, lat
            "spatial_coords": torch.tensor(proj_window_box.bounds),
            "filename": image_name,
        }
    

    def __len__(self) -> int:
        return len(self.list_of_windows)


def rasterio_read(
    image_name: str, lock, channels: List[int] = None, kwargs_rasterio: Dict = {}
) -> np.ndarray:
    with lock:
        with utils.rasterio_open_read(image_name) as f:
            im_tif = f.read(channels, **kwargs_rasterio)

    return im_tif


def load_input(
    tiff_input: str,
    channels: Union[List[int], List[str]],
    window: Optional[rasterio.windows.Window] = None,
) -> Tuple[torch.Tensor, rasterio.transform.Affine]:
    """
    Reads from a tiff the specified channel and window.

    Args:
        tiff_input: path to geotiff file
        window: rasterio.Window object to read (None to read all)
        channels: 0-based channels to read or names of the bands (we will use `.descriptions` to find the band
            names of the `tiff_input`).

    Returns:
        3-D tensor (len(channels), H, W), Affine transform to geo-reference the array read.

    """
    with utils.rasterio_open_read(tiff_input) as rst:
        if isinstance(channels[0], numbers.Number):
            indexes = (np.array(channels) + 1).tolist()
        else:
            channels_tiff = list(rst.descriptions)
            indexes = [channels_tiff.index(c) + 1 for c in channels]

        inputs = rst.read(indexes, window=window)

        # Shifted transform based on the given window (used for plotting)
        transform = (
            rst.transform if window is None else rasterio.windows.transform(window, rst.transform)
        )
        torch_inputs = torch.tensor(np.nan_to_num(inputs).astype(np.float32))
    return torch_inputs, transform
