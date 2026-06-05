# type: ignore
# pylint: skip-file
import os
from typing import Callable, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import rasterio
from ml4floods.data.worldfloods.configs import BANDS_S2
from ml4floods.preprocess.tiling import WindowSize, WindowSlices, get_window_tiles
from torch.utils.data import DataLoader

from signal_task.dataset.floods.dataset import WorldFloodsDatasetTiled
from signal_task.utils.logging_utils import get_logger_initialized

"""
Copy-paste from ml4floods with small adjustments
"""


def get_list_of_window_slices(
    file_names: List[str], window_size: WindowSize, path_to_splits: str = ""
) -> List[WindowSlices]:
    """Function to return the list of window slices for the all the
    input images and the given window size.

    Args:
        file_names (List[str]): List of filenames that are to be sliced.
        window_size (WindowSize): Window size of the tiles.

    Returns:
        List[WindowSlices]: List of window slices for the each tile
        corresponding to each input image.
    """

    accumulated_list_of_windows = []
    for ifilename in file_names:
        with rasterio.open(os.path.join(path_to_splits, ifilename)) as dataset:
            # get list of windows
            list_of_windows = get_window_tiles(
                dataset, height=window_size.height, width=window_size.width
            )
            # create a list of filenames
            list_of_windows = [
                WindowSlices(file_name=ifilename, window=iwindow) for iwindow in list_of_windows
            ]

        accumulated_list_of_windows += list_of_windows

    return accumulated_list_of_windows


class WorldFloodsDataModule(pl.LightningDataModule):
    """A prepackaged WorldFloods Pytorch-Lightning data module
    This initializes a module given a set a directory with a subdirectory
    for the training and testing data ("image_folder" and "target_folder").
    Then we can search through the directory and load the images found. It
    creates the train, val and test datasets which then can be used to initialize
    the dataloaders. This is pytorch lightning compatible which can be used with
    the training fit framework.

    Args:
        filenames_train_test: path to json file with splits
        path_to_splits: path to worldfloods_v2/ folder in mounted data bucket,
            e.g. /gcs/2025-esl-extreme-environments-raw-data/raw_downstream/floods/worldfloods_v2
        input_folder (str): the input folder sub_directory
        target_folder (str): the target folder sub directory
        train_transformations (Callable): the transformations used within the
            training data module
        test_transformations (Callable): the transformations used within the
            testing data module
        window_size (Tuple[int,int]): the window size used to tile the images
            for training
        batch_size (int): the batchsize used for the dataloader
        bands (List(int)): the bands to be selected from the images

    Attributes:
        train_transform (Callable): the transformations used within the
            training data module
        test_transform (Callable): the transformations used within the
            testing data module
        bands (List(int)): the bands to be selected from the images
        image_prefix (str): the input folder sub_directory
        gt_prefix (str): the target folder sub directory
        window_size (Tuple[int,int]): the window size used to tile the images
            for training
        filter_windows (Callable): function to filter the training tiles by
            number of invalid and cloud pixels
        filenames_train_test (Dict): path to images and ground truth for
            the training, validation and test splits

    Example:
        >>> from ml4floods.data.worldfloods.lightning import WorldFloodsDataModule
        >>> wf_dm = WorldFloodsDataModule()
        >>> wf_dm.prepare_data()
        >>> wf_dm.setup()
        >>> train_dl = wf_dm.train_dataloader()
    """

    def __init__(
        self,
        filenames_train_test: Dict,
        path_to_splits: str,
        input_folder: str = "S2",
        target_folder: str = "gt",
        train_transformations: Optional[Callable] = None,
        test_transformations: Optional[Callable] = None,
        add_mndwi_input: bool = False,
        window_size: Tuple[int, int] = (64, 64),
        model_input_size: int = 224,
        batch_size: int = 32,
        bands: List[int] = [1, 2, 3],
        num_workers: int = 4,
        num_workers_val: int = 0,
        num_workers_test: int = 0,
        filter_windows: Callable = None,
        lock_read: bool = False,
        bootstrap_seed: int | None = None,
    ):
        super().__init__()
        self.train_transform = train_transformations
        self.test_transform = test_transformations
        self.num_workers = num_workers
        self.num_workers_test = num_workers_test
        self.num_workers_val = num_workers_val
        self.lock_read = lock_read
        self.path_to_splits = path_to_splits
        self.model_input_size = model_input_size

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.bands = bands
        self.add_mndwi_input = add_mndwi_input
        self.batch_size = batch_size
        # Prefixes
        self.image_prefix = input_folder
        self.gt_prefix = target_folder
        self.filter_windows = filter_windows
        self.window_size = WindowSize(height=window_size[0], width=window_size[1])
        self.filenames_train_test = filenames_train_test
        self.bootstrap_seed = bootstrap_seed
        self.logger = get_logger_initialized()

        files = {}
        splits = ["train", "test", "val"]

        # loop through the naming splits
        for isplit in splits:
            # TODO we might could use the train_test_split dict directly to avoid using image_prefix and gt_prefix
            files[isplit] = self.filenames_train_test[isplit][self.image_prefix]

        # save filenames
        self.train_files = files["train"]
        self.val_files = files["val"]
        self.test_files = files["test"]

    def prepare_data(self):
        """Does Nothing for now. Here for compatibility."""
        # TODO: here we can check for correspondence between the files

    def get_mndwi_indices(self, bands):
        band_names_current_image = [BANDS_S2[iband] for iband in bands]
        mndwi_indexes_current_image = [band_names_current_image.index(b) for b in ["B3", "B11"]]
        return mndwi_indexes_current_image

    def _filter_windows(self, dataset: WorldFloodsDatasetTiled, split: str) -> List[WindowSlices]:
        """Filter the windows in the dataset based on the provided filter function."""
        if self.filter_windows is not None:
            nb_windows = len(dataset.list_of_windows)
            filtered = self.filter_windows(dataset, split)
            self.logger.info(
                f"Filtered {nb_windows - len(filtered)}/{nb_windows} windows from {split} dataset."
            )
            return filtered
        return dataset.list_of_windows

    def setup(self, stage=None):
        """This creates the PyTorch dataset given the preconfigured
        file paths.
        """

        self.train_dataset = WorldFloodsDatasetTiled(
            list_of_windows=get_list_of_window_slices(
                self.train_files, self.window_size, self.path_to_splits
            ),
            image_prefix=self.image_prefix,
            gt_prefix=self.gt_prefix,
            bands=self.bands,
            mndwi_indices=self.get_mndwi_indices(self.bands) if self.add_mndwi_input else None,
            transforms=self.train_transform,
            lock_read=self.lock_read,
            path_to_splits=self.path_to_splits,
            model_input_size=self.model_input_size,
            bootstrap_seed=self.bootstrap_seed,
        )
        self.train_dataset.set_filtered_windows(self._filter_windows(self.train_dataset, "train"))

        self.val_dataset = WorldFloodsDatasetTiled(
            list_of_windows=get_list_of_window_slices(
                self.val_files, self.window_size, self.path_to_splits
            ),
            image_prefix=self.image_prefix,
            gt_prefix=self.gt_prefix,
            bands=self.bands,
            mndwi_indices=self.get_mndwi_indices(self.bands) if self.add_mndwi_input else None,
            transforms=self.test_transform,
            lock_read=self.lock_read,
            path_to_splits=self.path_to_splits,
            model_input_size=self.model_input_size,
        )
        self.val_dataset.set_filtered_windows(self._filter_windows(self.val_dataset, "val"))

        # this used to be WorldFloodsDataset but I changed it to Tiled
        self.test_dataset = WorldFloodsDatasetTiled(
            list_of_windows=get_list_of_window_slices(
                self.test_files, self.window_size, self.path_to_splits
            ),
            image_prefix=self.image_prefix,
            gt_prefix=self.gt_prefix,
            bands=self.bands,
            mndwi_indices=self.get_mndwi_indices(self.bands) if self.add_mndwi_input else None,
            transforms=self.test_transform,
            lock_read=self.lock_read,
            path_to_splits=self.path_to_splits,
            model_input_size=self.model_input_size,
        )
        self.test_dataset.set_filtered_windows(self._filter_windows(self.test_dataset, "test"))

    def train_dataloader(self):
        """Initializes and returns the training dataloader"""
        return (
            DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
            ),
            self.train_dataset.meta,
        )

    def val_dataloader(self, num_workers=None):
        """Initializes and returns the validation dataloader"""
        num_workers = num_workers or self.num_workers_val
        return (
            DataLoader(
                self.val_dataset, batch_size=self.batch_size, num_workers=num_workers, shuffle=False
            ),
            self.val_dataset.meta,
        )

    def test_dataloader(self, num_workers=None):
        """Initializes and returns the test dataloader"""
        num_workers = num_workers or self.num_workers_test
        return (
            DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=num_workers,
                shuffle=False,
            ),
            self.test_dataset.meta,
        )
