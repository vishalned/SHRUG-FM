import torch
import numpy as np
import pandas as pd
from pathlib import Path
import rasterio
from rasterio.windows import Window
from rasterio.windows import from_bounds
import cv2
from typing import Union

from fm_extremes_uncertainty.config.settings import settings

class LandslidesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        split: str = 'train',   
        bootstrapping_seed: Union[str, None] = None
    ):
        super().__init__()
        
        self.split = split
        self.data_path = Path(settings.default.user_path) / settings.default.data_path
        self.patch_size = 64
        self.bootstrapping_seed = bootstrapping_seed
        self.data_augmentation = False
        if self.data_augmentation and split == 'train':
            self.dataset = pd.read_csv(self.data_path / f'landslides_{split}_patches_{self.patch_size}_augmentation.csv')
        else:
            self.dataset = pd.read_csv(self.data_path / f'landslides_{split}_patches_{self.patch_size}.csv')
        self.bands_of_interest = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12', 'B8A']


        if self.bootstrapping_seed is not None and self.split in {"train", "val"}:
            print("Bootstrapping detected, generating samples ...")
            bootstrapping_sufix = f"-{self.bootstrapping_seed}.csv"
            bootstrapping_split_filepath = self.data_path / f'landslides_{split}_patches_{self.patch_size}{bootstrapping_sufix}'
            
            if not Path(bootstrapping_split_filepath).exists():
                split_data = pd.read_csv(self.data_path / f'landslides_{split}_patches_{self.patch_size}.csv')
                self.dataset = split_data.sample(
                    n=len(split_data), replace=True, random_state=int(self.bootstrapping_seed)
                )
                self.dataset.to_csv(
                    Path(self.data_path)
                    / f"landslides_{split}_patches_{self.patch_size}-{self.bootstrapping_seed}.csv",
                    index=False,
                )
                print(f"Bootstrapping file does not exist, generated {bootstrapping_split_filepath}")
            else:
                self.dataset = pd.read_csv(bootstrapping_split_filepath)
                print("Bootstrapping file does exist, loaded")

    def _normalization(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize the image by subtracting the mean and dividing by the standard deviation.

        Args:
            image: np.ndarray: The image to normalize.

        Returns:
            np.ndarray: The normalized image.
        """
        means = np.array(settings['landslide'].train["means"])
        stds = np.array(settings['landslide'].train["stds"])
        image = (image - means[:, None, None]) / stds[:, None, None]
        return image

    def _transform(self, image: np.ndarray) -> dict:
        """
        Transform the image to the desired size.

        Args:
            image: np.ndarray: The image to transform.

        Returns:
            dict: The transformed sample.
        """
        #image = x["x"]  # CHW
        new_chips = np.zeros((image.shape[0], settings['landslide'].dataloader.img_size, settings['landslide'].dataloader.img_size), dtype=np.float32)

        for i in range(image.shape[0]):
            # cv2.resize dsize receive the parameter(width, height), different from the img size of (H, W)
            new_slice = cv2.resize(  # pylint: disable=no-member
                image[i],
                (settings['landslide'].dataloader.img_size, settings['landslide'].dataloader.img_size),
                interpolation=cv2.INTER_NEAREST,  # pylint: disable=no-member
            )
            new_chips[i, :, :] = new_slice

        return new_chips

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx:int) -> dict:
        """
        Get sample from the dataset
        
        Args:
            idx: int: The index of the sample.

        Returns:
            dict: The samples
        """

        patch_info = self.dataset.iloc[idx]
        file = patch_info['file']
        date = patch_info['date']
        bbox = eval(patch_info['bbox'])

        image = np.zeros((10,self.patch_size,self.patch_size))
        for band_idx, band_name in enumerate(self.bands_of_interest):
            file_path = self.data_path / 'original_scenes' / patch_info['file'] / f'{file}__S2L2A__{date}__{band_name}.tif'
            with rasterio.open(file_path) as src:
                window = from_bounds(*bbox, transform=src.transform)
                image[band_idx] = src.read(window=window)[0]       
        
        mask_path = self.data_path / 'original_scenes' / patch_info['file'] / f'{file}__None__None__MASK.tif'
        with rasterio.open(mask_path) as src:
            window = from_bounds(*bbox, transform=src.transform)
            label = src.read(window=window)[0]  
        
        spatial_coords = eval(patch_info['bbox'])

        '''import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('S2 Post-Event RGB Image')
        plt.imshow(np.transpose(image[[2, 1, 0]]/10000 * 3, (1, 2, 0)))
        plt.axis('off') 
        plt.subplot(1, 2, 2)
        plt.title('Landslide Mask')
        plt.imshow(label, cmap='gray')
        plt.axis('off')
        plt.savefig(f'landslide_sample_{idx}.png')
        import sys; sys.exit()'''

        '''if self.data_augmentation:
            if patch_info['augmentation'] == 'Flip':
                r = np.random.rand()
                if r < 0.33:
                    pass
                elif r < 0.66:
                    image = np.flip(image, axis=2)   # horizontal
                else:
                    image = np.flip(image, axis=1)   # vertical

            elif patch_info['augmentation'] == 'Rot':
                from scipy.ndimage import rotate
                angle = np.random.uniform(-30, 30)  # degrees
                image = rotate(image, angle=angle, axes=(1, 2), reshape=False)'''

        if image.shape[-1] != settings['landslide'].dataloader.img_size:
            image = self._transform(image)

        image = self._normalization(image)
        #image = np.clip(image / 10000, 0,1)
        '''print(np.unique(image))
        import sys; sys.exit()'''

        sample = {
            "x": torch.tensor(image, dtype=torch.float32),
            "y": label.astype(np.int64),
            "spatial_coords": torch.tensor(spatial_coords, dtype=torch.float32),
        }


        return sample