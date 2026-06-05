import torch
from torch.utils.data import DataLoader

from fm_extremes_uncertainty.config.settings import settings
from fm_extremes_uncertainty.dataset.landslides.dataset import LandslidesDataset


class LandslidesDataloader:
    def __init__(
        self,
        seed: int = None,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.bootstrapping_seed = seed
    
    def get_data_loader(
        self,
        data_loader_type: str,
    ) -> DataLoader:
        
        if data_loader_type == "train":
            dataset = LandslidesDataset(split='train', bootstrapping_seed=self.bootstrapping_seed)
            return DataLoader(
                dataset,
                batch_size=settings['landslide'].dataloader.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                drop_last=True,
            ), None
        elif data_loader_type == "val":
            dataset = LandslidesDataset(split='val', bootstrapping_seed=self.bootstrapping_seed)
            return DataLoader(
                dataset,
                batch_size=settings['landslide'].dataloader.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                drop_last=False,
            ),None
        elif data_loader_type == "test":
            dataset = LandslidesDataset(split='test', bootstrapping_seed=self.bootstrapping_seed)
            return DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                drop_last=False,
            ), None

    def train_dataloader(self) -> DataLoader:
        return self.get_data_loader(data_loader_type="train")
        
    def val_dataloader(self) -> DataLoader:
        return self.get_data_loader(data_loader_type="val")
    
    def test_dataloader(self) -> DataLoader:
        return self.get_data_loader(data_loader_type="test")