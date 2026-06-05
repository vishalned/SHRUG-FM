from logging import config
import torchvision.transforms as T
from ssl4eo_wrapper import SSL4EOS12Wrapper, collate_skip_none
from torch.utils.data import DataLoader
from create_embeddings import CreateEmbeddings
from utils import DictImageTransform
import sys

from signal_task.dataset.floods.dataset_setup import get_dataset
from config.settings import settings


dataset_name = "landslide" # choices: burnscar, flood, landslide
root = "/data/databases/SSL4EO-S12/ssl4eo-s12/" # Data folder
split = "s2c"
seasons = 1
download = False
checksum = False
bands = "rgb"
batch_size = 64
crop_size = [224, 224] # has to be 224 since the model is pretrained on 224x224 images
prefix = f"{dataset_name}_rgb"
folder_path = "embeddings/"


S2C_MEAN = [
    1605.57504906, 
    1390.78157673,
    1314.8729939,
    1363.52445545,
    1549.44374991,
    2091.74883118,
    2371.7172463,
    2299.90463006,
    2560.29504086,
    830.06605044,   
    22.10351321,
    2177.07172323,
    1524.06546312,
]

S2C_STD = [
    786.78685367,
    850.34818441,
    875.06484736,
    1138.84957046,
    1122.17775652,
    1161.59187054,
    1274.39184232,
    1248.42891965,
    1345.52684884,
    577.31607053,
    51.15431158,
    1336.09932639,
    1136.53823676,
]


transforms = T.Compose(
    [
        T.CenterCrop(crop_size),
        T.Normalize(S2C_MEAN, S2C_STD),
    ]
)
dict_transforms = DictImageTransform(transforms)

if dataset_name == "burnscar":
    from fire_wrapper import HlsFire
    dataset = HlsFire(
        split="val", # Define data split
        data_path=root,
        bootstrapping_seed=None,
        )

elif dataset_name == 'ssl4eo':
    dataset = SSL4EOS12Wrapper(
        root=root, 
        split=split, 
        seasons=seasons, 
        transforms=dict_transforms,
        download=download, 
        checksum=checksum, 
        bands=bands
        )
elif dataset_name == 'flood':
    dm = get_dataset(
        settings.flood.dataloader,
        root,
        None, 
    )
    dm.prepare_data()
    dataset = dm.val_dataset # Define data split

elif dataset_name == 'landslide':
    from landslides_wrapper import LandslidesDataset
    dataset = LandslidesDataset(data_path = root, split='test', bootstrapping_seed=None) # Define data split

dataloader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    num_workers=4, 
    pin_memory=False, 
    drop_last=False, 
    shuffle=False, 
    collate_fn=collate_skip_none
    )

embedding_creator = CreateEmbeddings(
    dataset_name=dataset_name,
    num_channels=3,
    save_path=folder_path,
    prefix=prefix
    )

embedding_creator.get_embeddings_dl(dataloader)