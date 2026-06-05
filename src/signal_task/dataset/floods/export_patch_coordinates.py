from ml4floods.data.worldfloods.configs import CHANNELS_CONFIGURATIONS
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fm_extremes_uncertainty.dataset.floods import transformations
from fm_extremes_uncertainty.dataset.floods.dataset_setup import (
    filter_windows_fun,
    process_filename_train_test,
)
from fm_extremes_uncertainty.dataset.floods.lightning import WorldFloodsDataModule

download = {"train": False, "val": False, "test": False}
path_to_splits = "/home/rubencartuyvels/buckets/raw-data-2/raw_downstream/floods/worldfloods_v2/"
train_test_split_file = "/home/rubencartuyvels/buckets/raw-data-2/raw_downstream/floods/worldfloods_v2/train_test_split_from_csv_v2.json"
windows_file = "/home/rubencartuyvels/buckets/raw-data-2/raw_downstream/floods/worldfloods_v2/windows_train_test_split_from_csv_v2_224_th10.json"
filenames_train_test = process_filename_train_test(
    train_test_split_file,
    input_folder="S2",
    target_folder="gt",
    bucket_id=None,
    path_to_splits=path_to_splits,
    download=download,
)

filter_windows_config = filter_windows_fun(
    "v2",
    train_test_split_file,
    windows_file=windows_file,
    threshold_clouds=0.1,
    local_destination_dir=path_to_splits,
)

datamodule = WorldFloodsDataModule(
    filenames_train_test=filenames_train_test,
    path_to_splits=path_to_splits,
    input_folder="S2",
    target_folder="gt",
    train_transformations=transformations.ToTensor(),
    test_transformations=transformations.ToTensor(),
    bands=CHANNELS_CONFIGURATIONS["all"],
    add_mndwi_input=False,
    num_workers=4,
    window_size=[224, 224],
    batch_size=4,
    filter_windows=filter_windows_config,
    bootstrap_seed=None,
)
datamodule.setup()
datamodule.prepare_data()


def loader(dataset: Dataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=8,
        num_workers=8,
        shuffle=False,
    )


train_loader = loader(datamodule.train_dataset)
val_loader = loader(datamodule.val_dataset)
test_loader = loader(datamodule.test_dataset)

with open("./output/floods_patch_coords.txt", "w") as f:
    for split, dl in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        for batch in tqdm(dl, desc=split):
            # save all middlepoints of the tuple in "spatial_coords" to a file
            batch_coords, batch_files = batch["spatial_coords"], batch["filename"]
            for coords, filename in zip(batch_coords, batch_files):
                x1, y1, x2, y2 = coords.tolist()
                f.write(f"{split},{filename},{x1 + x2 / 2},{y1 + y2 / 2}\n")
            # break

print("Spatial coordinates saved to output/floods_patch_coords.txt")
