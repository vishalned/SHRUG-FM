"""Utilities for the floods dataset."""

import json
import os
from typing import Any, Dict

import pandas as pd


def convert_metadata_csv_to_json(csv_path: str, json_path: str) -> None:
    """
    Convert the metadata CSV file to a JSON file with the structure expected by the
    WorldFloodsDataModule.

    Args:
        csv_path: str: The path to the metadata CSV file.
        json_path: str: The path to save the JSON file.

    """
    out: Dict[str, Any] = {}
    modalities = ["S2", "gt"]
    csv = pd.read_csv(csv_path)

    for split in csv.split.unique():
        out[split] = {}
        files = csv[csv.split == split]["event id"]
        for mod in modalities:
            out[split][mod] = [os.path.join(split, mod, f"{fn}.tif") for fn in files.to_list()]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
