"""
Miscellaneous utility functions.
"""

import random

import geopandas as gpd
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set for random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def results_df_to_gdf(src_path: str, dest_path: str) -> None:
    """
    Function to translate csv results to common geojson format

    Args:
        src_path: Path to the csv file containing the dist team results
        dest_path: Path to save the resulting geojson
    """

    user = "maria"
    base_gdf = gpd.read_file(
        f"/home/{user}/2025-esl-extreme-environments-raw-data/geometries/burnscar_hydro.geojson"
    )

    df = pd.read_csv(src_path)

    df["Location_clean"] = df["Location"].str.replace(r"[()\s]", "", regex=True)
    df[["Longitude", "Latitude"]] = df["Location_clean"].str.split(",", expand=True).astype(float)
    df.drop(columns="Location_clean", inplace=True)

    metrics_of_interest = [
        "Longitude",
        "Latitude",
        "F1",
        "IoU",
        "Accuracy",
        "ECE_adaptive",
        "Entropy",
        "Variance",
        "Probability_pred_normalized",
        "Entropy_pred_normalized",
        "Variance_pred_normalized",
        "Mutual Information_pred_normalized",
    ]
    df = df[metrics_of_interest]

    results_gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326"
    ).drop(columns=["Longitude", "Latitude"])

    merged_gdf = base_gdf.merge(results_gdf, on="geometry", how="left")

    merged_gdf.to_file(dest_path)
