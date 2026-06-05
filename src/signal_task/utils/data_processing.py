"""This file contains the data processing functions."""

import torch


def process_input_bands(x_original: torch.Tensor, model_type: str, disaster: str) -> torch.Tensor:
    """
    This functions maps the original bands to what is expected by the model.

    Args:
        x_original: The original bands.
        model_type: The model type.
        disaster: The disaster.

    Returns:
        The processed bands.

    Raises:
        ValueError: If the model type or disaster is not supported.
    """
    if model_type not in ["ssl4eo"]:
        raise ValueError(f"Expected model_type to be 'ssl4eo', got {model_type}")

    if disaster not in ["fire", "flood", "landslide"]:
        raise ValueError(f"Disaster {disaster} not supported")

    if disaster == "fire":
        num_model_bands = {"ssl4eo": 13}[model_type]
        x_model = torch.zeros(
            (x_original.shape[0], num_model_bands, x_original.shape[2], x_original.shape[3]),
            dtype=x_original.dtype,
            device=x_original.device,
        )
        x_model[:, 1] = x_original[:, 0]  # B02 (Blue)
        x_model[:, 2] = x_original[:, 1]  # B03 (Green)
        x_model[:, 3] = x_original[:, 2]  # B04 (Red)
        x_model[:, 8] = x_original[:, 3]  # B08A (NIR narrow)
        x_model[:, 11] = x_original[:, 4]  # B11 (SWIR 1)
        x_model[:, 12] = x_original[:, 5]  # B12 (SWIR 2)
        return x_model
    
    if disaster == "landslide":
        num_model_bands = {"ssl4eo": 13}[model_type]
        x_model = torch.zeros(
            (x_original.shape[0], num_model_bands, x_original.shape[2], x_original.shape[3]),
            dtype=x_original.dtype,
            device=x_original.device,
        )

        ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12', 'B8A']
        x_model[:, 1] = x_original[:, 0]  # B02 (Blue)
        x_model[:, 2] = x_original[:, 1]  # B03 (Green)
        x_model[:, 3] = x_original[:, 2]  # B04 (Red)
        x_model[:, 4] = x_original[:, 3]  # B05
        x_model[:, 5] = x_original[:, 4]  # B06
        x_model[:, 6] = x_original[:, 5]  # B07
        x_model[:, 7] = x_original[:, 6]  # B08
        x_model[:, 8] = x_original[:, 9]  # B08A (NIR narrow)
        x_model[:, 11] = x_original[:, 7]  # B11 (SWIR 1)
        x_model[:, 12] = x_original[:, 8]  # B12 (SWIR 2)
        return x_model

    # floods uses all S2 bands
    return x_original
