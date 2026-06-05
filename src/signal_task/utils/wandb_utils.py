"""This module contains functions related to logging to wandb."""

import numpy as np
import torch
import torch.nn.functional as F

import wandb


def log_sample_image(
    batch_sample_img: torch.Tensor,
    batch_sample_mask: torch.Tensor,
    y_pred: torch.Tensor,
) -> None:
    """
    Log a sample image to wandb.

    Args:
        batch_sample_img: The input image batch.
        batch_sample_mask: The ground truth mask batch.
        y_pred: The model prediction batch.
    """

    # Get the first image from the batch
    sample_img = batch_sample_img[0].cpu().numpy()  # Shape: (C, H, W)
    sample_mask = batch_sample_mask[0].cpu().numpy()  # Shape: (H, W) or (C, H, W)
    sample_pred = y_pred[0].cpu().detach().numpy()  # Shape: (H, W) or (C, H, W)

    # Resize image to match mask dimensions
    sample_img = torch.from_numpy(sample_img).unsqueeze(0)  # Add batch dimension
    # Scale image to same size as mask
    sample_img = F.interpolate(sample_img, size=sample_mask.shape[-2:], mode="nearest-exact")
    sample_img = sample_img.squeeze(0).cpu().numpy()  # Back to (C, H, W)

    # Get the RGB bands and clip to reasonable values (removing extreme outliers)
    rgb_img = sample_img[[2, 1, 0]]  # Red, Green, Blue for natural color

    # Clip to 98th percentile to remove extreme values
    p98 = np.percentile(rgb_img, 98)
    rgb_img = np.clip(rgb_img, 0, p98)

    # Scale to 0-255
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min()) * 255
    rgb_img = rgb_img.astype(np.uint8)

    # Ensure mask is 2D by taking first channel if needed
    # GT mask can be 0 (land), 1 (water), or 2 (invalid)
    if len(sample_mask.shape) > 2:
        sample_mask = sample_mask[0]  # Take first channel if multi-channel

    # Get model prediction for this sample
    # predicted mask can only be 0 (land) or 1 (water)
    pred_mask = sample_pred.argmax(0)

    # Create a wandb Image with both the input and masks
    wandb.log(
        {
            "sample_data_gt": wandb.Image(
                # Transpose from (C,H,W) to (H,W,C) for wandb
                np.transpose(rgb_img, (1, 2, 0)),
                masks={
                    "ground_truth": {
                        "mask_data": sample_mask,
                    },
                    "prediction": {
                        "mask_data": pred_mask,
                    },
                },
            ),
        }
    )
