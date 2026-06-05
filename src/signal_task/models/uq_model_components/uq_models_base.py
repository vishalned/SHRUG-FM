"""
This module provides the base class for Uncertainty Quantification (UQ) models.
It defines the common interface for various UQ methods, including ensemble-based
methods, models with density layers, and Monte Carlo Dropout.
"""

import abc

import torch
import torch.nn.functional as F
from torch import nn


class UQModel(abc.ABC, nn.Module):
    """
    Abstract base class for Uncertainty Quantification (UQ) models
    in deep learning binary classification.

    This class defines the common interface for various UQ methods,
    including ensemble-based methods, models with density layers,
    and Monte Carlo Dropout. The outputs are essential for their comparison
    in an evaluation framework.

    Concrete implementations must inherit from this class and implement
    all abstract methods.
    """

    def __init__(self, base_model: nn.Module, num_classes: int = 2):
        super().__init__()
        self.base_model = base_model
        self.target_size = [256, 256]
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2 for classification.")
        self.num_classes = num_classes  # Store the number of classes

    def scale_to_target_size(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scale the output to the target size.

        Args:
            x: A tensor of input data.

        Returns:
            A tensor of output data with the target size as stored in self.target_size.
        """
        if x.size()[-2:] != self.target_size:
            x = F.interpolate(
                x,
                size=self.target_size,
                mode="nearest",
            )
        return x

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the UQ model. This method should
        return whatever the training process needs.

        Args:
            x: A tensor of input data.

        Returns:
            UQModel specific output.
        """

    @abc.abstractmethod
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Based on the UQ method, it yields a single probability
        map for each pixel across classes. This is usually the
        mean or expected value of the UQ method
        for each pixel per classes

        Args:
            x: A tensor of input data.

        Returns:
            A tensor of predicted probabilities, typically of
            shape (batch_size, num_calasses, height, width).
        """

    @abc.abstractmethod
    def variance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the variance over the probabilities for the given inputs.

        Args:
            x: A tensor of input data.

        Returns:
            A tensor of variance values, typically of shape (batch_size,
            num_classes, height, width).
        """

    def entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the entropy of the predicted probabilities.

        Args:
            x: A tensor of input data.

        Returns:
            A tensor of entropy values, typically of shape (batch_size,
            height, width).
        """
        probabilities = self.predict_proba(
            x
        )  # Expected shape: (batch_size, num_classes, height, width)

        # Add a small epsilon for numerical stability to avoid log(0)
        epsilon = 1e-10
        probabilities = torch.clamp(probabilities, epsilon, 1.0 - epsilon)

        entropy_values = -torch.sum(probabilities * torch.log(probabilities), dim=1)

        return entropy_values

    @abc.abstractmethod
    def mutual_information(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the mutual information for predictions of given inputs.
        It is computed as the difference between the entropy of the expected
        value of the UQ method and expected value of the entropy.
        It is a proxy for the epistemic uncertainty of the predictions.

        Args:
            x: A tensor of input data.

        Returns:
            A tensor of mutual information values, typically of shape (batch_size,
            height, width).
        """

    @abc.abstractmethod
    def percentiles(
        self, x: torch.Tensor, q: torch.Tensor = torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95])
    ) -> torch.Tensor:
        """
        Computes the specified percentiles of the predictive distribution
        for the given inputs. This allows quantifying the spread of predictions.

        For multiclass, this typically means computing percentiles for the
        probability of *each* class at each pixel.

        Args:
            x: A tensor of input data.
            q: A 1D tensor of percentiles to compute (e.g., [0.05, 0.95] for 5th and 95th).
                Values should be between 0 and 1.

        Returns:
            A tensor of percentile values, typically of shape (batch_size, num_percentiles,
            num_classes, height, width).
        """
