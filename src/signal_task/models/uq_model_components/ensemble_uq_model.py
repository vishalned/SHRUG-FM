"""
This module provides an ensemble of UQ models for uncertainty quantification.
It implements the UQModel interface and aggregates predictions from multiple
base classifiers to estimate uncertainty for binary classification tasks.
"""

import torch
from torch import nn

from signal_task.models.uq_model_components.uq_models_base import UQModel


class EnsembleUQModel(UQModel):
    """
    Concrete implementation of UQModel using an ensemble of classifiers.

    This model aggregates predictions from multiple base classifiers to
    estimate uncertainty for binary classification tasks.

    Each member of the ensemble is expected to be a PyTorch nn.Module
    that, when called with an input tensor, produces a single probability
    (or logit that can be sigmoided) for the positive class.
    """

    def __init__(self, ensemble_members: list[nn.Module], num_classes: int = 2):
        """
        Initializes the EnsembleUQModel.

        Args:
            ensemble_members: A list of PyTorch nn.Module instances,
                              each representing a classifier in the ensemble.
                              Each member should be trained independently.
        """
        # Create a dummy base model for the parent class (not used in ensemble)
        dummy_model = nn.Identity()
        super().__init__(base_model=dummy_model, num_classes=num_classes)

        # Store ensemble members as a ModuleList so PyTorch correctly registers them
        # and their parameters for operations like .to(device), .eval(), .train()
        self.ensemble_members = nn.ModuleList(ensemble_members)
        self.num_members = len(ensemble_members)

    def _get_raw_ensemble_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Helper method to get raw predictions (probabilities) from all ensemble members.

        Ensures all members are in evaluation mode and collects their outputs.
        It assumes each member's forward pass outputs a probability map (0-1)
        of shape (batch_size, num_classes, height, width) for binary segmentation.

        Args:
            x: A tensor of input data.

        Returns:
            A tensor of shape (num_ensemble_members, batch_size, num_classes, height, width),
            where the first dimension represents the prediction from one ensemble member.
        """
        all_member_predictions = []
        for member in self.ensemble_members:
            member.eval()
            with torch.no_grad():
                prob_output = member(x)
                all_member_predictions.append(prob_output)

        # Stack predictions to get (num_ensemble_members, batch_size, num_classes, height, width)
        return torch.stack(all_member_predictions, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the EnsembleUQModel.

        Args:
            x: A tensor of input data.

        Returns:
            A tensor of shape (batch_size, num_classes, height, width).
        """
        ensemble_predictions = self._get_raw_ensemble_predictions(x)
        return ensemble_predictions.mean(dim=0)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mean of the probabilities from all ensemble members. Essentially, the forward pass.

        Args:
            x: A tensor of input image data.

        Returns:
            A tensor of predicted probability maps, typically of shape
            (batch_size, num_classes, height, width).
        """
        return self.forward(x)

    def variance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the variance over the probabilities for the given inputs
        across the ensemble members, resulting in a variance map.

        Args:
            x: A tensor of input image data.

        Returns:
            A tensor of variance maps, typically of shape (batch_size, num_classes, height, width).
        """
        ensemble_predictions = self._get_raw_ensemble_predictions(x)
        return ensemble_predictions.var(dim=0)

    def mutual_information(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the mutual information for pixel-wise predictions of given inputs.
        It is computed as the difference between the entropy of the expected
        value of the UQ method (mean prediction) and expected value of the entropy
        across individual ensemble members.
        It is a proxy for the epistemic uncertainty of the predictions.
        This is computed element-wise for each pixel.

        Args:
            x: A tensor of input image data.

        Returns:
            A tensor of mutual information maps, typically of shape (batch_size, height, width).
        """
        ensemble_probs = self._get_raw_ensemble_predictions(x)

        # 1. Entropy of Expected (H_E): H(E[p])
        entropy_of_expected = self.entropy(x)

        # 2. Expected Entropy (E_H): E[H(p_i)]
        epsilon = 1e-10
        clamped_ensemble_probs = torch.clamp(ensemble_probs, epsilon, 1.0 - epsilon)
        individual_entropies = -torch.sum(
            clamped_ensemble_probs * torch.log(clamped_ensemble_probs), dim=2
        )

        expected_entropy = individual_entropies.mean(dim=0)

        # MI = H_E - E_H
        mutual_info = entropy_of_expected - expected_entropy

        # Mutual information should be non-negative. Small negative values can occur due to
        # numerical precision. Clamp to zero.
        return torch.clamp(mutual_info, min=0.0)

    def percentiles(
        self, x: torch.Tensor, q: torch.Tensor = torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95])
    ) -> torch.Tensor:
        """
        Computes the specified percentiles of the predictive distribution
        for the given inputs across the ensemble members, resulting in a percentile map.

        Args:
            x: A tensor of input image data.
            q: A 1D tensor of percentiles to compute (e.g., [0.05, 0.95] for 5th and 95th).
               Values should be between 0 and 1.

        Returns:
            A tensor of percentile maps, typically of shape
            (batch_size, num_percentiles, height, width) for easier pixel-wise access.
        """
        ensemble_probs = self._get_raw_ensemble_predictions(x)
        # (N_members, B, C, H, W) -> (B, C, H, W, N_members)
        ensemble_probs_permuted = ensemble_probs.permute(1, 2, 3, 4, 0)
        if not isinstance(q, torch.Tensor):
            q = torch.tensor(q, dtype=torch.float32, device=ensemble_probs.device)

        # torch.quantile(input, q, dim=1) will return (len(q), B, H, W)
        percentile_values = torch.quantile(ensemble_probs_permuted, q, dim=-1)

        # Permute back to the desired output shape:
        # From (num_percentiles, batch_size, num_classes, height, width)
        # To   (batch_size, num_percentiles, num_classes, height, width)
        percentile_values = percentile_values.permute(1, 0, 2, 3, 4)
        return percentile_values
