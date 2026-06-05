"""Testing code for the fire disaster."""

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import Accuracy, CalibrationError, F1Score, JaccardIndex
from tqdm import tqdm

from signal_task.analysis.reliability_plot import (
    calculate_adaptive_ece,
    plot_calibration,
)
from signal_task.utils.data_processing import process_input_bands
from signal_task.utils.score_utils import metric_scaled


class Tester:
    """Class for testing various disaster models with and without uncertainty quantification.

    This class provides functionality to test models on different disaster data, compute various
    metrics including F1 score, IoU, accuracy, and uncertainty metrics when applicable.
    It also supports visualization of results and calibration plots.
    """

    def __init__(self, disaster: str) -> None:
        """Initialize the Tester.

        Args:
            disaster (str): The type of disaster to test for.
        """
        self.disaster = disaster
        self.metric_f1 = F1Score(num_classes=2, task="binary")
        self.metric_iou = JaccardIndex(task="multiclass", num_classes=2)
        self.metric_macro_iou = JaccardIndex(task="multiclass", num_classes=2, average="macro")
        self.metric_macro_acc = Accuracy(task="multiclass", num_classes=2, average="macro")
        self.metric_calibration = CalibrationError(task="multiclass", num_classes=2)
        self.criterion = smp.losses.DiceLoss(mode="multiclass")

        self.uncertainty = None
        self.metrics_dataframe = pd.DataFrame()

        # Initialize some attributes as None
        self.model = None
        self.test_loader = None
        self.test_train_loader = None
        self.test_val_loader = None
        self.save_path = None
        self.debug = False

    def test(
        self,
        *,
        model: nn.Module,
        data: Any,
        uncertainty: bool,
        save_samples: bool,
        device: torch.device,
        save_path: Path,
        debug: bool = False,
    ) -> None:
        """Test the model on the dataset.

        Args:
            model (nn.Module): The model to test.
            data (Any): The data loader containing test data.
            uncertainty (bool): Whether to test with uncertainty quantification.
            device (torch.device): The device to run testing on.
            save_path (Path): Path to save test results.
            debug (bool, optional): Whether to run in debug mode. Defaults to False.
        """
        self.model = model
        self.uncertainty = uncertainty
        self.test_loader, _ = data.test_dataloader()
        self.save_path = save_path
        self.debug = debug

        if not self.uncertainty:
            self.metrics_dataframe = pd.DataFrame(
                columns=["Sample ID", "Set", "Location", "F1", "IoU", "Accuracy"]
            )
            self.__baseline_test(device, model.model_type)
        else:
            self.metrics_dataframe = pd.DataFrame(
                columns=[
                    "Sample ID",
                    "Set",
                    "Location",
                    "F1",
                    "IoU",
                    "ECE",
                    "Accuracy",
                    "Entropy",
                    "Variance",
                    "Mutual Information",
                    "Entropy_0.0_scaled",
                    "Entropy_0.05_scaled",
                    "Entropy_0.1_scaled",
                    "Entropy_0.2_scaled",
                    "Entropy_0.3_scaled",
                    "Entropy_0.4_scaled",
                    "Entropy_0.5_scaled",
                    "Variance_0.0_scaled",
                    "Variance_0.05_scaled",
                    "Variance_0.1_scaled",
                    "Variance_0.2_scaled",
                    "Variance_0.3_scaled",
                    "Variance_0.4_scaled",
                    "Variance_0.5_scaled",
                    "Mutual Information_0.0_scaled",
                    "Mutual Information_0.05_scaled",
                    "Mutual Information_0.1_scaled",
                    "Mutual Information_0.2_scaled",
                    "Mutual Information_0.3_scaled",
                    "Mutual Information_0.4_scaled",
                    "Mutual Information_0.5_scaled",
                ]
            )
            self.__uq_test(data, device, save_samples=save_samples)

        save_path_csv = Path(save_path) / "test_metrics.csv"
        save_path_csv.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_dataframe.to_csv(save_path_csv, index=False)

    # pylint: disable=too-many-locals
    def __baseline_test(
        self,
        device: torch.device,
        model_type: str,
    ) -> None:
        """Run baseline testing without uncertainty quantification.

        Args:
            device (torch.device): The device to run testing on.
            model_type (str): The type of model being tested.
        """
        total_pred, total_gt = [], []
        with torch.no_grad():
            for sample_idx, test_data in enumerate(self.test_loader):
                x_hls = test_data["x"].to(device)

                x = process_input_bands(
                    x_original=x_hls, model_type=model_type, disaster=self.disaster
                )

                mask = test_data.get("mask")
                # To do: if the noise mask not impact the final performance, then remove it.
                test_data.get("noise_mask")
                x_test = torch.cat([x, mask.to(device)], dim=1) if mask is not None else x
                y_test = test_data["y"].to(device)  # b1hw
                coord_test = tuple(t.item() for t in test_data.get("spatial_coords")[0])
                self.model.eval()

                logits_test = self.model(x_test)

                if logits_test.size()[-2:] != y_test.size()[-2:]:
                    logits_test = F.interpolate(
                        logits_test,
                        size=y_test.size()[-2:],
                        # check the note on
                        # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
                        mode="nearest-exact",
                    )

                # remove invalid pixels from testing
                if self.disaster == "flood":
                    logits_test = logits_test.permute(0, 2, 3, 1)[y_test < 2]
                    y_test = y_test[y_test < 2]

                self.criterion(logits_test, y_test)

                pred_test = torch.nn.functional.softmax(logits_test, dim=1)  # b2hw or b2N
                # Returns the indices of the maximum value of all elements in the input tensor
                pred_test = torch.argmax(pred_test, dim=1).int()  # bhw or bN

                total_pred.append(pred_test.flatten())
                total_gt.append(y_test.flatten().int())
                # total_gt.append(y_test.squeeze(1).flatten().int())

                sample_dictionary = {
                    "Sample ID": sample_idx,
                    "Set": "test",
                    "Location": coord_test,
                    "F1": self.metric_f1(
                        pred_test.detach().cpu().flatten(),
                        # y_test.detach().cpu().squeeze(1).flatten(),
                        y_test.detach().cpu().flatten(),
                    ).numpy(),
                    "IoU": self.metric_iou(
                        pred_test.detach().cpu().flatten(),
                        y_test.detach().cpu().flatten(),
                        # pred_test.detach().cpu(), y_test.detach().cpu().squeeze(1)
                    ).numpy(),
                    "Accuracy": "",
                }
                self.metrics_dataframe = pd.concat(
                    [self.metrics_dataframe, pd.DataFrame([sample_dictionary])], ignore_index=True
                )

                if self.debug:
                    break

            final_pred = torch.cat(total_pred, dim=0)
            final_gt = torch.cat(total_gt, dim=0)

            sample_dictionary = {
                "Sample ID": "Global",
                "Set": "test",
                "Location": None,
                "F1": self.metric_f1(final_pred.detach().cpu(), final_gt.detach().cpu()).numpy(),
                "IoU": self.metric_iou(final_pred.detach().cpu(), final_gt.detach().cpu()).numpy(),
                "Accuracy": self.metric_macro_acc(
                    final_pred.detach().cpu(), final_gt.detach().cpu()
                ).numpy(),
            }
            self.metrics_dataframe = pd.concat(
                [self.metrics_dataframe, pd.DataFrame([sample_dictionary])], ignore_index=True
            )

        return self.metrics_dataframe

    def save_samples(self, save_path: Path, sample_idx: int, split: str, x_test: torch.Tensor, pred_test: torch.Tensor, probabilities: torch.Tensor, y_test: torch.Tensor) -> None:
        """Save sample predictions and probabilities for later analysis.

        Args:
            save_path (Path): The path to save the samples.
            sample_idx (int): The index of the sample.
            x_test (torch.Tensor): The input data for the test sample.
            pred_test (torch.Tensor): The predicted mask for the test sample.
            probabilities (torch.Tensor): The predicted probabilities for the test sample.
            y_test (torch.Tensor): The ground truth mask for the test sample.
        """
        os.makedirs(save_path / f"predictions_{split}", exist_ok=True)
        np.savez(
            save_path / f"predictions_{split}" / f"prediction_{sample_idx}",
            original_image=x_test.detach().cpu().numpy(),
            prediction_scaled=pred_test.detach().cpu().numpy(),
            probabilities_scaled=probabilities.detach().cpu().numpy(),
            prediction=torch.argmax(probabilities, dim=1).int().detach().cpu().numpy(),
            probabilities=probabilities.detach().cpu().numpy(),
            entropy=self.model.entropy(x_test).detach().cpu().numpy(),
            variances=self.model.variance(x_test).detach().cpu().numpy(),
            mutual_information=self.model.mutual_information(x_test).detach().cpu().numpy(),
            ground_truth=y_test.detach().cpu().numpy(),
        )

    def __uq_test(self, data: Any, save_samples: bool, device: torch.device, plot_calibration: bool = False) -> None:
        """Test the model with uncertainty quantification.

        Args:
            data (Any): The data loader containing test data.
            save_samples (bool): Whether to save sample predictions and probabilities.
            device (torch.device): The device to run testing on.
        """
        self.test_loader, _ = data.test_dataloader()
        self.train_loader, _ = data.train_dataloader()
        self.val_loader, _ = data.val_dataloader()
        splits = ["train", "val", "test"]

        with torch.no_grad():
            for split_idx, loader in enumerate(
                [self.train_loader, self.val_loader, self.test_loader]
            ):
                print(f"Testing {splits[split_idx]} set")
                for sample_idx, test_data in enumerate(tqdm(loader)):
                    mask = test_data.get("mask")
                    x_test = (
                        torch.cat([test_data["x"].to(device), mask.to(device)], dim=1)
                        if mask is not None
                        else test_data["x"].to(device)
                    )
                    y_test = test_data["y"].to(device)
                    coord_test = tuple(t.item() for t in test_data.get("spatial_coords")[0])
                    self.model.eval()

                    probabilities_raw = self.model.predict_proba(x_test)
                    if probabilities_raw.size()[-2:] != y_test.size()[-2:]:
                        probabilities = F.interpolate(
                            probabilities_raw,
                            size=y_test.size()[-2:],
                            mode="nearest",
                        )
                    else:
                        probabilities = probabilities_raw

                    if self.disaster == "flood":
                        probabilities = probabilities.permute(0, 2, 3, 1)[y_test < 2]
                        y_test = y_test[y_test < 2]

                    pred_test = torch.argmax(probabilities, dim=1).int().detach().cpu()
                    y_test = y_test.detach().cpu()
                    probabilities = probabilities.detach().cpu()
                    entropy = self.model.entropy(x_test).detach().cpu().numpy()
                    variance = self.model.variance(x_test).detach().cpu().numpy()
                    mutual_information = (
                        self.model.mutual_information(x_test).detach().cpu().numpy()
                    )

                    if save_samples:
                        self.save_samples(self.save_path, sample_idx, splits[split_idx], x_test, pred_test, probabilities, y_test)

                    if self.disaster == "fire":
                        y_test = y_test[0]

                    sample_dictionary = {
                        "Sample ID": sample_idx,
                        "Set": splits[split_idx],
                        "Location": coord_test,
                        "F1": self.metric_f1(
                            pred_test.flatten(), y_test.flatten()
                        ).item() if torch.any(y_test == 1) else float("nan"),
                        "IoU": self.metric_iou(pred_test.flatten(), y_test.flatten()).item(),
                        "ECE": self.metric_calibration(probabilities, y_test).item(),
                        "ECE_adaptive": calculate_adaptive_ece(
                            probabilities[:, 1].flatten(), y_test.flatten()
                        ),
                        "Accuracy": self.metric_macro_acc(pred_test, y_test).item(),
                        "Probability": np.mean(probabilities.numpy()),
                        "Entropy": np.mean(entropy),
                        "Variance": np.mean(variance),
                        "Mutual Information": np.mean(mutual_information),
                        "Pred_Area": pred_test.flatten().sum().item(),
                        "Prob_Area": (probabilities.numpy() > 0.01).sum()}
                    
                    if self.disaster == "flood":
                        # We return to the original shape (without filtering no-data samples) to compute the flags
                        y_test = test_data["y"].to(device)
                        probabilities = self.model.predict_proba(x_test)
                        if probabilities.size()[-2:] != y_test.size()[-2:]:
                            probabilities = F.interpolate(
                                probabilities,
                                size=y_test.size()[-2:],
                                mode="nearest",
                            )
                        y_test = y_test.detach().cpu()
                        probabilities = probabilities.detach().cpu()
                        pred_test = torch.argmax(probabilities, dim=1).int().detach().cpu()

                    sample_dictionary.update({
                        **{
                            f"{metric}_{threshold}_scaled": metric_scaled(
                                metric_data, probabilities[:, 1].numpy(), threshold
                            ).item()
                            for metric, metric_data in [
                                ("Entropy", entropy),
                                ("Variance", variance),
                                ("Mutual Information", mutual_information),
                            ]
                            for threshold in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
                        },
                        "Probability_gt_scaled": metric_scaled(
                            probabilities.numpy(), test_data["y"].squeeze(1).numpy()
                        ).item(),
                        "Entropy_gt_scaled": metric_scaled(
                            entropy, y_test.squeeze(1).numpy()
                        ).item(),
                        "Variance_gt_scaled": metric_scaled(
                            variance, y_test.squeeze(1).numpy()
                        ).item(),
                        "Mutual Information_gt_scaled": metric_scaled(
                            mutual_information, y_test.squeeze(1).numpy()
                        ).item(),
                        "Probability_pred_scaled": metric_scaled(
                            probabilities.numpy(), pred_test.numpy()
                        ).item(),
                        "Entropy_pred_scaled": metric_scaled(entropy, pred_test.numpy()).item(),
                        "Variance_pred_scaled": metric_scaled(variance, pred_test.numpy()).item(),
                        "Mutual Information_pred_scaled": metric_scaled(
                            mutual_information, pred_test.numpy()
                        ).item(),
                        "Pred_Area": pred_test.flatten().sum().item(),
                        "GT_Area": y_test.squeeze(1).flatten().sum().item(),
                        "Prob_Area": (probabilities.numpy() > 0.01).sum(),
                    })

                    self.metrics_dataframe = pd.concat(
                        [self.metrics_dataframe, pd.DataFrame([sample_dictionary])],
                        ignore_index=True,
                    )

                    if plot_calibration:
                        binning = ['uniform', 'adaptive']
                        for b in binning:
                            plot_calibration(
                                probabilities[:, 1].flatten(),
                                y_test.squeeze(1).flatten(),
                                save_path=self.save_path,
                                tag=f"{splits[split_idx]}_{sample_idx}",
                                sample_dictionary=sample_dictionary,
                                binning=b,
                            )

                    plt.close("all")

                    if self.debug and sample_idx > 5:
                        break

    def plot(
        self,
        *,
        sample_idx: int,
        y_test: torch.Tensor,
        x_test: torch.Tensor,
        pred_test: torch.Tensor,
        stats: dict,
    ) -> None:
        """Plot the results for a single sample.

        Args:
            sample_idx (int): The index of the sample.
            y_test (torch.Tensor): The ground truth mask.
            x_test (torch.Tensor): The input data.
            pred_test (torch.Tensor): The predicted mask.
            stats (dict): The statistics of the input data.
        """
        if sample_idx % 10 == 0:
            mean = np.array([stats["means"][i] for i in [5, 3, 2]])
            std = np.array([stats["stds"][i] for i in [5, 3, 2]])

            target_test = y_test.detach().cpu().numpy()[0, 0]
            # only visualize the 1st item of a batch
            x_test = x_test.detach().cpu().numpy()[:, [5, 3, 2], :, :][0]
            output_test = pred_test.detach().cpu().numpy()[0]

            fig, axes = plt.subplots(3, 1, figsize=(5, 15))
            cmap = plt.cm.get_cmap("viridis", 2)  # Discrete colormap with 3 categories
            labels = ["No \nburned", "Burned"]

            norm_backed_data = x_test * std[:, None, None] + mean[:, None, None]
            norm_backed_data = (norm_backed_data - np.amin(norm_backed_data)) / (
                np.amax(norm_backed_data) - np.amin(norm_backed_data)
            )
            im0 = axes[0].imshow(norm_backed_data.transpose((1, 2, 0)))
            cbar0 = fig.colorbar(im0, ax=axes[0], orientation="vertical", pad=0.05)
            cbar0.set_label("Normalized reflectance", fontsize=14)
            axes[0].set_title("Input", fontsize=14)
            axes[0].axis("off")

            im1 = axes[1].imshow(target_test, vmin=0, vmax=1.0, cmap=cmap)
            cbar1 = fig.colorbar(im1, ax=axes[1], orientation="vertical", pad=0.05)
            cbar1.set_ticks([0, 1])  # Explicitly set the tick locations
            cbar1.set_ticklabels(labels, fontsize=14)  # Set the tick labels
            for tick in cbar1.ax.get_yticklabels():  # Rotate tick labels
                tick.set_rotation(90)
            axes[1].set_title("Ground Truth", fontsize=14)
            axes[1].axis("off")

            im2 = axes[2].imshow(output_test, vmin=0, vmax=1.0, cmap=cmap)
            cbar2 = fig.colorbar(im2, ax=axes[2], orientation="vertical", pad=0.05)
            cbar2.set_ticks([0, 1])  # Explicitly set the tick locations
            cbar2.set_ticklabels(labels, fontsize=14)  # Set the tick labels
            axes[2].set_title("Prediction", fontsize=14)
            axes[2].axis("off")
