"""This module contains classes and methods used for training deep learning models.

The module provides various utility functions and classes for training, including:
- Loss functions: TiledMSE, TiledMAPE, TiledMAPE2
- Normalization layers: LayerNorm, GRN
- Attention mechanisms: SE_Block variants
- Activation and normalization utilities
- Training schedulers and utilities
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn


class TiledMSE(nn.Module):
    """Mean Squared Error loss that combines full-image and pixel-level MSE.

    This loss function calculates MSE at both the full image level and pixel level,
    then combines them using a weighted average:
    result = (sum_mse * (1 - bias)) + (mse * bias)

    Args:
        bias (float, optional): Weight for pixel-level MSE. Defaults to 0.8.
        scale_term (float, optional): Scale factor for inputs. Defaults to 1.0.
    """

    def __init__(self, bias=0.8, scale_term=1.0):
        super().__init__()
        self.bias = bias
        self.scale_term = scale_term

    def forward(self, y_pred, y_true):
        """Calculate the weighted MSE loss.

        Args:
            y_pred (torch.Tensor): Predicted values
            y_true (torch.Tensor): Ground truth values

        Returns:
            torch.Tensor: Weighted MSE loss value
        """
        y_true = (y_true + 1) * self.scale_term
        y_pred = (y_pred + 1) * self.scale_term

        y_pred_sum = torch.sum(y_pred, dim=(2, 3)) / (
            y_pred.shape[1] * y_pred.shape[2] * y_pred.shape[3]
        )
        y_true_sum = torch.sum(y_true, dim=(2, 3)) / (
            y_true.shape[1] * y_true.shape[2] * y_true.shape[3]
        )

        sum_mse = torch.mean((y_pred_sum - y_true_sum) ** 2, dim=1).mean()
        mse = torch.mean((y_pred - y_true) ** 2, dim=1).mean()

        weighted = (sum_mse * (1 - self.bias)) + (mse * self.bias)
        return weighted


class TiledMAPE(nn.Module):
    """Mean Absolute Percentage Error loss that combines full-image and pixel-level MAPE.

    This loss function calculates MAPE at both the full image level and pixel level,
    then combines them using a weighted average:
    result = (mape_sum * (1 - bias)) + (mape * bias)

    Args:
        beta (float, optional): Small constant to prevent division by zero. Defaults to 0.1.
        bias (float, optional): Weight for pixel-level MAPE. Defaults to 0.8.
    """

    def __init__(self, beta=0.1, bias=0.8):
        super().__init__()
        self.beta = beta
        self.bias = bias
        self.eps = 1e-6

    def forward(self, y_pred, y_true):
        """Calculate the weighted MAPE loss.

        Args:
            y_pred (torch.Tensor): Predicted values
            y_true (torch.Tensor): Ground truth values

        Returns:
            torch.Tensor: Weighted MAPE loss value
        """
        y_pred_sum = torch.sum(y_pred, dim=(2, 3)) / (
            y_pred.shape[1] * y_pred.shape[2] * y_pred.shape[3]
        )
        y_true_sum = torch.sum(y_true, dim=(2, 3)) / (
            y_true.shape[1] * y_true.shape[2] * y_true.shape[3]
        )

        mape_sum = torch.mean(
            torch.abs((y_true_sum - y_pred_sum) / (y_true_sum + self.eps + self.beta)),
            dim=1,
        ).mean()
        mape = torch.mean(
            torch.abs((y_true - y_pred) / (y_true + self.eps + self.beta)), dim=1
        ).mean()

        weighted = (mape_sum * (1 - self.bias)) + (mape * self.bias)
        return weighted


class TiledMAPE2(nn.Module):
    """Enhanced Mean Absolute Percentage Error loss with improved stability.

    This version uses a maximum operation to prevent division by very small numbers
    and calculates the weighted average percentage error (WAPE) at the full image level.

    Args:
        beta (float, optional): Small constant to prevent division by zero. Defaults to 0.1.
        bias (float, optional): Weight for pixel-level MAPE. Defaults to 0.8.
    """

    def __init__(self, beta=0.1, bias=0.8):
        super().__init__()
        self.beta = beta
        self.bias = bias
        self.eps = 1e-6

    def forward(self, y_pred, y_true):
        """Calculate the enhanced MAPE loss.

        Args:
            y_pred (torch.Tensor): Predicted values
            y_true (torch.Tensor): Ground truth values

        Returns:
            torch.Tensor: Enhanced MAPE loss value
        """
        eps = torch.Tensor([self.eps]).to(y_pred.device)
        y_true_sum = torch.sum(y_true, dim=(2, 3)) / (
            y_true.shape[1] * y_true.shape[2] * y_true.shape[3]
        )

        abs_diff = torch.abs(y_true - y_pred)
        abs_diff_sum = torch.sum(abs_diff, dim=(2, 3)) / (
            y_pred.shape[1] * y_pred.shape[2] * y_pred.shape[3]
        )

        wape = torch.mean(abs_diff_sum / torch.maximum(y_true_sum + self.beta, eps), dim=1).mean()
        mape = torch.mean(abs_diff / torch.maximum(y_true + self.beta, eps), dim=1).mean()

        weighted = (wape * (1 - self.bias)) + (mape * self.bias)
        return weighted


def drop_path(x, keep_prob=1.0, inplace=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Args:
        x (torch.Tensor): Input tensor
        keep_prob (float, optional): Probability of keeping a path. Defaults to 1.0.
        inplace (bool, optional): Whether to perform operation in-place. Defaults to False.

    Returns:
        torch.Tensor: Output after applying stochastic depth
    """
    mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(mask_shape).bernoulli_(keep_prob)
    mask.div_(keep_prob)

    if inplace:
        x.mul_(mask)
        return x
    return x * mask


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Args:
        p (float, optional): Probability of dropping the path. Defaults to 0.5.
        inplace (bool, optional): Whether to perform operation in-place. Defaults to False.
    """

    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        """Apply stochastic depth to input tensor.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output after applying stochastic depth
        """
        if self.training and self.p > 0:
            x = drop_path(x, 1 - self.p, self.inplace)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError

        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LayerNorm to input tensor.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output after applying LayerNorm

        Raises:
            ValueError: If data format is invalid
        """
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        if self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

        raise ValueError(f"Invalid data format: {self.data_format}")


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim, channel_first=False):
        super().__init__()
        self.channel_first = channel_first
        if self.channel_first:
            self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        else:
            self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GRN to input tensor.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output after applying GRN
        """
        if self.channel_first:
            g_x = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
            n_x = g_x / (g_x.mean(dim=1, keepdim=True) + 1e-6)
        else:
            g_x = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
            n_x = g_x / (g_x.mean(dim=-1, keepdim=True) + 1e-6)

        return self.gamma * (x * n_x) + self.beta + x


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    *,
    warmup_epochs: int = 0,
    start_warmup_value: float = 0,
    warmup_steps: int = -1,
):
    """Create a cosine learning rate schedule with optional warmup.

    Args:
        base_value (float): Initial learning rate after warmup
        final_value (float): Final learning rate
        epochs (int): Total number of epochs
        warmup_epochs (int, optional): Number of warmup epochs. Defaults to 0.
        start_warmup_value (float, optional): Initial warmup learning rate. Defaults to 0.
        warmup_steps (int, optional): Number of warmup steps. Defaults to -1.

    Returns:
        np.ndarray: Array containing the learning rate schedule
    """
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs
    if warmup_steps > 0:
        warmup_iters = warmup_steps

    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs - warmup_iters)
    schedule = np.array(
        [
            final_value
            + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters))))
            for i in iters
        ]
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs
    return schedule


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block implementation.

    This implementation follows the original SE-Net paper, applying channel-wise attention
    to feature maps. It uses global average pooling followed by a bottleneck structure
    to compute channel-wise attention weights.

    credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4

    Args:
        channels (int): Number of input channels
        reduction (int, optional): Reduction ratio for the bottleneck. Defaults to 16.
        activation (str, optional): Type of activation to use. Not used in base implementation.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.reduction = reduction
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // self.reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SE attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Attended feature map of same shape as input
        """
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class SEBlockV2(nn.Module):
    """Enhanced Squeeze-and-Excitation block with spatial attention.

    This version enhances the original SE block by adding spatial attention
    through an additional convolutional pathway.

    The is a custom implementation of the ideas presented in the paper:
    https://www.sciencedirect.com/science/article/abs/pii/S0031320321003460

    Args:
        channels (int): Number of input channels
        reduction (int, optional): Channel reduction ratio. Defaults to 16.
        activation (str, optional): Type of activation function to use.
    """

    def __init__(self, channels, reduction=16, activation="relu"):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.activation = get_activation(activation)

        self.fc_spatial = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Conv2d(channels, channels, kernel_size=2, stride=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.fc_reduction = nn.Linear(
            in_features=channels * (4 * 4), out_features=channels // self.reduction
        )
        self.fc_extention = nn.Linear(in_features=channels // self.reduction, out_features=channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply enhanced SE attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Attended feature map of same shape as input
        """
        identity = x
        x = self.fc_spatial(identity)
        x = self.activation(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_reduction(x)
        x = self.activation(x)
        x = self.fc_extention(x)
        x = self.sigmoid(x)
        x = x.reshape(x.size(0), x.size(1), 1, 1)
        return x


class SEBlockV3(nn.Module):
    """Advanced Squeeze-and-Excitation block with dual attention.

    This version implements both channel and spatial attention mechanisms
    with configurable reduction ratios and normalization options.

    Args:
        channels (int): Number of input channels
        reduction_c (int, optional): Channel reduction ratio. Defaults to 2.
        reduction_s (int, optional): Spatial reduction ratio. Defaults to 8.
        activation (str, optional): Type of activation function.
        norm (str, optional): Type of normalization to use.
        first_layer (bool, optional): Whether this is the first layer. Defaults to False.
    """

    def __init__(
        self,
        *,
        channels: int,
        reduction_c: int = 2,
        reduction_s: int = 8,
        activation: str = "relu",
        norm: str = "batch",
        first_layer: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.first_layer = first_layer
        self.reduction_c = reduction_c if not first_layer else 1
        self.reduction_s = reduction_s
        self.activation = get_activation(activation)

        self.fc_pool = nn.AdaptiveAvgPool2d(reduction_s)
        self.fc_conv = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=2,
            stride=2,
            groups=self.channels,
            bias=False,
        )
        self.fc_norm = get_normalization(norm, self.channels)

        self.linear1 = nn.Linear(
            in_features=self.channels * (reduction_s // 2 * reduction_s // 2),
            out_features=self.channels // self.reduction_c,
        )
        self.linear2 = nn.Linear(
            in_features=self.channels // self.reduction_c, out_features=self.channels
        )

        self.activation_output = nn.Softmax(dim=1) if first_layer else nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dual attention mechanism to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Attended feature map of same shape as input
        """
        identity = x
        x = self.fc_pool(x)
        x = self.fc_conv(x)
        x = self.fc_norm(x)
        x = self.activation(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        if self.first_layer:
            x = self.activation_output(x) * x.size(1)
        else:
            x = self.activation_output(x)

        x = identity * x.reshape(x.size(0), x.size(1), 1, 1)
        return x


# pylint: disable=too-many-return-statements, too-many-branches
def get_activation(activation_name: str) -> nn.Module:
    """Get activation function by name or return the provided activation function.

    Args:
        activation_name (Union[str, nn.Module]): Name of the activation function or the function itself

    Returns:
        nn.Module: The requested activation function

    Raises:
        ValueError: If activation_name is not recognized
    """
    if activation_name == "relu":
        return nn.ReLU6(inplace=False)
    if isinstance(activation_name, torch.nn.modules.activation.ReLU6):
        return activation_name
    if activation_name == "gelu":
        return nn.GELU()
    if isinstance(activation_name, torch.nn.modules.activation.GELU):
        return activation_name
    if activation_name == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    if isinstance(activation_name, torch.nn.modules.activation.LeakyReLU):
        return activation_name
    if activation_name == "prelu":
        return nn.PReLU()
    if isinstance(activation_name, torch.nn.modules.activation.PReLU):
        return activation_name
    if activation_name == "selu":
        return nn.SELU(inplace=True)
    if isinstance(activation_name, torch.nn.modules.activation.SELU):
        return activation_name
    if activation_name == "sigmoid":
        return nn.Sigmoid()
    if isinstance(activation_name, torch.nn.modules.activation.Sigmoid):
        return activation_name
    if activation_name == "tanh":
        return nn.Tanh()
    if isinstance(activation_name, torch.nn.modules.activation.Tanh):
        return activation_name
    if activation_name == "mish":
        return nn.Mish()
    if isinstance(activation_name, torch.nn.modules.activation.Mish):
        return activation_name

    raise ValueError(
        f"activation must be one of leaky_relu, prelu, selu, gelu, sigmoid, tanh, relu. Got: {activation_name}"
    )


# pylint: disable=too-many-return-statements, too-many-branches
def get_normalization(
    normalization_name: str, num_channels: int, num_groups: int = 32, dims: int = 2
) -> nn.Module:
    """Get normalization layer by name.

    Args:
        normalization_name (str): Name of the normalization layer
        num_channels (int): Number of channels to normalize
        num_groups (int, optional): Number of groups for GroupNorm. Defaults to 32.
        dims (int, optional): Number of dimensions (1D, 2D, 3D). Defaults to 2.

    Returns:
        nn.Module: The requested normalization layer

    Raises:
        ValueError: If normalization_name is not recognized or dims is invalid
    """
    if normalization_name == "batch":
        if dims == 1:
            return nn.BatchNorm1d(num_channels)
        if dims == 2:
            return nn.BatchNorm2d(num_channels)
        if dims == 3:
            return nn.BatchNorm3d(num_channels)
    if normalization_name == "instance":
        if dims == 1:
            return nn.InstanceNorm1d(num_channels)
        if dims == 2:
            return nn.InstanceNorm2d(num_channels)
        if dims == 3:
            return nn.InstanceNorm3d(num_channels)
    if normalization_name == "layer":
        return nn.LayerNorm(num_channels)
    if normalization_name == "group":
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    if normalization_name == "bcn":
        if dims == 1:
            return nn.Sequential(nn.BatchNorm1d(num_channels), nn.GroupNorm(1, num_channels))
        if dims == 2:
            return nn.Sequential(nn.BatchNorm2d(num_channels), nn.GroupNorm(1, num_channels))
        if dims == 3:
            return nn.Sequential(nn.BatchNorm3d(num_channels), nn.GroupNorm(1, num_channels))
    if normalization_name == "none":
        return nn.Identity()

    raise ValueError(
        f"normalization must be one of batch, instance, layer, group, none. Got: {normalization_name}"
    )


def convert_torch_to_float(tensor: torch.Tensor) -> float:
    """Convert a PyTorch tensor, numpy array, or numeric value to a Python float.

    Args:
        tensor (Union[torch.Tensor, np.ndarray, float, int]): Value to convert

    Returns:
        float: The converted value

    Raises:
        ValueError: If the input cannot be converted to float
    """
    if torch.is_tensor(tensor):
        return float(tensor.detach().cpu().numpy().astype(np.float32))
    if isinstance(tensor, np.ndarray) and tensor.size == 1:
        return float(tensor.astype(np.float32))
    if isinstance(tensor, float):
        return tensor
    if isinstance(tensor, int):
        return float(tensor)

    raise ValueError("Cannot convert tensor to float")


class AttrDict(dict):
    """A dictionary subclass that allows attribute-style access to dictionary items.

    This class allows you to access dictionary items using dot notation:
    d = AttrDict({'foo': 'bar'})
    d.foo  # returns 'bar'
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def read_yaml(path: str) -> AttrDict:
    """Read a YAML file and return its contents as an AttrDict.

    Args:
        path (str): Path to the YAML file

    Returns:
        AttrDict: Dictionary with the YAML contents, supporting attribute access
    """
    with open(path, encoding="utf-8") as f:
        params = yaml.load(f, Loader=yaml.Loader)
    return AttrDict(params)
