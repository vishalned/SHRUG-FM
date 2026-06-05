"""Neural network decoder components for the extreme environments model.

This module provides decoder components used in the neural network architecture
for extreme environment modeling. It includes CNN blocks and decoder structures
for processing and upsampling features.
"""

from torch import nn

from signal_task.models.model_components.training_utils import (
    SEBlock,
    get_activation,
    get_normalization,
)


class CoreCNNBlock(nn.Module):
    """Core CNN block with residual connections and squeeze-excitation.

    This block implements a CNN structure with residual connections and squeeze-excitation
    mechanism for better feature refinement.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        norm (str, optional): Normalization type. Defaults to "batch"
        activation (str, optional): Activation function type. Defaults to "relu"
        padding (str, optional): Padding type. Defaults to "same"
        residual (bool, optional): Whether to use residual connection. Defaults to True
        dropout (float, optional): Probability of dropout. Defaults to 0.0, no dropout
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        norm="batch",
        activation="relu",
        padding="same",
        residual=True,
        dropout=0.0,
    ):
        super().__init__()
        self.activation = get_activation(activation)
        self.residual = residual
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.squeeze = SEBlock(self.out_channels)
        self.match_channels = nn.Identity()
        if in_channels != out_channels:
            self.match_channels = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
                get_normalization(norm, out_channels),
            )
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, padding=0)
        self.norm1 = get_normalization(norm, self.out_channels)
        self.conv2 = nn.Conv2d(
            self.out_channels,
            self.out_channels,
            3,
            padding=self.padding,
            groups=self.out_channels,
        )
        self.norm2 = get_normalization(norm, self.out_channels)

        self.conv3 = nn.Conv2d(
            self.out_channels, self.out_channels, 3, padding=self.padding, groups=1
        )
        self.norm3 = get_normalization(norm, self.out_channels)

        assert 0 <= dropout <= 1
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """Forward pass of the CoreCNNBlock.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Processed feature tensor
        """
        identity = x

        x = self.activation(self.norm1(self.conv1(x)))
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = self.activation(self.norm2(self.conv2(x)))
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = self.norm3(self.conv3(x))
        x = x * self.squeeze(x)
        if self.residual:
            x = x + self.match_channels(identity)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block for upsampling and processing features.

    This block implements upsampling followed by multiple CNN blocks for
    feature processing.

    Args:
        depth (int): Number of CNN blocks to use
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        norm (str, optional): Normalization type. Defaults to "batch"
        activation (str, optional): Activation function type. Defaults to "relu"
        padding (str, optional): Padding type. Defaults to "same"
        dropout (float, optional): Probability of dropout. Defaults to 0.0, no dropout
    """

    def __init__(
        self,
        depth,
        in_channels,
        out_channels,
        *,
        norm="batch",
        activation="relu",
        padding="same",
        dropout=0.0,
    ):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_blocks = activation
        self.activation = get_activation(activation)
        self.norm = norm
        self.padding = padding
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.match_channels = CoreCNNBlock(
            self.in_channels,
            self.out_channels,
            norm=self.norm,
            activation=self.activation_blocks,
            padding=self.padding,
            dropout=dropout,
        )

        self.blocks = []
        for _ in range(self.depth):
            block = CoreCNNBlock(
                self.out_channels,
                self.out_channels,
                norm=self.norm,
                activation=self.activation_blocks,
                padding=self.padding,
                dropout=dropout,
            )
            self.blocks.append(block)
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        """Forward pass of the DecoderBlock.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Upsampled and processed feature tensor
        """
        x = self.upsample(x)
        x = self.match_channels(x)
        for i in range(self.depth):
            x = self.blocks[i](x)
        return x


class CoreDecoder(nn.Module):
    """Core decoder architecture for feature upsampling and processing.

    This module implements the main decoder architecture that processes embedded
    features through multiple decoder blocks to generate the final output.

    Args:
        embedding_dim (int, optional): Dimension of input embeddings. Defaults to 10
        output_dim (int, optional): Dimension of output features. Defaults to 1
        depths (List[int], optional): Depths for each decoder block. Defaults to [3,3,9,3]
        dims (List[int], optional): Channel dimensions for each level. Defaults to [96,192,384,768]
        activation (str, optional): Activation function type. Defaults to "relu"
        norm (str, optional): Normalization type. Defaults to "batch"
        padding (str, optional): Padding type. Defaults to "same"
        dropout (float, optional): Probability of dropout. Defaults to 0.0, no dropout
    """

    def __init__(
        self,
        *,
        embedding_dim=10,
        output_dim=1,
        depths=None,
        dims=None,
        activation="relu",
        norm="batch",
        padding="same",
        dropout=0.0,
    ):
        super().__init__()
        self.depths = [3, 3, 9, 3] if depths is None else depths
        self.dims = [96, 192, 384, 768] if dims is None else dims
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.activation = activation
        self.norm = norm
        self.padding = padding
        self.decoder_blocks = []
        assert len(self.depths) == len(self.dims), "depths and dims must have the same length."
        for i in reversed(range(len(self.depths))):
            decoder_block = DecoderBlock(
                self.depths[i],
                self.dims[i],
                self.dims[i - 1] if i > 0 else self.dims[0],
                norm=norm,
                activation=activation,
                padding=padding,
                dropout=dropout,
            )
            self.decoder_blocks.append(decoder_block)
        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)
        self.decoder_downsample_block = nn.Identity()
        self.decoder_bridge = nn.Sequential(
            CoreCNNBlock(
                embedding_dim,
                self.dims[-1],
                norm=norm,
                activation=activation,
                padding=padding,
                dropout=dropout,
            ),
        )
        self.decoder_head = nn.Sequential(
            CoreCNNBlock(
                self.dims[0],
                self.dims[0],
                norm=norm,
                activation=activation,
                padding=padding,
                dropout=dropout,
            ),
            nn.Conv2d(self.dims[0], self.output_dim, kernel_size=1, padding=0),
        )

    def forward_decoder(self, x):
        """Process features through decoder blocks.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Processed feature tensor
        """
        for block in self.decoder_blocks:
            x = block(x)
        return x

    def forward(self, x):
        """Forward pass of the CoreDecoder.

        Args:
            x (torch.Tensor): Input tensor containing embeddings

        Returns:
            torch.Tensor: Final output tensor
        """
        x = self.decoder_bridge(x)
        x = self.forward_decoder(x)
        x = self.decoder_head(x)
        return x
