import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNAggregator(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 num_layers: int = 3,
                 downsample: int = 1,
                 stride: int = 2,
                 upsample: bool = False):
        """
        A Temporal Convolutional Network that optionally reduces
        temporal resolution by 'downsample' and then (optionally)
        upsamples back to the original length.

        Args:
            in_channels (int): Input feature dimension (D).
            out_channels (int): Output feature dimension.
            kernel_size (int): Kernel size of the temporal convolutions.
            num_layers (int): Number of convolutional layers.
            downsample (int): Factor to reduce temporal length by.
                E.g. 2 → halve the number of time‐steps.
            upsample (bool): Whether to interpolate output
                back to the original length.
        """
        super().__init__()
        self.downsample = downsample
        self.upsample = upsample

        # 1D avg‐pool for temporal downsampling (no change if downsample=1)
        if downsample > 1:
            self.pool = nn.AvgPool1d(
                kernel_size=downsample,
                stride=stride,
                padding=0,
                ceil_mode=False
            )
        else:
            self.pool = None

        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,  # preserve length of the pooled signal
                    bias=True
                )
            )
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.conv_net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): Shape (B, T, D)
        Returns:
            Tensor: (B, T', out_channels), where
                    T' = ceil(T/downsample) if upsample=False,
                    else T' = T again.
        """
        # (B, T, D) → (B, D, T)
        x = x.transpose(1, 2)
        # remember the *true* input length
        T_orig = x.size(-1)

        # downsample in time
        if self.pool is not None:
            x = self.pool(x)

        # conv at reduced frame‐rate
        x = self.conv_net(x)

        # optionally interpolate back to the original T
        if self.upsample and self.downsample > 1:
            # linear interpolation along the time‐axis
            x = F.interpolate(
                x,
                size=T_orig,
                mode='linear',
                align_corners=False
            )

        # back to (B, T', D_out)
        x = x.transpose(1, 2)
        return x
