import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNAggregator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_layers=3):
        """
        A simple Temporal Convolutional Network that preserves the temporal resolution.
        
        Args:
            in_channels (int): Input feature dimension (D).
            out_channels (int): Output feature dimension.
            kernel_size (int): Kernel size of the temporal convolutions.
            num_layers (int): Number of convolutional layers.
        """
        super(TCNAggregator, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2  # padding to preserve length
                )
            )
            layers.append(nn.ReLU())
            in_channels = out_channels
        self.conv_net = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Shape (B, T, D)
        Returns:
            Tensor: Shape (B, T, out_channels)
        """
        # Transpose to (B, D, T) for 1D convolutions
        x = x.transpose(1, 2)
        out = self.conv_net(x)
        # Transpose back to (B, T, out_channels)
        out = out.transpose(1, 2)
        return out
