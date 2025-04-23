import torch
import torch.nn as nn
import torch.nn.functional as F

class Shifted1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Shifted1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=0, dilation=dilation)
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x):
        # x: [B, C, T]
        padding = (self.kernel_size - 1) * self.dilation

        # Pad to the left (causal)
        x_padded = F.pad(x, (padding, 0))  # pad only on the left side

        out = self.conv1(x_padded)
        return out

class SingleStageTCN(nn.Module):

    class DilatedResidualLayer(nn.Module):
        def __init__(self, dilation, in_channels, out_channels):
            super(SingleStageTCN.DilatedResidualLayer, self).__init__()
            self.conv_dilated = Shifted1D(in_channels, out_channels, kernel_size=3, dilation=dilation)
            self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
            self.dropout = nn.Dropout()

        def forward(self, x, mask):
            out = F.relu(self.conv_dilated(x))
            out = self.conv_1x1(out)
            out = self.dropout(out)
            return (x + out) * mask[:, 0:1, :]


    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dilate):
        super(SingleStageTCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.layers = nn.ModuleList([
            SingleStageTCN.DilatedResidualLayer(
                2 ** i if dilate else 1, hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(hidden_dim, out_dim, 1)

    def forward(self, x, m=None):
        batch_size, clip_len, _ = x.shape
        if m is None:
            m = torch.ones((batch_size, 1, clip_len), device=x.device)
        else:
            m = m.permute(0, 2, 1)
        x = self.conv_1x1(x.permute(0, 2, 1))
        for layer in self.layers:
            x = layer(x, m)
        x = self.conv_out(x) * m[:, 0:1, :]
        return x.permute(0, 2, 1)

class TCNPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, num_stages=1, num_layers=5):
        super().__init__()

        self._tcn = SingleStageTCN(
            feat_dim, 256, num_classes, num_layers, True)
        self._stages = None
        if num_stages > 1:
            self._stages = nn.ModuleList([SingleStageTCN(
                num_classes, 256, num_classes, num_layers, True)
                for _ in range(num_stages - 1)])

    def forward(self, x):
        x = self._tcn(x)
        if self._stages is None:
            return x
        else:
            outputs = [x]
            for stage in self._stages:
                x = stage(F.softmax(x, dim=2))
                outputs.append(x)
            pooled_output = torch.max(torch.stack(outputs, dim=0), dim=0)[0]
            return pooled_output