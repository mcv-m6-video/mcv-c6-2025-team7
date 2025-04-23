# --- File: netvlad_pp.py (new file) ---

import torch
import torch.nn as nn
import torch.nn.functional as F


# Downscaling to 1
class NetVLADPlusPlus(nn.Module):
    def __init__(self, feature_dim, num_clusters):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters
        # Learnable cluster centers
        self.centers = nn.Parameter(torch.rand(num_clusters, feature_dim))
        # Additional parameters for NetVLAD++
        self.conv = nn.Conv1d(feature_dim, num_clusters, kernel_size=1, bias=True)

    def forward(self, x):
        """
        x is assumed to be shape (B, T, D) 
        i.e., batch_size x clip_len x feature_dim
        """
        # (B, D, T) for 1D conv
        x_perm = x.permute(0, 2, 1)
        # Compute soft-assignment scores with conv
        assignment = self.conv(x_perm)  # (B, num_clusters, T)
        assignment = F.softmax(assignment, dim=1)  # cluster assignment along the cluster dimension

        # Now we compute the residuals to cluster centers
        x_expand = x.unsqueeze(1)  # (B, 1, T, D)
        c_expand = self.centers.unsqueeze(0).unsqueeze(2)  # (1, num_clusters, 1, D)

        # broadcast to (B, num_clusters, T, D)
        residuals = x_expand - c_expand
        # Weighted by assignment (B, num_clusters, T, D)
        weighted_res = residuals * assignment.unsqueeze(-1)

        # Summation over time dimension T -> (B, num_clusters, D)
        vlad = weighted_res.sum(dim=2)
        # L2 normalize each cluster descriptor
        vlad = F.normalize(vlad, p=2, dim=-1)
        # Flatten to get final vector: (B, num_clusters * D)
        vlad = vlad.view(x.size(0), -1)
        # In practice, might do an additional normalization step
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad


# Variable downscaling
class NetVLADpp(nn.Module):
    def __init__(
        self,
        num_clusters: int,
        dim: int,
        downsample: int = 1,
        stride: int = 2,
        upsample: bool = False,
        upsample_mode: str = "linear"   # "linear" or "nearest"
    ):
        """
        Args:
          num_clusters: K
          dim: feature‐dimensionality D
          downsample: temporal factor to reduce by (1=no downsample)
          upsample: whether to interpolate/repeat back to original T
          upsample_mode: "linear" for smooth, "nearest" for blocky
        """
        super().__init__()
        assert upsample_mode in ("linear", "nearest"), \
               "upsample_mode must be 'linear' or 'nearest'"

        self.K = num_clusters
        self.D = dim
        self.downsample = downsample
        self.stride = stride
        self.upsample = upsample
        self.upsample_mode = upsample_mode

        self.centers = nn.Parameter(torch.randn(self.K, self.D))
        self.conv = nn.Conv1d(self.D, self.K, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, D)
        returns:
          if upsample=False → (B, T_ds, K*D)
          if upsample=True  → (B, T,    K*D)
        """
        B, T, D = x.shape
        x_perm = x.permute(0, 2, 1)  # (B, D, T)

        # 1) downsample
        if self.downsample > 1:
            x_ds = F.avg_pool1d(x_perm,
                                kernel_size=self.downsample,
                                stride=self.stride)
        else:
            x_ds = x_perm
        T_ds = x_ds.size(-1)

        # 2) soft‐assignment
        assn = self.conv(x_ds)          # (B, K, T_ds)
        assn = F.softmax(assn, dim=1)

        # 3) compute VLAD per segment
        x_seg = x_ds.permute(0, 2, 1)   # (B, T_ds, D)
        x_e   = x_seg.unsqueeze(1)      # (B, 1, T_ds, D)
        c_e   = self.centers.unsqueeze(0).unsqueeze(2)  # (1, K, 1, D)
        resid = x_e - c_e               # (B, K, T_ds, D)
        weighted = resid * assn.unsqueeze(-1)
        v = weighted.permute(0, 2, 1, 3).contiguous()   # (B, T_ds, K, D)
        v = F.normalize(v, p=2, dim=-1)
        v = v.view(B, T_ds, self.K * self.D)             # (B, T_ds, K*D)

        v = F.normalize(v, p=2, dim=-1)  

        # 4) Upsampling back to T frames
        if self.upsample and T_ds != T:
            # bring to (B, C, T_ds)
            v2 = v.permute(0, 2, 1)
            if self.upsample_mode == "nearest":
                # blocky repeat
                v2 = F.interpolate(v2,
                                   size=T,
                                   mode="nearest")
            else:
                # smooth linear ramp
                v2 = F.interpolate(v2,
                                   size=T,
                                   mode="linear",
                                   align_corners=False)
            # back to (B, T, C)
            v = v2.permute(0, 2, 1)

        return v