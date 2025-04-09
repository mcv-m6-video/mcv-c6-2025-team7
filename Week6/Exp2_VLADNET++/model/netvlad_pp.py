# --- File: netvlad_pp.py (new file) ---

import torch
import torch.nn as nn
import torch.nn.functional as F

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
