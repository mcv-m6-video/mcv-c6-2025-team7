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
        x: Tensor of shape (B, T, D)
        Returns:
            vlad: Tensor of shape (B, T, num_clusters * D)
        """
        B, T, D = x.shape
        # Permute to (B, D, T) for the 1D conv
        x_perm = x.permute(0, 2, 1)  # (B, D, T)
        
        # Compute soft assignment scores using a 1D convolution.
        assignment = self.conv(x_perm)  # (B, num_clusters, T)
        assignment = F.softmax(assignment, dim=1)  # softmax over clusters
        
        # Compute residuals: expand x and centers.
        x_expand = x.unsqueeze(1)  # (B, 1, T, D)
        c_expand = self.centers.unsqueeze(0).unsqueeze(2)  # (1, num_clusters, 1, D)
        residuals = x_expand - c_expand  # (B, num_clusters, T, D)
        
        # Weight the residuals by the assignment scores.
        weighted_res = residuals * assignment.unsqueeze(-1)  # (B, num_clusters, T, D)
        
        # Instead of summing over T, keep the T dimension:
        # vlad now has shape (B, num_clusters, T, D)
        vlad = weighted_res
        
        # Permute to (B, T, num_clusters, D)
        vlad = vlad.permute(0, 2, 1, 3)
        
        # Flatten the cluster and feature dimensions: (B, T, num_clusters*D)
        vlad = vlad.reshape(B, T, -1)
        
        # Normalize each frame's descriptor
        vlad = F.normalize(vlad, p=2, dim=-1)
        
        return vlad
