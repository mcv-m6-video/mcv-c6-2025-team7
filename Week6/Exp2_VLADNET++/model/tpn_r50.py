# File: tpn_r50.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SimpleTPN(nn.Module):
    """
    A simple TPN neck that fuses features from multiple backbone levels.
    It applies lateral 1x1 convolutions to each input feature map,
    upsamples them to the same spatial size, concatenates, and fuses via a 3x3 convolution.
    """
    def __init__(self, in_channels_list, out_channels):
        super(SimpleTPN, self).__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])
        self.fusion_conv = nn.Conv2d(out_channels * len(in_channels_list),
                                     out_channels,
                                     kernel_size=3,
                                     padding=1)
    
    def forward(self, features):
        # features: a list of feature maps, each of shape (B, C_i, H_i, W_i)
        lateral_feats = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        target_size = lateral_feats[0].shape[-2:]
        upsampled = [F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
                     for feat in lateral_feats]
        fused = torch.cat(upsampled, dim=1)
        fused = self.fusion_conv(fused)
        return fused

class TPN_R50(nn.Module):
    def __init__(self, num_classes, clip_len=50, pretrained=True, return_frame_features=False):
        """
        A simplified TPN-r50 module.
        
        Args:
            num_classes (int): Number of classes (if using the classification head).
            clip_len (int): Number of frames per clip.
            pretrained (bool): Whether to load pretrained weights for ResNet-50.
            return_frame_features (bool): If True, return per-frame features rather than aggregating temporally.
        """
        super(TPN_R50, self).__init__()
        self.clip_len = clip_len
        self.return_frame_features = return_frame_features
        # Backbone: using timm's resnet50 with features_only=True.
        # This returns a list of intermediate feature maps.
        self.backbone = timm.create_model('resnet50', pretrained=pretrained, features_only=True)
        # For TPN, we'll use the last two levels.
        in_channels_list = self.backbone.feature_info.channels()[-2:]
        # Save the feature dimension of the final stage for reference.
        self.out_dim = in_channels_list[-1]  # e.g., 2048 normally.
        # Define the TPN neck (we set output channels to, say, 1024).
        neck_out_channels = 1024
        self.neck = SimpleTPN(in_channels_list, neck_out_channels)
        # If using a classification head (for clip-level classification):
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(neck_out_channels, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, T, C, H, W)
        Returns:
            If return_frame_features is False: logits of shape (B, num_classes)
            Otherwise: per-frame features of shape (B, T, D)
        """
        B, T, C, H, W = x.shape
        # Process each frame independently.
        x = x.view(B * T, C, H, W)
        # Get multi-level features from the backbone.
        features = self.backbone(x)
        # Select the last two levels.
        features = features[-2:]
        # For each feature map, reshape to (B, T, C, H_i, W_i) and apply temporal pooling if desired.
        aggregated_feats = []
        for feat in features:
            feat = feat.view(B, T, feat.size(1), feat.size(2), feat.size(3))
            # In this simplified example, we perform a temporal average pooling to get one feature per clip.
            # If you want per-frame features, you can skip the pooling.
            if not self.return_frame_features:
                feat = feat.mean(dim=1)  # (B, C, H, W)
            aggregated_feats.append(feat)
        
        # If return_frame_features is True, we need to process the sequence frame-by-frame.
        # Otherwise, aggregated_feats now contain two tensors of shape (B, C, H, W)
        # which we fuse with the TPN neck.
        if self.return_frame_features:
            # Here, you might want to apply the neck in a frame-wise manner.
            # One strategy is to iterate over the time dimension.
            frame_features = []
            for t in range(T):
                # For each level, extract the t-th frame: resulting in (B, C, H, W)
                feats_t = [feat[:, t, ...] for feat in aggregated_feats]
                # Fuse them using the neck:
                fused_t = self.neck(feats_t)  # (B, neck_out_channels, H_fused, W_fused)
                # Global average pool:
                pooled = F.adaptive_avg_pool2d(fused_t, (1, 1)).view(B, -1)
                frame_features.append(pooled)
            # Stack along time dimension: (B, T, neck_out_channels)
            out_features = torch.stack(frame_features, dim=1)
            return out_features
        else:
            # Use the TPN neck to fuse the two levels.
            fused_feats = self.neck(aggregated_feats)  # (B, neck_out_channels, H_fused, W_fused)
            # Classification head:
            logits = self.cls_head(fused_feats)  # (B, num_classes)
            return logits

def load_tpn_r50(args):
    """
    Helper function to load TPN_R50.
    
    Expects args to have:
      - num_classes: int, number of classes.
      - clip_len: int, number of frames per clip.
      - Optionally, return_frame_features (bool).
    
    Returns:
        model (TPN_R50): An instance of TPN_R50.
    """
    return TPN_R50(num_classes=args.num_classes,
                   clip_len=args.clip_len,
                   pretrained=True,
                   return_frame_features=getattr(args, "return_frame_features", False))


# Example usage:
if __name__ == '__main__':
    class Args:
        num_classes = 400
        clip_len = 50
        num_vlad_clusters = 8
        use_tpn = True  # Your flag to use TPN
    
    args = Args()
    model = load_tpn_r50(args)
    # Create a dummy input with shape (B, T, C, H, W)
    dummy_input = torch.randn(2, args.clip_len, 3, 224, 398)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected: (2, 400)
