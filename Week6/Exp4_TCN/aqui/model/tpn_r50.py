# File: tpn_r50.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SimpleTPN(nn.Module):
    """
    A simplified TPN (Temporal Pyramid Network) neck that fuses multi-level features.
    This version takes two feature maps (e.g., from intermediate stages of ResNet-50)
    and processes them via lateral convolutions before fusing them.
    """
    def __init__(self, in_channels_list, out_channels):
        """
        Args:
            in_channels_list (list[int]): List with the number of channels for each input feature.
            out_channels (int): Number of channels for each lateral output and for fusion.
        """
        super(SimpleTPN, self).__init__()
        # Lateral convolutions: reduce each feature map to out_channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])
        # Fusion convolution: combine the lateral features from different levels.
        self.fusion_conv = nn.Conv2d(out_channels * len(in_channels_list),
                                     out_channels,
                                     kernel_size=3,
                                     padding=1)

    def forward(self, features):
        # features: a list of feature maps, each of shape (B, C_i, H_i, W_i)
        lateral_feats = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        # To fuse, we upsample each lateral feature map to the size of the first one.
        target_size = lateral_feats[0].shape[-2:]
        upsampled = [F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
                     for feat in lateral_feats]
        fused = torch.cat(upsampled, dim=1)
        fused = self.fusion_conv(fused)
        return fused

class TPN_R50(nn.Module):
    """
    A simplified TPN-r50 model.
    
    This model uses:
      - A ResNet-50 backbone from timm (with features_only=True) to extract multi-level frame features.
      - A SimpleTPN neck that fuses features from two selected levels.
      - A classification head that operates on the aggregated (fused) feature map.
      
    The model expects an input tensor of shape (B, T, C, H, W) and returns clip-level logits.
    """
    def __init__(self, num_classes, num_vlad_clusters=8, clip_len=50, pretrained=True):
        """
        Args:
            num_classes (int): Number of target classes.
            num_vlad_clusters (int): (Not used in this simplified version) kept for compatibility.
            clip_len (int): Number of frames per clip.
            pretrained (bool): If True, load a pretrained ResNet-50.
        """
        super(TPN_R50, self).__init__()
        self.clip_len = clip_len
        # Use timm to create a ResNet-50 backbone that returns intermediate features.
        # Set features_only=True to obtain a list of feature maps.
        self.backbone = timm.create_model('resnet50', pretrained=pretrained, features_only=True)
        # For example, use the last two stages' features.
        # The backbone returns a list; we take indices -2 and -1.
        in_channels_list = self.backbone.feature_info.channels()[-2:]
        # Define the TPN neck. Here we set the neck's output channels to 1024 (as in the MMAction2 config).
        neck_out_channels = 512
        self.neck = SimpleTPN(in_channels_list, neck_out_channels)
        # Classification head: perform global average pooling and then classify.
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
            logits (Tensor): Predictions with shape (B, num_classes)
        """
        B, T, C, H, W = x.shape
        # Process each frame independently.
        x = x.view(B * T, C, H, W)
        # Get multi-level features from the backbone.
        # features is a list of feature maps (each of shape (B*T, C_i, H_i, W_i)).
        features = self.backbone(x)
        # Select the last two levels.
        features = features[-2:]
        # Reshape each feature map to (B, T, C_i, H_i, W_i) and average over the temporal dimension.
        aggregated_feats = []
        for feat in features:
            feat = feat.view(B, T, feat.size(1), feat.size(2), feat.size(3))
            # Temporal average pooling.
            feat = feat.mean(dim=1)  # shape becomes (B, C_i, H_i, W_i)
            aggregated_feats.append(feat)
        # Fuse the aggregated features with the TPN neck.
        fused_feats = self.neck(aggregated_feats)  # shape (B, neck_out_channels, H_fused, W_fused)
        # Classification head.
        logits = self.cls_head(fused_feats)  # shape (B, num_classes)
        return logits

def load_tpn_r50(args):
    """
    Helper function to load the TPN-R50 feature extractor/model.
    
    Expects the following attributes in args:
      - num_classes: int, number of classes.
      - num_vlad_clusters: (optional) number of clusters (kept for compatibility; not used here).
      - clip_len: int, number of frames per clip.
      - pretrained: bool, whether to load pretrained weights.
      
    Returns:
        model (TPN_R50): An instance of the TPN_R50 model.
    """
    model = TPN_R50(num_classes=args.num_classes,
                    clip_len=args.clip_len,
                    pretrained=True)
    model.eval()
    return model

# Example usage:
if __name__ == '__main__':
    class Args:
        num_classes = 400
        clip_len = 50
        use_tpn = True  # Your flag to use TPN
    
    args = Args()
    model = load_tpn_r50(args)
    # Create a dummy input with shape (B, T, C, H, W)
    dummy_input = torch.randn(2, args.clip_len, 3, 224, 398)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected: (2, 400)
