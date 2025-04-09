"""
File containing the main model.
"""

#Standard imports
import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torch.nn import TransformerEncoder, TransformerEncoderLayer



#Local imports
from model.modules import BaseRGBModel, FCLayers, step


class Model(BaseRGBModel):
    class Impl(nn.Module):
        def __init__(self, args=None):
            super().__init__()
            self._feature_arch = args.feature_arch

            if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
                features = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny004': 'regnety_004',
                    'rny008': 'regnety_008',
                }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
                feat_dim = features.head.fc.in_features
                features.head.fc = nn.Identity()

            elif self._feature_arch.startswith(('convnext_tiny')):
                features = timm.create_model('convnext_tiny', pretrained=True)
                feat_dim = features.head.fc.in_features
                features.head.fc = nn.Identity()

            self._features = features
            self._d = feat_dim

            # Temporal Windowing
            self.window_size = args.window_size

            # Positional Embedding
            self.pos_embedding = nn.Parameter(torch.randn(1, args.clip_len, self._d))

            # Dilated Temporal Convolution (Advanced)
            self.dilated_tcn_1 = nn.Conv1d(self._d, self._d, kernel_size=3, dilation=2, padding=2)
            self.dilated_tcn_2 = nn.Conv1d(self._d, self._d, kernel_size=3, dilation=4, padding=4)

            # Multi-Scale Temporal Convolutions
            self.multi_scale_tcn_1 = nn.Conv1d(self._d, self._d, kernel_size=3, padding=1)
            self.multi_scale_tcn_2 = nn.Conv1d(self._d, self._d, kernel_size=5, padding=2)
            self.multi_scale_tcn_3 = nn.Conv1d(self._d, self._d, kernel_size=7, padding=3)

            self.fc_transform = nn.Linear(250, self._d)  # Adjusting the input size for the linear layer

            # Self-Attention Layer
            self.self_attention = nn.MultiheadAttention(self._d, num_heads=8, dropout=0.2)

            # Learned Attention Mechanism
            self.learned_attention = nn.Sequential(
                nn.Linear(self._d, 1),
                nn.Softmax(dim=1)
            )

            # Dual-Stage Temporal Fusion
            self.lstm = nn.LSTM(self._d, self._d, batch_first=True, bidirectional=True)
            self.fusion = nn.Sequential(
                nn.Linear(self._d * 2, self._d),  # For bidirectional LSTM output
                nn.ReLU(),
                nn.Dropout(0.5)
            )

            # Residual Attention Mechanism with Self-Attention
            self.residual_self_attention = nn.MultiheadAttention(self._d, num_heads=8, dropout=0.2)

            # Classification Head
            self._fc = FCLayers(self._d, args.num_classes + 1)

            # Augmentation and Standardization
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
                T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.GaussianBlur(5)], p=0.25),
                T.RandomHorizontalFlip(),
            ])

            self.standarization = T.Compose([ 
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # Imagenet mean and std
            ])

        def forward(self, x):
            B, T, C, H, W = x.shape
            x = self.normalize(x)

            if self.training:
                x = self.augment(x)
            x = self.standarize(x)

            x = x.view(-1, C, H, W)  # (B*T, C, H, W)

            with torch.no_grad():
                features = self._features(x)  # Extract features

            features = features.view(B, T, -1)  # (B, T, D)

            # Positional Embedding
            features = features + self.pos_embedding[:, :T, :]
            

            # Apply Temporal Convolutions (Dilated + Multi-Scale)
            dilated_1 = F.relu(self.dilated_tcn_1(features.permute(0, 2, 1)))  # (B, D, window_size)
            dilated_2 = F.relu(self.dilated_tcn_2(features.permute(0, 2, 1)))  # (B, D, window_size)
            multi_scale_1 = F.relu(self.multi_scale_tcn_1(features.permute(0, 2, 1)))  # (B, D, window_size)
            multi_scale_2 = F.relu(self.multi_scale_tcn_2(features.permute(0, 2, 1)))  # (B, D, window_size)
            multi_scale_3 = F.relu(self.multi_scale_tcn_3(features.permute(0, 2, 1)))  # (B, D, window_size)

            # Combine features from different scales
            combined_features = torch.cat([dilated_1, dilated_2, multi_scale_1, multi_scale_2, multi_scale_3], dim=-1)

            combined_features = self.fc_transform(combined_features)

            # Self-Attention
            attn_output, _ = self.self_attention(combined_features, combined_features, combined_features)  # (B, window_size, D)

            # Learned Attention Mechanism
            attention_weights = self.learned_attention(attn_output)  # (B, window_size, 1)
            attn_output = attn_output * attention_weights  # (B, window_size, D)

            # LSTM-based Fusion
            lstm_output, _ = self.lstm(attn_output)  # (B, window_size, D)
            fused_features = self.fusion(lstm_output)  # (B, window_size, D)

            # Residual Attention Mechanism with Self-Attention
            residual_attn_output, _ = self.residual_self_attention(fused_features, fused_features, fused_features)  # (B, T, D)
            residual_attention_weights = self.learned_attention(residual_attn_output)  # (B, T, 1)
            pooled_features = torch.sum(residual_attn_output * residual_attention_weights, dim=1, keepdim=True)  # (B, 1, D)
            features = features + pooled_features  # residual boost

            # Classification
            output = self._fc(features)  # (B, T, num_classes + 1)
            return output


        
        def normalize(self, x):
            return x / 255.
        
        def augment(self, x):
            for i in range(x.shape[0]):
                x[i] = self.augmentation(x[i])
            return x

        def standarize(self, x):
            for i in range(x.shape[0]):
                x[i] = self.standarization(x[i])
            return x

        def print_stats(self):
            print('Model params:',
                sum(p.numel() for p in self.parameters()))

    def __init__(self, args=None):
        self.device = "cpu"
        if torch.cuda.is_available() and ("device" in args) and (args.device == "cuda"):
            self.device = "cuda"

        self._model = Model.Impl(args=args)
        self._model.print_stats()
        self._args = args

        self._model.to(self.device)
        self._num_classes = args.num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        weights = torch.tensor([1.0] + [5.0] * (self._num_classes), dtype=torch.float32).to(self.device)

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label']
                label = label.to(self.device).long()

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    pred = pred.view(-1, self._num_classes + 1) # B*T, num_classes
                    label = label.view(-1) # B*T
                    loss = F.cross_entropy(
                            pred, label, reduction='mean', weight = weights)

                if optimizer is not None:
                    step(optimizer, scaler, loss,
                        lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)     # Avg loss

    def predict(self, seq):

        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4: # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = self._model(seq)

            # apply sigmoid
            pred = torch.softmax(pred, dim=-1)
            
            return pred.cpu().numpy()
