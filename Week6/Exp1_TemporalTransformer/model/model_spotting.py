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

        def __init__(self, args = None):
            super().__init__()
            self._feature_arch = args.feature_arch
            self.args = args
            self.model_params = None

            if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
                features = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny004': 'regnety_004',
                    'rny008': 'regnety_008',
                }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
                feat_dim = features.head.fc.in_features

                # Remove final classification layer
                features.head.fc = nn.Identity()
                self._d = feat_dim

            else:
                raise NotImplementedError(args._feature_arch)

            self._features = features
            # print(self._features)


            self.feature_extractor = IntermediateLayerGetter(self._features, return_layers={"s4": "feat"})

            self.pool = nn.AdaptiveAvgPool2d((1, 1))

            # Learnable positional embeddings (B, T, D)
            self.pos_embedding = nn.Parameter(torch.randn(1, args.clip_len, self._d))

            # Transformer Encoder
            encoder_layer = TransformerEncoderLayer(
                d_model=self._d,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True
            )
            self.temporal_transformer = TransformerEncoder(
                encoder_layer, num_layers=2
            )

            # MLP for classification
            self._fc = FCLayers(self._d, args.num_classes+1) # +1 for background class (we now perform per-frame classification with softmax, therefore we have the extra background class)

            #Augmentations and crop
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
                T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.GaussianBlur(5)], p = 0.25),
                T.RandomHorizontalFlip(),
            ])

            #Standarization
            self.standarization = T.Compose([
                T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) #Imagenet mean and std
            ])

        def forward(self, x):
            B, T, C, H, W = x.shape
            x = self.normalize(x) #Normalize to 0-1
            # batch_size, clip_len, channels, height, width = x.shape #B, T, C, H, W

            if self.training:
                x = self.augment(x) #augmentation per-batch

            x = self.standarize(x) #standarization imagenet stats
            x = x.view(-1, C, H, W)  # (B*T, C, H, W)

            with torch.no_grad():
                features = self.feature_extractor(x)["feat"]
            features = self.pool(features).flatten(1)  # (B*T, D)
            features = features.view(B, T, -1)  # (B, T, D)

            # Positional embeddings
            pos_embed = self.pos_embedding[:, :T, :]
            features = features + pos_embed

            # Temporal modeling
            features = self.temporal_transformer(features)  # (B, T, D)

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
            self.model_params = sum(p.numel() for p in self.parameters())
            print('Model params:', self.model_params)

    def __init__(self, args=None):

        # First, initialize nn.Module.
        nn.Module.__init__(self)
        # Then initialize the other parent class.
        BaseRGBModel.__init__(self)

        self.device = "cpu"
        if torch.cuda.is_available() and ("device" in args) and (args.device == "cuda"):
            self.device = "cuda"

        self._model = Model.Impl(args=args)
        self._model.print_stats()
        self._args = args

        self._model.to(self.device)
        self._num_classes = args.num_classes

    # Delegate parameters() to the inner model.
    def parameters(self):
        return self._model.parameters()

    def get_model_parameters(self):
        return self._model.get_model_parameters()

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
    
    def forward(self, x):
    # Delegate forward to the inner model.
        return self._model(x)
    
        # If needed, also delegate children()
    def children(self):
        return self._model.children()