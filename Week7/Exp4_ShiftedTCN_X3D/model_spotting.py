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
from model.tcn_prediction import TCNPrediction
from model.x3d import X3DFeatureExtractor
from pytorchvideo.models.hub import x3d_s



#Local imports
from model.modules import BaseRGBModel, FCLayers, step

class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args = None):
            super().__init__()
            self._feature_arch = args.feature_arch
            self._multi_stage_tcn= args.multi_stage_tcn

            if self._feature_arch == "x3d_s":
                x3d_model = x3d_s(pretrained=True)
                print(x3d_model)
                features = nn.Sequential(*x3d_model.blocks[:4])
                self._d = 96

            else:
                raise NotImplementedError(args.feature_arch)

            self._features = features

            if self._multi_stage_tcn:
                self._tcn = TCNPrediction(feat_dim=self._d, num_classes=args.num_classes+1,
                          num_stages=args.tcn_stages, num_layers=args.tcn_layers)
            else:
                self._tcn = nn.Sequential(
                    nn.Conv1d(self._d, self._d, kernel_size=3, padding=1),  # preserves T
                    nn.ReLU(),
                    nn.Conv1d(self._d, self._d, kernel_size=3, padding=1),
                    nn.ReLU()
                ) 


            #Augmentations and crop
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
                T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
                # T.RandomApply([T.GaussianBlur(5)], p = 0.25),
                T.RandomHorizontalFlip(),
            ])

            self._fc = FCLayers(self._d, args.num_classes+1)

            #Standarization
            self.standarization = T.Compose([
                T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) #Imagenet mean and std
            ])

        def forward(self, x):
            x = self.normalize(x) #Normalize to 0-1
            batch_size, clip_len, channels, height, width = x.shape #B, T, C, H, W

            if self.training:
                x = self.augment(x) #augmentation per-batch

            x = self.standarize(x) #standarization imagenet stats
                        
            x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] → [B, C, T, H, W]
            x = self._features(x)  # returns [B, T, D]

            x = F.adaptive_avg_pool3d(x, (x.size(2), 1, 1))  # (B, D, T, 1, 1)
            x = x.squeeze(-1).squeeze(-1)  # (B, D, T)
            x = x.permute(0, 2, 1)  # (B, T, D)

            # TCN
            if self._multi_stage_tcn:
                x = self._tcn(x)  # B, T, num_classes+1
                if isinstance(x, torch.Tensor):
                    return x
                else:
                    return x[-1]  # Si devuelve lista de stages, nos quedamos con el último
            else:
                x = x.permute(0, 2, 1)  # (B, T, D) → (B, D, T)
                x = self._tcn(x)
                x = x.permute(0, 2, 1)  # (B, T, D) → (B, D, T)

            im_feat = self._fc(x)

            return im_feat
        
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
                    # print(f"Pred shape: {pred.shape}")
                    # print(f"Label shape: {label.shape}")

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
