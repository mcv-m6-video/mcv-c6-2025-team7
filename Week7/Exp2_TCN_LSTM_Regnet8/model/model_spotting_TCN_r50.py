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
from model.tcn_model import TCNAggregator
from model.tpn_r50 import load_tpn


#Local imports
from model.modules import BaseRGBModel, FCLayers, step

class Model(BaseRGBModel, nn.Module):

    class Impl(nn.Module):

        def __init__(self, args = None):
            super().__init__()
            self._feature_arch = args.feature_arch
            self.args = args
            self.model_params = None

            if hasattr(args, "use_tpn") and args.use_tpn:
                self._features = load_tpn(args)
                self._d = self._features.neck.fusion_conv.out_channels
                
            else:

                if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
                    features = timm.create_model({
                        'rny002': 'regnety_002',
                        'rny004': 'regnety_004',
                        'rny008': 'regnety_008',
                    }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)

                    # Assuming your model instance is called 'model'
                    # for name, param in features.named_parameters():
                    #     print(name)

                    feat_dim = features.head.fc.in_features
                    self._features = features
                    # Remove final classification layer
                    features.head.fc = nn.Identity()
                    self._d = feat_dim

                else:
                    raise NotImplementedError(args._feature_arch)

            # Define TCN aggregator to process per-frame features (B, T, D) -> (B, T, D)
            ds = getattr(self.args, "downsample_factor",
                 getattr(self.args, "stride"))
            # set upsampling=True if you still want outputs at the original T
            self.tcn = TCNAggregator(
                in_channels=self._d,
                out_channels=self._d,
                num_layers=3,
                downsample=ds,
                upsample=True   # or True, depending on whether you need T_out=T_in
            )
            
            # Define frame-level classification head.
            # FCLayers should output (B, T, num_classes+1).
            self.frame_fc = FCLayers(self._d, args.num_classes + 1)

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
            x = self.normalize(x) #Normalize to 0-1
            batch_size, clip_len, channels, height, width = x.shape #B, T, C, H, W

            if self.training:
                x = self.augment(x) #augmentation per-batch

            x = self.standarize(x) #standarization imagenet stats

            if hasattr(self.args, "use_tpn") and self.args.use_tpn:
                # When using TPN, pass the 5D tensor directly.
                im_feat = self._features(x)
                # We expect im_feat to be of shape (B, T, D) if TPN_R50 is implemented with
                # the flag return_frame_features=True.
            else:
                        
                # Extract per-frame features: reshape input to (B*T, C, H, W)
                im_feat = self._features(x.view(-1, channels, height, width))
                im_feat = im_feat.reshape(batch_size, clip_len, self._d)  # (B, T, D)

            # Pass features through TCN aggregator (preserving temporal resolution)
            tcn_out = self.tcn(im_feat)  # (B, T, D)

            # Frame-level classification: produce logits for each frame.
            logits = self.frame_fc(tcn_out)  # (B, T, num_classes+1)

            return logits
        
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

        def get_model_parameters(self):
            return self.model_params

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
