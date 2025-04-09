"""
File containing the different modules related to the model: T-DEED.
"""

#Standard imports
import abc
import torch
import torch.nn as nn
import math

#Local imports

class ABCModel:

    @abc.abstractmethod
    def get_optimizer(self, opt_args):
        raise NotImplementedError()

    @abc.abstractmethod
    def epoch(self, loader, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, seq):
        raise NotImplementedError()

    @abc.abstractmethod
    def state_dict(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, state_dict):
        raise NotImplementedError()

class BaseRGBModel(ABCModel):

    # def get_optimizer(self, opt_args):
    #     return torch.optim.AdamW(self._get_params(), **opt_args), \
    #         torch.cuda.amp.GradScaler() if self.device == 'cuda' else None
    def get_optimizer(self, opt_args):
        """
        Creates an AdamW optimizer with layerwise learning rates.
        
        opt_args: dictionary of optimizer arguments. It must include 'lr' (base learning rate)
                  and optionally 'backbone_scale' (a multiplier for the backbone's learning rate).
        
        Returns:
            optimizer: the AdamW optimizer with two parameter groups.
            scaler: GradScaler for mixed precision (if using cuda).
        """
        # Extract base learning rate and backbone scale, with defaults.
        base_lr = opt_args.pop("lr", 1e-4)
        backbone_scale = opt_args.pop("backbone_scale", 0.1)
        
        # Group parameters: Assume parameters with "_features" in their name belong to the backbone.
        backbone_params = []
        new_params = []
        for name, param in self._model.named_parameters():
            if "_features" in name:
                backbone_params.append(param)
            else:
                new_params.append(param)
        
        # Create two parameter groups with different learning rates.
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': base_lr * backbone_scale},
            {'params': new_params, 'lr': base_lr}
        ], **opt_args)
        
        scaler = torch.cuda.amp.GradScaler() if self.device == 'cuda' else None
        return optimizer, scaler

    """ Assume there is a self._model """

    def _get_params(self):
        return list(self._model.parameters())

    def state_dict(self):
        if isinstance(self._model, nn.DataParallel):
            return self._model.module.state_dict()
        return self._model.state_dict()

    def load(self, state_dict):
        if isinstance(self._model, nn.DataParallel):
            self._model.module.load_state_dict(state_dict)
        else:
            self._model.load_state_dict(state_dict)    

class FCLayers(nn.Module):

    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self._fc_out = nn.Linear(feat_dim, num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x):
        if len(x.shape) == 3:
            b, t, d = x.shape
            x = x.reshape(b * t, d)
            return self._fc_out(self.dropout(x)).reshape(b, t, -1)
        elif len(x.shape) == 2:
            return self._fc_out(self.dropout(x))

def step(optimizer, scaler, loss, lr_scheduler=None):
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()

    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    if lr_scheduler is not None:
        lr_scheduler.step()
    optimizer.zero_grad()