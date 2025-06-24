import torch
import torch.nn as nn
import torch.nn.init as init

class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.w = nn.Parameter(
            torch.zeros(out_features, in_features, device = device, dtype = dtype)
        )

        std = (2.0 / (out_features + in_features)) ** 0.5
        init.trunc_normal_(self.w, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def use_weight(self, w):
        with torch.no_grad():
            self.w.copy_(w)

    def forward(self, x):
        return torch.matmul(x, self.w.t())
