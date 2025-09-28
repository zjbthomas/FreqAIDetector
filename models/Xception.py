import torch
from torch import nn

import timm

class Xception(nn.Module):
    def __init__(self):
        super(Xception, self).__init__()

        self.model = timm.create_model('xception', num_classes = 1)

    def forward(self, x):
        return self.model(x)