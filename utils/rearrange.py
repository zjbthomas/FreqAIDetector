import random

from albumentations.core.transforms_interface import ImageOnlyTransform

import torch

class Mean(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(Mean, self).__init__(always_apply, p)

    def apply(self, img, **params):
        return torch.mean(img, dim = 0, keepdim=True)