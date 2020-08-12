import torch
from torchvision.transforms import functional as F

import logging

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms
        self.logger = logging.getLogger(__name__)
        self.logger.info("IN Compose __init__()")

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor():
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize():
    def __init__(self, mean, std, to_bgr=True):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr
        self.logger = logging.getLogger(__name__)
        self.logger.info("IN Normalize __init__()")

    def __call__(self, image, target):
        if self.to_bgr:
            image = image[[2, 1, 0]]
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
