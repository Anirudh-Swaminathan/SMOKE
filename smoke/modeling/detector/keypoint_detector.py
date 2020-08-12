import torch
from torch import nn

import logging

from smoke.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..heads.heads import build_heads


class KeypointDetector(nn.Module):
    '''
    Generalized structure for keypoint based object detector.
    main parts:
    - backbone
    - heads
    '''

    def __init__(self, cfg):
        super(KeypointDetector, self).__init__()

        self.logger = logging.getLogger(__name__)

        self.logger.info("KeypointDetector __init__(). Calling build_backbone()")
        self.backbone = build_backbone(cfg)
        self.logger.info("Back in KeypointDetector __init__(). Calling build_heads()")
        self.heads = build_heads(cfg, self.backbone.out_channels)
        self.logger.info("End of KeypointDetector constructor")

    def forward(self, images, targets=None):
        """
        Args:
            images:
            targets:

        Returns:

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        result, detector_losses = self.heads(features, targets)

        if self.training:
            losses = {}
            losses.update(detector_losses)

            return losses

        return result