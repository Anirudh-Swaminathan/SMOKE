import torch
from torch import nn

from .smoke_predictor import make_smoke_predictor
from .loss import make_smoke_loss_evaluator
from .inference import make_smoke_post_processor

import logging


class SMOKEHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super(SMOKEHead, self).__init__()

        self.logger = logging.getLogger(__name__)
        self.logger.info("In SMOKEHead() __init__(). Calling make_smoke_predictor()")

        self.cfg = cfg.clone()
        self.predictor = make_smoke_predictor(cfg, in_channels)
        self.logger.info("Back in SMOKEHead constructor. Calling make_smoke_loss_evaluator()")
        self.loss_evaluator = make_smoke_loss_evaluator(cfg)
        self.logger.info("Back in SMOKEHead constructor. Calling make_smoke_post_processor()")
        self.post_processor = make_smoke_post_processor(cfg)
        self.logger.info("End of SMOKEHead constructor")

    def forward(self, features, targets=None):
        x = self.predictor(features)

        if self.training:
            loss_heatmap, loss_regression = self.loss_evaluator(x, targets)

            return {}, dict(hm_loss=loss_heatmap,
                            reg_loss=loss_regression, )
        if not self.training:
            result = self.post_processor(x, targets)

            return result, {}


def build_smoke_head(cfg, in_channels):
    logger = logging.getLogger(__name__)
    logging.info("IN build_smoke_head(). Initializing object")
    return SMOKEHead(cfg, in_channels)
