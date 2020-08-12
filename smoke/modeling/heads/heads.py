import torch

from .smoke_head.smoke_head import build_smoke_head

import logging


def build_heads(cfg, in_channels):
    logger = logging.getLogger(__name__)
    logger.info("In build_heads()! Calling build_smoke_head()")
    if cfg.MODEL.SMOKE_ON:
        return build_smoke_head(cfg, in_channels)
