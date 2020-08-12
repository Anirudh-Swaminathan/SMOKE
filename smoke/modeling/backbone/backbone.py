from collections import OrderedDict

from torch import nn

from smoke.modeling import registry
from . import dla

import logging

@registry.BACKBONES.register("DLA-34-DCN")
def build_dla_backbone(cfg):
    logger = logging.getLogger(__name__)
    logger.info("in build_dla_backbone()")
    body = dla.DLA(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS
    return model


def build_backbone(cfg):
    logger = logging.getLogger(__name__)
    logger.info("In build_backbone(). Returning build_dla_backbone()")
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
