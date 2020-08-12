from . import transforms as T

import logging


def build_transforms(cfg, is_train=True):
    l = logging.getLogger(__name__)
    l.info("IN build_transforms()")
    to_bgr = cfg.INPUT.TO_BGR

    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr=to_bgr
    )

    transform = T.Compose(
        [
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform
