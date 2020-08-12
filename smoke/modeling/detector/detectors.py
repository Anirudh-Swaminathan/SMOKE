from .keypoint_detector import KeypointDetector
import logging

def build_detection_model(cfg):
    logger = logging.getLogger(__name__)
    logger.info("build_detection_model() now. Calling KeyPointDetector() next")
    return KeypointDetector(cfg)