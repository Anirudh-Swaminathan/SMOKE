import os

from smoke.data import build_test_loader
from smoke.engine.inference import inference
from smoke.utils import comm
from smoke.utils.miscellaneous import mkdir

import logging


def run_test(cfg, model):
    l = logging.getLogger(__name__)
    l.info("IN run_test()!")
    eval_types = ("detection",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    l.info("Calling build_test_loader()")
    data_loaders_val = build_test_loader(cfg)
    l.info("Back in run_test()\nLoop over all datasets and loaders now!")
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        l.info("Calling inference!")
        inference(
            model,
            data_loaders_val,
            dataset_name=dataset_name,
            eval_types=eval_types,
            device=cfg.MODEL.DEVICE,
            output_folder=output_folder,
        )
        comm.synchronize()
    l.info("End of run_test()")
