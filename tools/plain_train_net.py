import torch

from smoke.config import cfg
from smoke.data import make_data_loader
from smoke.solver.build import (
    make_optimizer,
    make_lr_scheduler,
)
from smoke.utils.check_point import DetectronCheckpointer
from smoke.engine import (
    default_argument_parser,
    default_setup,
    launch,
)
from smoke.utils import comm
from smoke.engine.trainer import do_train
from smoke.modeling.detector import build_detection_model
from smoke.engine.test_net import run_test

import logging

def train(cfg, model, device, distributed):
    l = logging.getLogger(__name__)
    l.info("IN train()\n Calling make_optimizer()")
    optimizer = make_optimizer(cfg, model)
    l.info("Back in train(). Calling make_lr_scheduler()")
    scheduler = make_lr_scheduler(cfg, optimizer)
    l.info("Back in train(). Calling DetectronCheckpointer")

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR
    save_to_disk = comm.get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    l.info("Back in train(). Calling make_data_loader()")
    data_loader = make_data_loader(
        cfg,
        is_train=True,
    )
    l.info("Back in train(). Calling do_train()")

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        cfg,
        distributed,
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments
    )
    l.info("END of train()")


def setup(args):
    logger = logging.getLogger(__name__)
    logging.info("setup() called with args: {}\nReturning to main()".format(args))
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    return cfg


def main(args):
    logger = logging.getLogger(__name__)
    logger.info("main() function starting! calling setup")
    cfg = setup(args)

    logger.info("calling build_detection_model() now")
    model = build_detection_model(cfg)
    logger.info("Back in main()")
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    if args.eval_only:
        checkpointer = DetectronCheckpointer(
            cfg, model, save_dir=cfg.OUTPUT_DIR
        )
        ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
        _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)
        return run_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            find_unused_parameters=True,
        )

    logger.info("Calling train() method!")
    train(cfg, model, device, distributed)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
