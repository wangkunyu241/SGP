import logging

import torch
import torch.optim as optim

from robustbench.data import load_cifar100c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy

import pruning_100c
import torch.nn as nn
from conf import cfg, load_cfg_fom_args
import numpy as np

logger = logging.getLogger(__name__)


def evaluate(description):
    args = load_cfg_fom_args(description)

    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions)

    checkpoint = torch.load("../../resource/pretrain_cifar100.t7")
    new_state_dict = {}
    for k, v in checkpoint['model'].items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v
    base_model.load_state_dict(new_state_dict, strict=True)
    logger.info('load modify head cifar100 ckpt successful')

    base_model.cuda()

    if cfg.MODEL.ADAPTATION == "pruning":
        logger.info("test-time adaptation: PRUNING")
        model = setup_pruning(base_model)

    # evaluate on each severity and type of corruption in turn
    All_error = []
    for severity in cfg.CORRUPTION.SEVERITY:
        for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):

            x_test, y_test = load_cifar100c(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
                                           [corruption_type])
            x_test = torch.nn.functional.interpolate(x_test, size=(args.size, args.size), \
                mode='bilinear', align_corners=False)
            acc = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE, device = 'cuda')
            err = 1. - acc
            All_error.append(err)
            logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")
    
    final_score = np.mean(All_error)
    logger.info(f"Trial Finished Successfully. Final Mean Error: {final_score:.2%}")


def setup_pruning(model):
    model = pruning_100c.configure_model(model)
    params, param_names = pruning_100c.collect_params(model)
    optimizer = setup_optimizer(params)
    pruning_model = pruning_100c.Pruning(model, optimizer,
                           lamda_std=cfg.PRUNING.LAMDA_STD,
                           lamda_align=cfg.PRUNING.LAMDA_ALIGN,
                           e_margin=cfg.PRUNING.MARGIN,
                           rate=cfg.PRUNING.RATE,
                           lamda_sparse=cfg.PRUNING.LAMDA_SPARSE,
                           threshold=cfg.PRUNING.THRESHOLD,
                           lr_mask=cfg.PRUNING.LR,
                           steps=cfg.OPTIM.STEPS)
    return pruning_model


def setup_optimizer(params):
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=0.9,
                   dampening=0,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=True)
    elif cfg.OPTIM.METHOD == 'AdamW':
        return optim.AdamW(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    evaluate('"CIFAR-100-C evaluation.')
