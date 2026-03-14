import os
import logging
import numpy as np
from models.model import get_model
from utils import get_accuracy, eval_domain_dict
from datasets.data_loading import get_test_loader
from conf import cfg, load_cfg_from_args, get_num_classes, get_domain_sequence, adaptation_method_lookup
from methods.pruning import Pruning

logger = logging.getLogger(__name__)


def evaluate(description):
    load_cfg_from_args(description)

    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)
    base_model = get_model(cfg, num_classes, device=cfg.DEVICE)

    model = eval(f'{adaptation_method_lookup(cfg.MODEL.ADAPTATION)}')(cfg=cfg, model=base_model, num_classes=num_classes)
    logger.info(f"Successfully prepared test-time adaptation method: {cfg.MODEL.ADAPTATION.upper()}")

    dom_names_all = cfg.CORRUPTION.TYPE
    dom_names_loop = dom_names_all
    severities = cfg.CORRUPTION.SEVERITY

    errs = []
    errs_5 = []
    domain_dict = {}

    for i_dom, domain_name in enumerate(dom_names_loop):
        for severity in severities:
            test_data_loader = get_test_loader(setting=cfg.SETTING,
                                            adaptation=cfg.MODEL.ADAPTATION,
                                            dataset_name=cfg.CORRUPTION.DATASET,
                                            root_dir=cfg.DATA_DIR,
                                            domain_name=domain_name,
                                            severity=severity,
                                            num_examples=cfg.CORRUPTION.NUM_EX,
                                            rng_seed=cfg.RNG_SEED,
                                            domain_names_all=dom_names_all,
                                            alpha_dirichlet=cfg.TEST.ALPHA_DIRICHLET,
                                            batch_size=cfg.TEST.BATCH_SIZE,
                                            shuffle=False,
                                            workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()))

            # evaluate the model
            acc, domain_dict = get_accuracy(model,
                                            data_loader=test_data_loader,
                                            dataset_name=cfg.CORRUPTION.DATASET,
                                            domain_name=domain_name,
                                            setting=cfg.SETTING,
                                            domain_dict=domain_dict,
                                            device=cfg.DEVICE)

            err = 1. - acc
            errs_5.append(err)
            logger.info(f"{cfg.CORRUPTION.DATASET} error % [{domain_name}{severity}][#samples={len(test_data_loader.dataset)}]: {err:.2%}")

    logger.info(f"mean error at 5: {np.mean(errs_5):.2%}")

if __name__ == '__main__':
    evaluate('"Evaluation.')

