"""
-------------------------------------------------
   File Name:    main.py
   Author:       Adarsh k
   Date:         2021/04/25
   Description:  Modified from:
                 https://github.com/kadarsh22/Disentanglementlibpytorch
-------------------------------------------------
"""

import sys
from config import get_config, save_config
from train import Trainer
from training_wrapper import run_training_wrapper
from evaluation_wrapper import run_evaluation_wrapper
from logger import PerfomanceLogger
from data_loader import get_data_loader
import logging


def main(configurations, opt):
    logging.info("Executing for random seed : "+str(opt.random_seed))
    Trainer.set_seed(opt.random_seed)
    PerfomanceLogger.configure_logger(configurations)
    perf_logger = PerfomanceLogger()
    save_config(configurations, opt)
    data = get_data_loader(configurations, opt)
    if configurations['evaluation']:
        run_evaluation_wrapper(configurations, data, perf_logger)
    else:
        run_training_wrapper(configurations, opt, data, perf_logger)


if __name__ == "__main__":
    config, options = get_config(sys.argv[1:])
    main(config, options)
