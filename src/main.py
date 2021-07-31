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
from logger import PerfomanceLogger


def main(configurations, opt):
    Trainer.set_seed(opt.random_seed)
    PerfomanceLogger.configure_logger(configurations)
    perf_logger = PerfomanceLogger()
    save_config(configurations, opt)
    run_training_wrapper(configurations, opt, perf_logger)


if __name__ == "__main__":
    config, options = get_config(sys.argv[1:])
    main(config, options)
