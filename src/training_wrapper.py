"""
-------------------------------------------------
   File Name:    training_wrapper.py
   Author:       Adarsh k
   Date:         2021/04/25
   Description:  Modified from:
                 https://github.com/kadarsh22/Disentanglementlibpytorch
-------------------------------------------------
"""
import logging

from model_loader import get_model
from train import Trainer
from saver import Saver
from utils import *


def run_training_wrapper(configuration, opt, perf_logger):
    directories = list_dir_recursively_with_ignore('.', ignores=['checkpoint.pt', '__pycache__'])
    filtered_dirs = []
    for file in directories:
        x, y = file
        if x.count('/') < 3:
            filtered_dirs.append((x, y))
    files = [(f[0], os.path.join(opt.result_dir, "src_copy", f[1])) for f in filtered_dirs]

    copy_files_and_create_dirs(files)

    perf_logger.start_monitoring("Fetching data, models and class instantiations")
    models = get_model(opt)
    model_trainer = Trainer(configuration, opt)
    saver = Saver(configuration)
    perf_logger.stop_monitoring("Fetching data, models and class instantiations")

    generator, deformator, deformator_opt, rank_predictor, rank_predictor_opt = models
    generator.generator.eval()
    deformator.train()
    for step in range(opt.algo.ours.num_steps):
        directions = model_trainer.train_ganspace(generator)

