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
from face_identity_score import compute_face_identity_score


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
    should_gen_classes = False
    if opt.gan_type == 'BigGAN':
        should_gen_classes = True
    generator.eval()
    deformator.train()
    rank_predictor_loss_list = []
    deformator_ranking_loss_list = []
    # deformator, rank_predictor, deformator_opt, rank_predictor_opt = saver.load_model((deformator, deformator_opt, rank_predictor, rank_predictor_opt))
    face_identity_score = compute_face_identity_score(generator, deformator,opt,[23,108,10,136])
    for step in range(opt.algo.ours.num_steps):
        deformator, deformator_opt, rank_predictor, rank_predictor_opt, rank_predictor_loss, deformator_ranking_loss = \
            model_trainer.train_ours(generator, deformator, deformator_opt, rank_predictor, rank_predictor_opt,
                                     should_gen_classes)
        rank_predictor_loss_list.append(rank_predictor_loss)
        deformator_ranking_loss_list.append(deformator_ranking_loss)

        if step % opt.algo.ours.logging_freq == 0:
            rank_predictor_loss_avg = sum(rank_predictor_loss_list) / len(rank_predictor_loss_list)
            deformator_ranking_loss_avg = sum(deformator_ranking_loss_list) / len(deformator_ranking_loss_list)
            logging.info("step : %d / %d Rank predictor loss : %.4f Deformator_ranking loss  %.4f " % (
                step, opt.algo.ours.num_steps, rank_predictor_loss_avg, deformator_ranking_loss_avg))
            rank_predictor_loss_list = []
            deformator_ranking_loss_list = []

        if step % opt.algo.ours.saving_freq == 0:
            params = (deformator, deformator_opt, rank_predictor, rank_predictor_opt)
            perf_logger.start_monitoring("Saving Model")
            saver.save_model(params, step)
            perf_logger.stop_monitoring("Saving Model")
