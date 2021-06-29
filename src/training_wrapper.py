"""
-------------------------------------------------
   File Name:    training_wrapper.py
   Author:       Adarsh k
   Date:         2021/04/25
   Description:  Modified from:
                 https://github.com/kadarsh22/Disentanglementlibpytorch
-------------------------------------------------
"""
from model_loader import get_model
from train import Trainer
from evaluation import Evaluator
from saver import Saver
from visualiser import Visualiser, plot_generated_images
import logging
import torch
import time
from statistics import mean, stdev
from utils import *


def run_training_wrapper(configuration, opt, data, perf_logger):
    device = torch.device(opt.device + opt.device_id)

    directories = list_dir_recursively_with_ignore('.', ignores=['checkpoint.pt', '__pycache__'])
    filtered_dirs = []
    for file in directories:
        x, y = file
        if x.count('/') < 3:
            filtered_dirs.append((x, y))
    files = [(f[0], os.path.join(opt.result_dir, "src", f[1])) for f in filtered_dirs]
    copy_files_and_create_dirs(files)

    for i in range(opt.num_seeds):
        resume_step = 0
        opt.pretrained_gen_path = opt.pretrained_gen_root + opt.dataset + '/' + str(i) + '.pt'
        perf_logger.start_monitoring("Fetching data, models and class instantiations")
        models = get_model(opt)
        model_trainer = Trainer(configuration, opt)
        evaluator = Evaluator(configuration, opt)
        saver = Saver(configuration)
        visualise_results = Visualiser(configuration, opt)
        perf_logger.stop_monitoring("Fetching data, models and class instantiations")
        train_loader , _ = data
        if opt.algorithm == 'discrete_ld':
            generator,discriminator,disc_opt, deformator, shift_predictor, deformator_opt, shift_predictor_opt = models
            if configuration['resume_train']:
                deformator, shift_predictor, deformator_opt, shift_predictor_opt, resume_step = saver.load_model(
                    (deformator, shift_predictor, deformator_opt, shift_predictor_opt), algo='LD')
            # plot_generated_images(opt, generator)
            loss, logit_loss, shift_loss = 0, 0, 0

            for k in range(resume_step + 1, opt.algo.discrete_ld.num_steps):
                generator.to(device).eval()
                deformator.to(device).train()
                # shift_predictor.to(device).train()
                start_time = time.time()
                deformator,discriminator,disc_opt, shift_predictor, deformator_opt, shift_predictor_opt, losses = \
                    model_trainer.train_discrete_ld(
                        generator,discriminator,disc_opt, deformator, shift_predictor, deformator_opt,
                        shift_predictor_opt, train_loader)
                logit_loss = logit_loss + losses[1]
                if k % opt.algo.discrete_ld.logging_freq == 0 and k != 0:
                    total_loss, logit_loss, shift_loss = losses
                    logging.info(
                        "Step  %d / %d Time taken %d sec , logitLoss: %.5f " % (
                            k, opt.algo.discrete_ld.num_steps, time.time() - start_time,
                            logit_loss / opt.algo.discrete_ld.logging_freq))
                    perf_logger.start_monitoring("Latent Traversal Visualisations")
                    deformator_layer = torch.nn.Linear(opt.algo.discrete_ld.num_directions, generator.dim_z, bias=False)
                    if opt.algo.discrete_ld.deformator_type == 'ortho':
                        deformator_layer.weight.data = torch.FloatTensor(deformator.ortho_mat.data.cpu())
                    else:
                        deformator_layer.weight.data = torch.FloatTensor(deformator.linear.weight.data.cpu())
                    z = torch.randn(100, generator.dim_z)
                    visualise_results.make_interpolation_chart(k, z, generator, deformator_layer, shift_r=10,
                                                               shifts_count=5)
                    # metrics = evaluator.compute_metrics_discrete_ld(data, shift_predictor)
                    perf_logger.stop_monitoring("Latent Traversal Visualisations")
                    logit_loss = 0
                # if k % opt.algo.discrete_ld.saving_freq == 0 and k != 0:
                #     params = (deformator, shift_predictor, deformator_opt, shift_predictor_opt)
                #     perf_logger.start_monitoring("Saving Model")
                #     saver.save_model(params, k, i, algo='LD')
                #     perf_logger.stop_monitoring("Saving Model")
