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
    metrics_seed = {'betavae_metric': [], 'factorvae_metric': [], 'mig': [], 'dci': []}
    directories = list_dir_recursively_with_ignore('.', ignores=['checkpoint.pt', '__pycache__'])
    filtered_dirs = []
    for file in directories:
        x, y = file
        if x.count('/') < 3:
            filtered_dirs.append((x, y))
    files = [(f[0], os.path.join(opt.result_dir, "src", f[1])) for f in filtered_dirs]
    copy_files_and_create_dirs(files)
    for i in range(5,opt.num_generator_seeds):
        logging.info("Running for generator model : " + str(i))
        resume_step = 0
        opt.pretrained_gen_path = opt.pretrained_gen_root + opt.dataset + '/' + str(i) + '.pt'
        perf_logger.start_monitoring("Fetching data, models and class instantiations")
        models = get_model(opt)
        model_trainer = Trainer(configuration, opt)
        evaluator = Evaluator(configuration, opt)
        saver = Saver(configuration)
        visualise_results = Visualiser(configuration, opt)
        perf_logger.stop_monitoring("Fetching data, models and class instantiations")
        if opt.algorithm == 'LD':
            generator, deformator, shift_predictor, deformator_opt, shift_predictor_opt = models
            if configuration['resume_train']:
                deformator, shift_predictor, deformator_opt, shift_predictor_opt, resume_step = saver.load_model(
                    (deformator, shift_predictor, deformator_opt, shift_predictor_opt), algo='LD')
            # plot_generated_images(opt, generator)
            generator.to(device).eval()
            deformator.to(device).train()
            shift_predictor.to(device).train()
            loss, logit_loss, shift_loss = 0, 0, 0
            for k in range(resume_step+1, opt.algo.ld.num_steps):
                start_time = time.time()
                deformator, shift_predictor, deformator_opt, shift_predictor_opt, losses = \
                    model_trainer.train_latent_discovery(
                        generator, deformator, shift_predictor, deformator_opt,
                        shift_predictor_opt)
                loss = loss + losses[0]
                logit_loss = logit_loss + losses[1]
                shift_loss = shift_loss + losses[2]
                if k % opt.algo.ld.logging_freq == 0 and k != 0:
                    metrics = evaluator.compute_metrics(generator, deformator, data, epoch=0)
                    # accuracy = evaluator.evaluate_model(generator, deformator, shift_predictor, model_trainer)
                    total_loss, logit_loss, shift_loss = losses
                    logging.info(
                        "Step  %d / %d Time taken %d sec loss: %.5f  logitLoss: %.5f, shift_Loss %.5F " % (
                            k, opt.algo.ld.num_steps, time.time() - start_time,
                            total_loss / opt.algo.ld.logging_freq, logit_loss / opt.algo.ld.logging_freq,
                            shift_loss / opt.algo.ld.logging_freq))
                    perf_logger.start_monitoring("Latent Traversal Visualisations")
                    deformator_layer = torch.nn.Linear(opt.algo.ld.num_directions, opt.algo.ld.latent_dim)
                    if opt.algo.ld.deformator_type == 'ortho':
                        deformator_layer.weight.data = torch.FloatTensor(deformator.ortho_mat.data.cpu())
                    else:
                        deformator_layer.weight.data = torch.FloatTensor(deformator.linear.weight.data.cpu())
                    visualise_results.make_interpolation_chart(i, generator, deformator_layer, shift_r=10,
                                                               shifts_count=5)
                    perf_logger.stop_monitoring("Latent Traversal Visualisations")
                    loss, logit_loss, shift_loss = 0, 0, 0
                if k % opt.algo.ld.saving_freq == 0 and k != 0:
                    params = (deformator, shift_predictor, deformator_opt, shift_predictor_opt)
                    perf_logger.start_monitoring("Saving Model")
                    saver.save_model(params, k, i , algo='LD')
                    perf_logger.stop_monitoring("Saving Model")
        elif opt.algorithm == 'CF':
            generator = models
            plot_generated_images(opt, generator)
            directions = model_trainer.train_closed_form(generator)
            visualise_results.make_interpolation_chart(i, generator, directions, shift_r=10, shifts_count=5)
            metrics = evaluator.compute_metrics(generator, directions, data, epoch=0)
        elif opt.algorithm == 'GS':
            generator = models
            plot_generated_images(opt, generator)
            directions = model_trainer.train_ganspace(generator)
            visualise_results.make_interpolation_chart(i, generator, directions, shift_r=10, shifts_count=5)
            metrics = evaluator.compute_metrics(generator, directions, data, epoch=0)
        elif opt.algorithm == 'ours':
            generator,deformator, deformator_opt, cr_discriminator, cr_optimizer = models
            initialisation = model_trainer.train_closed_form(generator)
            deformator.ortho_mat.data = initialisation.weight
            deformator.cuda()
 #           deformator_opt = torch.optim.Adam(deformator.parameters(), lr=opt.algo.ours.deformator_lr)
 #           metrics = evaluator.compute_metrics(generator, deformator, data, epoch=0)
 #           logging.info("---------------------Closed form initialisation results------------------------")
 #           logging.info("BetaVAE Metric : " + str(metrics['beta_vae']['eval_accuracy']))
 #           logging.info("Factor Metric : " + str(metrics['factor_vae']['eval_accuracy']))
 #           logging.info("MIG : " + str(metrics['mig']))
 #           logging.info("DCI Metric : " + str(metrics['dci']))
 #           deformator, deformator_opt, cr_discriminator, cr_optimizer = saver.load_model((deformator, deformator_opt, cr_discriminator, cr_optimizer), algo='ours')
            deformator.train()
            for k in range(opt.algo.ours.num_steps):
                deformator, deformator_opt, cr_discriminator, cr_optimizer, losses = \
                    model_trainer.train_ours(
                        generator, deformator, deformator_opt, cr_discriminator, cr_optimizer)
                if k % opt.algo.ld.saving_freq == 0 and k != 0:
                    params = (deformator, deformator_opt, cr_discriminator, cr_optimizer)
                    perf_logger.start_monitoring("Saving Model")
                    saver.save_model(params, k, i , algo='ours')
                    perf_logger.stop_monitoring("Saving Model")

                if k % opt.algo.ours.logging_freq == 0 and k!=0:
                    metrics = evaluator.compute_metrics(generator, deformator, data, epoch=0)
                    perf_logger.start_monitoring("Latent Traversal Visualisations")
                    deformator_layer = torch.nn.Linear(opt.algo.ours.num_directions,
                                                       opt.algo.ours.latent_dim)
                    if opt.algo.ours.deformator_type == 'ortho':
                        deformator_layer.weight.data = torch.FloatTensor(deformator.ortho_mat.data.cpu())
                    else:
                        deformator_layer.weight.data = torch.FloatTensor(deformator.weight.data.cpu())
                    visualise_results.make_interpolation_chart(i, generator, deformator_layer, shift_r=10,
                                                               shifts_count=5)
                    perf_logger.stop_monitoring("Latent Traversal Visualisations")
        else:
            raise NotImplementedError
        metrics_seed['betavae_metric'].append(metrics['beta_vae']['eval_accuracy'])
        metrics_seed['factorvae_metric'].append(metrics['factor_vae']['eval_accuracy'])
        metrics_seed['mig'].append(metrics['mig'])
        metrics_seed['dci'].append(metrics['dci'])

    if opt.dataset != 'dsprites':  # since running only for one seed ..not enough points for std dev calculation
        logging.info('BetaVAE metric : ' + str(mean(metrics_seed['betavae_metric'])) + u"\u00B1" + str(
            stdev(metrics_seed['betavae_metric'])) + '\n' +
                     'FactorVAE metric : ' + str(mean(metrics_seed['factorvae_metric'])) + u"\u00B1" + str(
            stdev(metrics_seed['factorvae_metric'])) + '\n'
                                                       'MIG : ' + str(mean(metrics_seed['mig'])) + u"\u00B1" + str(
            stdev(metrics_seed['mig'])) + '\n' +
                     'DCI:' + str(mean(metrics_seed['dci'])) + u"\u00B1" + str(stdev(metrics_seed['dci'])))
