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
from visualiser import Visualiser
import logging
import torch
import time
from statistics import mean, stdev


def run_training_wrapper(configuration, opt, data, perf_logger):
    for key, values in configuration.items():
        logging.info(' {} : {}'.format(key, values))
    device = torch.device(opt.device + opt.device_id)
    metrics_seed = {'betavae_metric': [], 'factorvae_metric': [], 'mig': [], 'dci': []}
    for i in range(opt.num_seeds-1):
        opt.pretrained_gen_path = opt.pretrained_gen_root + opt.dataset + '/' + str(i) + '.pt'
        perf_logger.start_monitoring("Fetching data, models and class instantiations")
        models = get_model(configuration, opt)
        model_trainer = Trainer(configuration, opt)
        evaluator = Evaluator(configuration, opt)
        saver = Saver(configuration)
        visualise_results = Visualiser(configuration, opt)
        perf_logger.stop_monitoring("Fetching data, models and class instantiations")

        if opt.algorithm == 'LD':
            generator, deformator, shift_predictor, deformator_opt, shift_predictor_opt = models
            # visualise_results.plot_generated_images(opt, generator)
            generator.to(device).eval()
            deformator.to(device).train()
            shift_predictor.to(device).train()
            loss, logit_loss, shift_loss = 0, 0, 0
            for k in range(opt.algo.ld.num_steps):
                start_time = time.time()
                deformator, shift_predictor, deformator_opt, shift_predictor_opt, losses = \
                    model_trainer.train_latent_discovery(
                        generator, deformator, shift_predictor, deformator_opt,
                        shift_predictor_opt)
                loss = loss + losses[0]
                logit_loss = logit_loss + losses[1]
                shift_loss = shift_loss + losses[2]
                if k % opt.algo.ld.saving_freq == 0 and k != 0:
                    params = (deformator, shift_predictor, deformator_opt, shift_predictor_opt)
                    perf_logger.start_monitoring("Saving Model")
                    saver.save_model(params, i, algo='LD')
                    total_loss, logit_loss, shift_loss = losses
                    logging.info(
                        "Step  %d / %d Time taken %d sec loss: %.5f  logitLoss: %.5f, shift_Loss %.5F " % (
                            k, opt.algo.ld.num_steps, time.time() - start_time,
                            total_loss / opt.algo.ld.logging_freq, logit_loss / opt.algo.ld.logging_freq,
                            shift_loss / opt.algo.ld.logging_freq))
                    loss, logit_loss, shift_loss = 0, 0, 0
                    perf_logger.stop_monitoring("Saving Model")

                if k % opt.algo.ld.logging_freq == 0 and k != 0:
                    metrics = evaluator.compute_metrics(generator, deformator, data, epoch=0)
                    # accuracy = evaluator.evaluate_model(generator, deformator, shift_predictor, model_trainer)
                    total_loss, logit_loss, shift_loss = losses
                    logging.info(
                        "Step  %d / %d Time taken %d sec loss: %.5f  logitLoss: %.5f, shift_Loss %.5F " % (
                            i, opt.algo.ld.num_steps, time.time() - start_time,
                            total_loss / opt.algo.ld.logging_freq, logit_loss / opt.algo.ld.logging_freq,
                            shift_loss / opt.algo.ld.logging_freq))
                    perf_logger.start_monitoring("Latent Traversal Visualisations")
                    visualise_results.make_interpolation_chart(i, generator, deformator,
                                                               shift_r=10, shifts_count=5, dims_count=5)
                    perf_logger.stop_monitoring("Latent Traversal Visualisations")
                    start_time = time.time()
                    loss, logit_loss, shift_loss = 0, 0, 0
        elif opt.algorithm == 'CF':
            generator = models
            directions = model_trainer.train_closed_form(generator)
            visualise_results.make_interpolation_chart(i, generator, directions, shift_r=10, shifts_count=5)
            metrics = evaluator.compute_metrics(generator, directions, data, epoch=0)
        elif opt.algorithm == 'GS':
            generator = models
            directions = model_trainer.train_ganspace(generator)
            visualise_results.make_interpolation_chart(i, generator, directions, shift_r=10, shifts_count=5)
            metrics = evaluator.compute_metrics(generator, directions, data, epoch=0)
        elif opt.algorithm == 'Cnn Train':
            from models.classifier import CRDiscriminator
            classifier = CRDiscriminator(dim_c_cont=2).cuda()
            cr_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.002, betas=(0.9, 0.999))

            classifier.train()
            for iteration in range(5000):
                cr_optimizer.zero_grad()
                label_real = torch.full((64,), 1, dtype=torch.long, device='cuda')
                label_fake = torch.full((64,), 0, dtype=torch.long, device='cuda')
                labels = torch.cat((label_real, label_fake))
        else:
            raise NotImplementedError
        metrics_seed['betavae_metric'].append(metrics['beta_vae']['eval_accuracy'])
        metrics_seed['factorvae_metric'].append(metrics['factor_vae']['eval_accuracy'])
        metrics_seed['mig'].append(metrics['mig'])
        metrics_seed['dci'].append(metrics['dci'])

    logging.info('BetaVAE metric : ' + str(mean(metrics_seed['betavae_metric'])) + u"\u00B1" + str(
        stdev(metrics_seed['betavae_metric'])) + '\n' +
                 'FactorVAE metric : ' + str(mean(metrics_seed['factorvae_metric'])) + u"\u00B1" + str(
        stdev(metrics_seed['factorvae_metric'])) + '\n'
                                                   'MIG : ' + str(mean(metrics_seed['mig'])) + u"\u00B1" + str(
        stdev(metrics_seed['mig'])) + '\n' +
                 'DCI:' + str(mean(metrics_seed['dci'])) + u"\u00B1" + str(stdev(metrics_seed['dci'])))
