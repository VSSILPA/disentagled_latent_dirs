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
from config import save_config
import logging
import torch
import time
import numpy as np


def run_training_wrapper(configuration, opt, data, perf_logger):
    for key, values in configuration.items():
        logging.info(' {} : {}'.format(key, values))
    device = torch.device(opt.device + opt.device_id)
    save_config(configuration, opt)
    perf_logger.start_monitoring("Fetching data, models and class instantiations")
    models = get_model(configuration, opt)
    model_trainer = Trainer(configuration, opt)
    evaluator = Evaluator(configuration, opt)
    saver = Saver(configuration)
    visualise_results = Visualiser(configuration, opt)
    # deformator, shift_predictor, deformator_opt, shift_predictor_opt = saver.load_model(deformator,shift_predictor,
    # deformator_opt,shift_predictor_opt)
    perf_logger.stop_monitoring("Fetching data, models and class instantiations")

    if opt.algorithm == 'LD':
        generator, deformator, shift_predictor, deformator_opt, shift_predictor_opt = models
        visualise_results.plot_generated_images(opt, generator)
        generator.to(device).eval()
        deformator.to(device).train()
        shift_predictor.to(device).train()
        loss, logit_loss, shift_loss = 0, 0, 0
        start_time = time.time()
        for i in range(opt.num_steps):
            deformator, shift_predictor, deformator_opt, shift_predictor_opt, losses = \
                model_trainer.train_latent_discovery(
                    generator, deformator, shift_predictor, deformator_opt,
                    shift_predictor_opt)
            loss = loss + losses[0]
            logit_loss = logit_loss + losses[1]
            shift_loss = shift_loss + losses[2]
            if i % opt.saving_freq == 0 and i != 0:
                params = (deformator, shift_predictor, deformator_opt, shift_predictor_opt)
                perf_logger.start_monitoring("Saving Model")
                saver.save_model(params, i, algo='LD')
                perf_logger.stop_monitoring("Saving Model")

            if i % opt.logging_freq == 0 and i != 0:
                metrics = evaluator.compute_metrics(generator, deformator, data, epoch=0)
                accuracy = evaluator.evaluate_model(generator,deformator,shift_predictor,model_trainer)
                total_loss, logit_loss, shift_loss = losses
                logging.info(
                    "Step  %d / %d Time taken %d sec loss: %.5f  logitLoss: %.5f, shift_Loss %.5F Accuracy %.3F " % (
                        i, opt.num_steps, time.time() - start_time,
                        total_loss / opt.logging_freq, logit_loss / opt.logging_freq,
                        shift_loss / opt.logging_freq,accuracy))
                perf_logger.start_monitoring("Latent Traversal Visualisations")
                visualise_results.make_interpolation_chart(i, generator, deformator,
                                                           shift_r=10, shifts_count=5, dims_count=5)
                perf_logger.stop_monitoring("Latent Traversal Visualisations")
                start_time = time.time()
                loss, logit_loss, shift_loss = 0, 0, 0
    elif opt.algorithm == 'CF':
        generator = models
        weights = []
        weights.append(generator.g_synthesis.init_block.epi1.style_mod.lin.weight.T.cpu().detach().numpy())
        weights.append(generator.g_synthesis.init_block.epi2.style_mod.lin.weight.T.cpu().detach().numpy())
        weights.append(generator.g_synthesis.blocks[0].epi1.style_mod.lin.weight.T.cpu().detach().numpy())
        weights.append(generator.g_synthesis.blocks[0].epi2.style_mod.lin.weight.T.cpu().detach().numpy())
        weights.append(generator.g_synthesis.blocks[1].epi1.style_mod.lin.weight.T.cpu().detach().numpy())
        weights.append(generator.g_synthesis.blocks[1].epi2.style_mod.lin.weight.T.cpu().detach().numpy())
        weights.append(generator.g_synthesis.blocks[2].epi1.style_mod.lin.weight.T.cpu().detach().numpy())
        weights.append(generator.g_synthesis.blocks[2].epi2.style_mod.lin.weight.T.cpu().detach().numpy())
        weights.append(generator.g_synthesis.blocks[3].epi1.style_mod.lin.weight.T.cpu().detach().numpy())
        weights.append(generator.g_synthesis.blocks[3].epi2.style_mod.lin.weight.T.cpu().detach().numpy())
        weight = np.concatenate(weights, axis=1).astype(np.float32)
        weight = weight / np.linalg.norm(weight, axis=0, keepdims=True)
        eigen_values, eigen_vectors = np.linalg.eig(weight.dot(weight.T))
        deformator =  eigen_vectors[:opt.algo.ld.directions_count]
        deformator_layer = torch.nn.Linear(64,512)
        deformator_layer.weight.data = torch.FloatTensor(deformator)
        metrics = evaluator.compute_metrics(generator, deformator_layer, data, epoch=0)
        print(metrics)




    else:
        raise NotImplementedError
