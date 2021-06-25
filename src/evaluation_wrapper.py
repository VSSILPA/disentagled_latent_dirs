from model_loader import get_model
from evaluation import Evaluator
from train import Trainer
from saver import Saver
from visualiser import Visualiser
import torch
from config import save_config
import logging


def run_evaluation_wrapper(configuration, opt ,data, perf_logger):
	for key, values in configuration.items():
		logging.info(' {} : {}'.format(key, values))
	Trainer.set_seed(opt.random_seed)
	save_config(configuration, opt)
	perf_logger.start_monitoring("Fetching data, models and class instantiations")
	model = get_model(opt)
	evaluator = Evaluator(configuration,opt)
	saver = Saver(configuration)
	visualise_results = Visualiser(configuration,opt)
	perf_logger.stop_monitoring("Fetching data, models and class instantiations")

	# model, optimizer, loss = saver.load_model(model=model, optimizer=optimizer)
	metrics = evaluator.compute_metrics(data, model)
	# visualise_results.visualise_latent_traversal(z, model.decoder, 0)
	saver.save_results(metrics, 'metrics')
