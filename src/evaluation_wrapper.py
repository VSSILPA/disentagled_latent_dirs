from model_loader import get_model
from evaluation import Evaluator
from train import Trainer
from saver import Saver
from visualiser import Visualiser
from config import save_config
import logging
import torch


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

	metrics = evaluator.compute_metrics(data, model)
	logging.info(
		"ACC: %.2f NMI: %.2f ARI: %.2f" % (metrics['ACC'], metrics['NMI'], metrics['ARI']))
	if opt.algorithm == 'infogan':
		visualise_results.plot_infogan_grid(model)
	else:
		z = torch.randn(100, generator.dim_z)
		visualise_results.make_interpolation_chart(k, z, generator, deformator_layer, shift_r=10, shifts_count=5)


