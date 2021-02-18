from model_loader import get_model
from train import Trainer
from evaluation import Evaluator
from saver import Saver
from visualiser import Visualiser
from config import save_config
import logging
import torch
import time
device = torch.device('cuda:0')

def run_training_wrapper(configuration, perf_logger):
	for key, values in configuration.items():
		logging.info(' {} : {}'.format(key, values))
	device = torch.device('cuda:' + str(configuration['device_id']))
	save_config(configuration)
	perf_logger.start_monitoring("Fetching data, models and class instantiations")
	generator ,deformator , shift_predictor , deformator_opt ,shift_predictor_opt = get_model(configuration)
	model_trainer = Trainer(configuration)
	evaluator = Evaluator(configuration)
	saver = Saver(configuration)
	visualise_results = Visualiser(configuration)
	# deformator, shift_predictor, deformator_opt, shift_predictor_opt = saver.load_model(deformator,shift_predictor,deformator_opt,shift_predictor_opt)
	perf_logger.stop_monitoring("Fetching data, models and class instantiations")
	generator.to(device).eval()
	deformator.to(device).train()
	shift_predictor.to(device).train()
	start_time = time.time()
	for i in range(configuration['num_steps']):
		deformator, shift_predictor, deformator_opt, shift_predictor_opt ,losses = model_trainer.train_gan(generator, deformator , shift_predictor, deformator_opt,
														 shift_predictor_opt)
		if i % configuration['saving_freq'] == 0 and i!=0:
			perf_logger.start_monitoring("Saving Model")
			saver.save_model(deformator, shift_predictor , deformator_opt, shift_predictor_opt, i)
			perf_logger.stop_monitoring("Saving Model")

		if i % configuration['logging_freq'] == 0 and i!=0:
			accuracy = evaluator.evaluate_model(generator, deformator, shift_predictor,model_trainer)
			total_loss, logit_loss , shift_loss  = losses
			logging.info("Step  %d / %d Time taken %d sec loss: %.5f  logitLoss: %.5f, shift_Loss %.5F  Accuracy %.5f" % (
				i , configuration['num_steps'], time.time() - start_time,total_loss/configuration['logging_freq'] , logit_loss/configuration['logging_freq'],shift_loss/configuration['logging_freq'],accuracy*100))
			perf_logger.start_monitoring("Latent Traversal Visualisations")
			visualise_results.make_interpolation_chart(i, generator, deformator,z=None,
								 shift_r=10, shifts_count = 5, dims = None, dims_count = 10, texts = None)
			perf_logger.stop_monitoring("Latent Traversal Visualisations")
			start_time = time.time()
			model_trainer.shift_loss = 0
			model_trainer.logit_loss = 0
			model_trainer.loss = 0
