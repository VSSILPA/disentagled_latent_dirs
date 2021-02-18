import os
import torch


class Saver(object):
	def __init__(self, config):
		self.config = config
		self.experiment_name = self.config['experiment_name']
		self.model_name = self.config['file_name']

	def save_model(self, deformator , shift_predictor, deformator_opt,shift_predictor_opt, step):
		cwd = os.path.dirname(os.getcwd()) + f'/results/{self.experiment_name}'  # project root
		models_dir = cwd + '/models/'

		if not os.path.exists(models_dir):
			os.makedirs(models_dir)
		torch.save({
			'step': step,
			'deformator': deformator.state_dict(),
			'shift_predictor': shift_predictor.state_dict(),
			'deformator_opt': deformator_opt.state_dict(),
			'shift_predictor_opt': shift_predictor_opt.state_dict()
		}, os.path.join(models_dir, str(step) + '_model.pkl'))


	def load_model(self, deformator , shift_predictor, deformator_opt,shift_predictor_opt):
		models_dir = os.path.dirname(os.getcwd()) + f'/pretrained_models/18000_model.pkl'  # project root
		checkpoint = torch.load(models_dir)
		deformator.load_state_dict(checkpoint['deformator'])
		shift_predictor.load_state_dict(checkpoint['shift_predictor'])
		deformator_opt.load_state_dict(checkpoint['deformator_opt'])
		shift_predictor_opt.load_state_dict(checkpoint['shift_predictor_opt'])
		return deformator ,shift_predictor,deformator_opt,shift_predictor_opt

	def save_results(self, results, filename):
		file_location = os.path.dirname(os.getcwd()) + f'/results/{self.experiment_name}' + '/experimental_results/'
		if not os.path.exists(file_location):
			os.makedirs(file_location)
		path = file_location + str(filename) + '.pkl'
		torch.save(results, path)

	def load_results(self, filename):
		file_location = os.path.dirname(os.getcwd()) + f'/results/{self.experiment_name}' + '/experimental_results/'
		path = file_location + str(filename) + '.pkl'
		results = torch.load(path)
		return results
