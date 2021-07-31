import os
import torch
import numpy as np
import random


class Saver(object):
    def __init__(self, config):
        self.config = config
        self.experiment_name = self.config['experiment_name']
        self.model_name = self.config['file_name']

    def save_model(self, params, step):
        cwd = os.path.dirname(os.getcwd()) + f'/results/{self.experiment_name}'  # project root
        models_dir = cwd + '/models/'

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        deformator, deformator_opt, rank_predictor, rank_predictor_opt = params
        torch.save({
            'step': step,
            'deformator': deformator.state_dict(),
            'rank_predictor': rank_predictor.state_dict(),
            'deformator_opt': deformator_opt.state_dict(),
            'rank_predictor_opt': rank_predictor_opt.state_dict(),
            'torch_rng_state': torch.get_rng_state(),
            'np_rng_state': np.random.get_state(),
            'random_state': random.getstate()

        }, os.path.join(models_dir, str(step) + '_model.pkl'))
        return True

    def load_model(self, params):
        models_dir = os.path.dirname(os.getcwd()) + f'/results/{self.experiment_name}' + '/models/' + self.config['file_name']  # project root
        checkpoint = torch.load(models_dir)

        deformator, deformator_opt, rank_predictor, rank_predictor_opt = params
        deformator.load_state_dict(checkpoint['deformator'])
        rank_predictor.load_state_dict(checkpoint['rank_predictor'])
        deformator_opt.load_state_dict(checkpoint['deformator_opt'])
        rank_predictor_opt.load_state_dict(checkpoint['rank_predictor_opt'])
        torch.set_rng_state(checkpoint['torch_rng_state'])
        np.random.set_state(checkpoint['np_rng_state'])
        random.setstate(checkpoint['random_state'])
        return deformator, rank_predictor, deformator_opt, rank_predictor_opt
