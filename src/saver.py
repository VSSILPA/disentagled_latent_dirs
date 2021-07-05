import os
import torch
import numpy as np
import random


class Saver(object):
    def __init__(self, config):
        self.config = config
        self.experiment_name = self.config['experiment_name']
        self.model_name = self.config['file_name']

    def save_model(self, params, step,generator_idx,algo='LD'):
        cwd = os.path.dirname(os.getcwd()) + f'/results/{self.experiment_name}'  # project root
        models_dir = cwd + '/models/'

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        if algo == 'LD':
            deformator, shift_predictor, deformator_opt, shift_predictor_opt = params
            torch.save({
                'step': step,
                'deformator': deformator.state_dict(),
                'shift_predictor': shift_predictor.state_dict(),
                'deformator_opt': deformator_opt.state_dict(),
                'shift_predictor_opt': shift_predictor_opt.state_dict(),
                'torch_rng_state': torch.get_rng_state(),
                'np_rng_state': np.random.get_state(),
                'random_state': random.getstate()

            }, os.path.join(models_dir, str(step) + str(generator_idx) + '_model.pkl'))
        elif algo == 'ours-natural':
                deformator, cr_discriminator, deformator_opt, cr_optimizer = params
                torch.save({
                    'step': step,
                    'deformator': deformator.state_dict(),
                    'cr_discriminator': cr_discriminator.state_dict(),
                    'deformator_opt': deformator_opt.state_dict(),
                    'cr_optimizer': cr_optimizer.state_dict(),
                    'torch_rng_state': torch.get_rng_state(),
                    'np_rng_state': np.random.get_state(),
                    'random_state': random.getstate()

                }, os.path.join(models_dir, str(step) + str(generator_idx) + '_model.pkl'))
        else:
            raise NotImplementedError

    def load_model(self, params, algo='LD'):
        models_dir = os.path.dirname(os.getcwd()) + f'/pretrained_models/' + self.config['file_name']  # project root
        checkpoint = torch.load(models_dir)
        print(algo)
        if algo == 'LD':
            deformator, shift_predictor, deformator_opt, shift_predictor_opt = params
            deformator.load_state_dict(checkpoint['deformator'])
            shift_predictor.load_state_dict(checkpoint['shift_predictor'])
            deformator_opt.load_state_dict(checkpoint['deformator_opt'])
            shift_predictor_opt.load_state_dict(checkpoint['shift_predictor_opt'])
            torch.set_rng_state(checkpoint['torch_rng_state'])
            np.random.set_state(checkpoint['np_rng_state'])
            random.setstate(checkpoint['random_state'])

            return deformator, shift_predictor, deformator_opt, shift_predictor_opt , checkpoint['step']
        elif algo == 'ours-natural':
                deformator, cr_discriminator, deformator_opt, cr_optimizer = params
                torch.save({
                    'step': step,
                    'deformator': deformator.state_dict(),
                    'cr_discriminator': cr_discriminator.state_dict(),
                    'deformator_opt': deformator_opt.state_dict(),
                    'cr_optimizer': cr_optimizer.state_dict(),
                    'torch_rng_state': torch.get_rng_state(),
                    'np_rng_state': np.random.get_state(),
                    'random_state': random.getstate()

                }, os.path.join(models_dir, str(step) + str(generator_idx) + '_model.pkl'))
        else:
            raise NotImplementedError

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
