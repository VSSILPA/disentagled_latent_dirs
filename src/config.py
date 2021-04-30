"""
-------------------------------------------------
   File Name:    config.py
   Author:       Adarsh k
   Date:         2021/04/25
   Description:  Modified from:
                 https://github.com/kadarsh22/Disentanglementlibpytorch
-------------------------------------------------
"""

import argparse
import os
import sys
from yacs.config import CfgNode as CN
from contextlib import redirect_stdout

test_mode = True
if test_mode:
    experiment_name = 'test'
    experiment_description = 'test'
else:
    experiment_name = input("Enter experiment name ")
    experiment_description = 'test'
    if experiment_name == '':
        print('enter valid experiment name')
        sys.exit()
    else:
        experiment_description = input("Enter description of experiment ")
    if experiment_description == '':
        print('enter proper description')
        sys.exit()

# ---------------------------------------------------------------------------- #
# Options for experiment identifier
# ---------------------------------------------------------------------------- #

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default=experiment_name)
parser.add_argument('--experiment_description', type=str, default=experiment_description)
parser.add_argument('--evaluation', type=bool, default=False, help='wether to run in evaluation mode or not')
parser.add_argument('--file_name', type=str, default='45_vae.pkl', help='name of the model to be loaded')

# ---------------------------------------------------------------------------- #
# Options for General settings
# ---------------------------------------------------------------------------- #
cfg = CN()
cfg.gan_type = 'StyleGAN'  # choices=['BigGAN', 'ProgGAN', 'StyleGAN', 'SNGAN']
cfg.algorithm = 'LD'
cfg.dataset = 'dsprites'  # choices=['dsprites', 'mpi3d', 'cars3d','anime_face', 'shapes3d']
cfg.logging_freq = 10
cfg.saving_freq = 5
cfg.device = 'cuda:'
cfg.device_id = '0'
cfg.random_seed = 2
cfg.num_steps = int(1e+5)
cfg.batch_size = 128

BB_KWARGS = {
    "3dshapes": {"in_channel": 3, "size": 64},
    "mpi3d": {"in_channel": 3, "size": 64},
    # grayscale -> rgb
    "dsprites": {"in_channel": 1, "size": 64},
    "cars": {"in_channel": 3, "size": 64, "f_size": 512},
    "isaac": {"in_channel": 3, "size": 128, "f_size": 512},
}

# ---------------------------------------------------------------------------- #
# Options for Latent Discovery
# ---------------------------------------------------------------------------- #
cfg.algo = CN()
cfg.algo.ld = CN()
cfg.algo.ld.max_latent_dim = 64
cfg.algo.ld.num_interpretable_dir = 64
cfg.algo.ld.deformator_lr = 0.0001
cfg.algo.ld.shift_predictor_lr = 0.0001
cfg.algo.ld.beta1 = 0.9
cfg.algo.ld.beta2 = 0.999
cfg.algo.ld.deformator_randint = True
cfg.algo.ld.deformator_type = 'proj'  # choices=['fc', 'linear', 'id', 'ortho', 'proj', 'random']
cfg.algo.ld.shift_predictor = 'LeNet'  # choices=['ResNet', 'LeNet']
cfg.algo.ld.shift_distribution = 'uniform'  # choices=['normal', 'uniform']
cfg.algo.ld.shift_scale = 6
cfg.algo.ld.min_shift = 0.5
cfg.algo.ld.directions_count = 64
cfg.algo.ld.label_weight = 1.0
cfg.algo.ld.shift_weight = 0.25
cfg.algo.ld.truncation = True

# ---------------------------------------------------------------------------- #
# Options for StyleGAN
# ---------------------------------------------------------------------------- #

cfg.output_dir = ''
cfg.structure = 'linear'
cfg.loss = "logistic"
cfg.drift = 0.001
cfg.d_repeats = 1
cfg.use_ema = False
cfg.ema_decay = 0.999
cfg.alpha = 1
cfg.depth = 4

# ---------------------------------------------------------------------------- #
# Options for Generator
# ---------------------------------------------------------------------------- #
cfg.model = CN()
cfg.model.gen = CN()
cfg.model.gen.latent_size = 512
# 8 in original paper
cfg.model.gen.mapping_layers = 4
cfg.model.gen.blur_filter = [1, 2, 1]
cfg.model.gen.truncation_psi = 0.7
cfg.model.gen.truncation_cutoff = 8

# ---------------------------------------------------------------------------- #
# Options for Discriminator
# ---------------------------------------------------------------------------- #
cfg.model.dis = CN()
cfg.model.dis.use_wscale = True
cfg.model.dis.blur_filter = [1, 2, 1]

# ---------------------------------------------------------------------------- #
# Options for Generator Optimizer
# ---------------------------------------------------------------------------- #
cfg.model.g_optim = CN()
cfg.model.g_optim.learning_rate = 0.003
cfg.model.g_optim.beta_1 = 0
cfg.model.g_optim.beta_2 = 0.99
cfg.model.g_optim.eps = 1e-8

# ---------------------------------------------------------------------------- #
# Options for Discriminator Optimizer
# ---------------------------------------------------------------------------- #
cfg.model.d_optim = CN()
cfg.model.d_optim.learning_rate = 0.003
cfg.model.d_optim.beta_1 = 0
cfg.model.d_optim.beta_2 = 0.99
cfg.model.d_optim.eps = 1e-8

# ---------------------------------------------------------------------------- #
# Options for Encoder
# ---------------------------------------------------------------------------- #

cfg.encoder = CN()
cfg.encoder.num_samples = 10000
cfg.encoder.num_batches = 50
cfg.encoder.generator_bs = 50
cfg.encoder.batch_size = 128
cfg.encoder.num_directions = 5
cfg.encoder.root = 'generated_data'
cfg.encoder.latent_train_size = 500000
cfg.encoder.latent_nb_epochs = 20
cfg.encoder.latent_lr = 1e-3
cfg.encoder.latent_step_size = 10
cfg.encoder.latent_gamma = 0.5


def get_config(inputs):
    config = parser.parse_args(inputs)
    print(cfg)
    return config.__dict__, cfg


def save_config(config, opt):
    exp_name = config['experiment_name']
    cwd = os.path.dirname(os.getcwd()) + f'/results/{exp_name}'  # project root
    opt.result_dir = cwd
    models_dir = cwd + '/models'  # models directory
    visualisations_dir = cwd + '/visualisations'  # directory in which images and plots are saved
    os.makedirs(cwd, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(visualisations_dir, exist_ok=True)

    with open(f'{cwd}/config.yml', 'w') as f:
        with redirect_stdout(f):
            print(opt.dump())

    return


def str2bool(v):
    return v.lower() in ('true', '1')
