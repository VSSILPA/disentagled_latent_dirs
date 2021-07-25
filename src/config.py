
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
import logging
from yacs.config import CfgNode as CN
from contextlib import redirect_stdout

test_mode = True
if test_mode:
    experiment_name = 'CelebA HQ Ours with identity loss - hyperparameter tuning'
    experiment_description = 'Testing if identity loss is working'
else:
    experiment_name = input("Enter experiment name ")
    experiment_description = None
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

# ---------------------------------------------------------------------------- #
# Options for General settings
# ---------------------------------------------------------------------------- #
parser.add_argument('--evaluation', type=bool, default=False, help='whether to run in evaluation mode or not')
parser.add_argument('--file_name', type=str, default=None, help='name of the model to be loaded')
parser.add_argument('--resume_train', type=bool, default=False, help='name of the model to be loaded')

opt = CN()
opt.gan_type = 'prog-gan'  # choices=['BigGAN', 'ProgGAN', 'StyleGAN2','SNGAN']
opt.algorithm = 'ours-natural'  # choices=['LD', 'CF', 'linear_combo', 'GS', 'ours']
opt.dataset = 'CelebAHQ'  # choices=['dsprites', 'mpi3d', 'cars3d','shapes3d','anime_face','mnist','CelebA]
opt.gan_resolution = 1024
opt.num_channels = 3
opt.w_shift = True
opt.pretrained_gen_root = '/home/ubuntu/src/disentagled_latent_dirs/src/models/pretrained/ProgGAN/pggan_celebahq1024.pth'
opt.classifier_pretrained_path = '../pretrained_models/best_identity_classifier.pkl'
opt.device = 'cuda:'
opt.device_id = '0'
opt.random_seed = 123

# ---------------------------------------------------------------------------- #
# Options for Ours
# ---------------------------------------------------------------------------- #
opt.algo.ours = CN()
opt.algo.ours.initialisation = 'cf'
opt.algo.ours.num_steps = 140001
opt.algo.ours.batch_size = 6
opt.algo.ours.deformator_type = 'ortho'
opt.algo.ours.deformator_randint = True
opt.algo.ours.deformator_lr = 0.0001
opt.algo.ours.num_directions = 512
opt.algo.ours.latent_dim = 512
opt.algo.ours.shift_predictor_size = None
opt.algo.ours.logging_freq = 2000
opt.algo.ours.saving_freq = 2000
opt.algo.ours.shift_predictor_lr = 0.0001
opt.algo.ours.ranking_weight = 1
opt.algo.ours.identity_weight = 0.5

# ---------------------------------------------------------------------------- #
# Options for Closed form
# ---------------------------------------------------------------------------- #
opt.algo.cf = CN()
opt.algo.cf.num_directions = 10

# ---------------------------------------------------------------------------- #
# Options for Gan space
# ---------------------------------------------------------------------------- #
opt.algo.gs = CN()
opt.algo.gs.num_directions = 10
opt.algo.gs.num_samples = 20000

# ---------------------------------------------------------------------------- #
# Options for StyleGAN2
# ---------------------------------------------------------------------------- #
generator_kwargs = {
    "input_is_latent": True,
    "randomize_noise": False,
    "truncation": 1} ##todo changed from 0.8 to 1

# ---------------------------------------------------------------------------- #
# Options for Encoder
# ---------------------------------------------------------------------------- #

opt.encoder = CN()
opt.encoder.num_samples = 10000
opt.encoder.latent_dimension = 10  # this is the number of directions (w)(1*512)*(A)(512*64) == (1*64)
opt.encoder.generator_bs = 50
opt.encoder.batch_size = 128
opt.encoder.root = 'generated_data'
opt.encoder.latent_train_size = 500000
opt.encoder.latent_nb_epochs = 20
opt.encoder.latent_lr = 0.001
opt.encoder.latent_step_size = 10
opt.encoder.latent_gamma = 0.5
opt.encoder.create_new_data = True

# ---------------------------------------------------------------------------- #
# Options for Encoder Backbone
# ---------------------------------------------------------------------------- #
BB_KWARGS = {
    "shapes3d": {"in_channel": 3, "size": 64},
    "mpi3d": {"in_channel": 3, "size": 64},
    # grayscale -> rgb
    "CelebAHQ": {"in_channel": 3, "size": 64},##TODO
    "dsprites": {"in_channel": 1, "size": 64},
    "cars3d": {"in_channel": 3, "size": 64, "f_size": 512},
    "isaac": {"in_channel": 3, "size": 128, "f_size": 512},
}
if opt.algorithm == 'LD':
    assert opt.encoder.latent_dimension == opt.algo.ld.num_directions


def get_config(inputs):
    config = parser.parse_args(inputs)
    print(opt)
    return config.__dict__, opt


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
