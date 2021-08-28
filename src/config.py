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
    experiment_name = 'lsun_cars_closed_form+ours'
    experiment_description = 'lsun cars'
else:
    experiment_name = input("Enter experiment name ")
    experiment_description = 'first run of shapes 3d for latent discovert with ortho'
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

parser.add_argument('--resume_train', type=bool, default=False, help='name of the model to be loaded')
opt = CN()
opt.gan_type = 'StyleGAN'  # choices=['BigGAN', 'ProgGAN', 'StyleGAN2','SNGAN' ,'StyleGAN']
# choices=['AnimeFaceS', 'ImageNet',CelebAHQ' ,'LSUN-cars', 'LSUN-cats' , 'LSUN-landscapes']
opt.random_seed = 123

# ---------------------------------------------------------------------------- #
# Options for Ours
# ---------------------------------------------------------------------------- #
opt.algo = CN()
opt.algo.ours = CN()
opt.algo.ours.model_name = 'stylegan_car512'  # choices = ['pggan_celebahq1024',stylegan_animeface512,stylegan_car512,stylegan_cat256]
opt.algo.ours.initialisation = 'closed_form'  # choices = ['closed_form', 'latent_discovery', 'gan_space]
opt.algo.ours.num_steps =40001
opt.algo.ours.batch_size = 2
opt.algo.ours.deformator_type = 'ortho'  # choices = ['linear','ortho']
opt.algo.ours.deformator_lr = 0.0001
opt.algo.ours.rank_predictor_lr = 0.0001
opt.algo.ours.num_directions = 512
opt.algo.ours.latent_dim = 512
opt.algo.ours.saving_freq = 2000
opt.algo.ours.logging_freq = 500
opt.algo.ours.shift_min = 3 ##TODO Hyperparameter tuning


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
