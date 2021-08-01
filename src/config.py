
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
    experiment_name = 'mpi3d with best setting and inequality reversed--linear'
    experiment_description = 'studying effect of flipping on best setting'
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
parser.add_argument('--evaluation', type=bool, default=False, help='whether to run in evaluation mode or not')
parser.add_argument('--file_name', type=str, default='40005_model.pkl', help='name of the model to be loaded')
parser.add_argument('--resume_train', type=bool, default=False, help='name of the model to be loaded')
opt = CN()
opt.gan_type = 'StyleGAN2'  # choices=['BigGAN', 'ProgGAN', 'StyleGAN', 'StyleGAN2','SNGAN']
opt.algorithm = 'ours'  # choices=['LD', 'CF', 'linear_combo', 'GS', 'ours']
opt.dataset = 'mpi3d'  # choices=['dsprites', 'mpi3d', 'cars3d','shapes3d','anime_face','mnist','CelebA]
#opt.pretrained_gen_root = 'models/pretrained/generators/new_generators/new_generators/'
opt.pretrained_gen_root = 'models/pretrained/new_generators/'
opt.num_channels = 3 if opt.dataset != 'dsprites' else 1
opt.device = 'cuda:'
opt.device_id = '0'
opt.num_generator_seeds = 5 if opt.dataset != 'cars3d' else 7
opt.random_seed =2
if opt.dataset == 'dsprites':
    opt.num_generator_seeds = 1

# ---------------------------------------------------------------------------- #
# Options for Latent Discovery
# ---------------------------------------------------------------------------- #
opt.algo = CN()
opt.algo.ld = CN()
opt.algo.ld.batch_size = 32
opt.algo.ld.latent_dim = 512
opt.algo.ld.num_steps = 5001
opt.algo.ld.num_directions = 10
opt.algo.ld.shift_scale = 6
opt.algo.ld.min_shift = 0.5
opt.algo.ld.deformator_lr = 0.0001
opt.algo.ld.shift_predictor_lr = 0.0001
opt.algo.ld.beta1 = 0.9
opt.algo.ld.beta2 = 0.999
opt.algo.ld.deformator_randint = True
opt.algo.ld.deformator_type = 'ortho'  # choices=['fc', 'linear', 'id', 'ortho', 'proj', 'random']
opt.algo.ld.shift_predictor = 'ResNet'  # choices=['ResNet', 'LeNet']1
opt.algo.ld.shift_distribution = 'uniform'  # choices=['normal', 'uniform']
opt.algo.ld.shift_predictor_size = None  # reconstructor resolution
opt.algo.ld.label_weight = 1.0
opt.algo.ld.shift_weight = 0.25
opt.algo.ld.truncation = None
opt.algo.ld.logging_freq = 1000
opt.algo.ld.saving_freq = 1000

# ---------------------------------------------------------------------------- #
# Options for Ours
# ---------------------------------------------------------------------------- #
opt.algo.ours = CN()
opt.algo.ours.initialisation = 'cf'
opt.algo.ours.num_steps = 6001
opt.algo.ours.batch_size = 32
opt.algo.ours.deformator_type = 'linear'
opt.algo.ours.deformator_randint = True
opt.algo.ours.deformator_lr = 0.0001
opt.algo.ours.num_directions = 10
opt.algo.ours.latent_dim = 512
opt.algo.ours.shift_predictor_size = None
opt.algo.ours.logging_freq = 6000
opt.algo.ours.saving_freq = 6000
opt.algo.ours.shift_predictor_lr = 0.0001

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
    "truncation": 0.8}

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
