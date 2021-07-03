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
    experiment_name = 'cifar10-ours-similarity-0.1 0.1'
    experiment_description = 'checking if similarity is sufficient for getting class identities'
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
parser.add_argument('--file_name', type=str, default='500_model.pkl', help='name of the model to be loaded')
parser.add_argument('--resume_train', type=bool, default=False, help='name of the model to be loaded')
opt = CN()
opt.gan_type = 'StyleGAN2-ada'  # choices=['BigGAN', 'ProgGAN', 'StyleGAN', 'StyleGAN2','SNGAN']
opt.algorithm = 'discrete_ld'  # choices=['infogan', 'discrete_ld', 'GS']
opt.dataset = 'cifar10'  # choices=[mnist,cifar10]
opt.pretrained_gen_root = 'models/pretrained/new_generators/'
# opt.pretrained_gen_root = 'models/pretrained/new_generators/'
opt.num_channels = 1 if opt.dataset == 'mnist' or 'fashion_mnist' else 3
opt.device = 'cuda:'
opt.image_size = 32
opt.num_classes = 10
opt.device_id = '0'
opt.random_seed = 2
opt.num_generator_seeds = 8
opt.num_seeds = 2

##Encoder backbone params
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
opt.algo = CN()
opt.algo.ld = CN()
opt.algo.ld.latent_dim = 512
opt.algo.ld.directions_count = 5

opt.algo.ld.deformator_lr = 0.0001
opt.algo.ld.shift_predictor_lr = 0.0001
opt.algo.ld.beta1 = 0.9
opt.algo.ld.beta2 = 0.999
opt.algo.ld.deformator_randint = True
opt.algo.ld.deformator_type = 'linear'  # choices=['fc', 'linear', 'id', 'ortho', 'proj', 'random']
opt.algo.ld.shift_predictor = 'ResNet'  # choices=['ResNet', 'LeNet']
opt.algo.ld.shift_distribution = 'uniform'  # choices=['normal', 'uniform']
opt.algo.ld.shift_predictor_size = None     #reconstructor resolution
opt.algo.ld.label_weight = 1.0
opt.algo.ld.shift_weight = 0.25
opt.algo.ld.truncation = None


# Options for StyleGAN2
# ---------------------------------------------------------------------------- #
generator_kwargs = {
    "input_is_latent": True,
    "randomize_noise": False,
    "truncation": 0.8}

# ---------------------------------------------------------------------------- #
# Options for Discrete latent discovery
# ---------------------------------------------------------------------------- #
opt.algo = CN()
opt.algo.discrete_ld = CN()
opt.algo.discrete_ld.batch_size = 16
opt.algo.discrete_ld.latent_dim = 128
opt.algo.discrete_ld.num_steps = 100001
opt.algo.discrete_ld.num_directions = 20
opt.algo.discrete_ld.deformator_lr = 0.001
opt.algo.discrete_ld.shift_predictor_lr = 0.001
opt.algo.discrete_ld.beta1 = 0.9
opt.algo.discrete_ld.beta2 = 0.999
opt.algo.discrete_ld.deformator_randint = True
opt.algo.discrete_ld.deformator_type = 'linear'  # choices=['fc', 'linear', 'id', 'ortho', 'proj', 'random']
opt.algo.discrete_ld.shift_predictor = 'ResNet'  # choices=['ResNet', 'LeNet']1
opt.algo.discrete_ld.shift_distribution = 'uniform'  # choices=['normal', 'uniform']
opt.algo.discrete_ld.shift_predictor_size = None  # reconstructor resolution
opt.algo.discrete_ld.truncation = None
opt.algo.discrete_ld.logging_freq = 1000
opt.algo.discrete_ld.saving_freq = 10000
opt.algo.discrete_ld.shift_scale = 6
opt.algo.discrete_ld.min_shift = 0.5
opt.algo.discrete_ld.label_weight = 1.0
opt.algo.discrete_ld.shift_weight = 0.25


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
