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
    experiment_name = 'stabilsation'
    experiment_description = 'setting up working code base'
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
parser.add_argument('--evaluation', type=bool, default=False, help='whether to run in evaluation mode or not')
parser.add_argument('--file_name', type=str, default='45_vae.pkl', help='name of the model to be loaded')

# ---------------------------------------------------------------------------- #
# Options for General settings
# ---------------------------------------------------------------------------- #
opt = CN()
opt.gan_type = 'StyleGAN2'  # choices=['BigGAN', 'ProgGAN', 'StyleGAN', 'StyleGAN2','SNGAN']
opt.algorithm = 'CF'  # choices=['LD', 'CF', 'Ours', 'GS']
opt.dataset = 'mpi3d'  # choices=['dsprites', 'mpi3d', 'cars3d','anime_face', 'shapes3d','mnist','CelebA]
# opt.pretrained_gen_path = 'models/pretrained/generators/new_generators/new_generators/cars3d/0.pt'
opt.pretrained_gen_root = 'models/pretrained/generators/new_generators/new_generators/'
# opt.pretrained_gen_path = 'models/pretrained/generators/new_generators/new_generators/shapes3d/2.pt'
opt.logging_freq = 500
opt.saving_freq = 500
opt.device = 'cuda:'
opt.device_id = '0'
opt.num_seeds = 2
opt.random_seed = 2
opt.num_steps = int(1e+5)
opt.batch_size = 128

# ---------------------------------------------------------------------------- #
# Options for Latent Discovery
# ---------------------------------------------------------------------------- #
opt.algo = CN()
opt.algo.ld = CN()
opt.algo.ld.latent_dim = 512
opt.algo.ld.directions_count = 64
opt.algo.ld.shift_scale = 6
opt.algo.ld.min_shift = 0.5
opt.algo.ld.deformator_lr = 0.0001
opt.algo.ld.shift_predictor_lr = 0.0001
opt.algo.ld.beta1 = 0.9
opt.algo.ld.beta2 = 0.999
opt.algo.ld.deformator_randint = True
opt.algo.ld.deformator_type = 'proj'  # choices=['fc', 'linear', 'id', 'ortho', 'proj', 'random']
opt.algo.ld.shift_predictor = 'LeNet'  # choices=['ResNet', 'LeNet']
opt.algo.ld.shift_distribution = 'uniform'  # choices=['normal', 'uniform']
opt.algo.ld.shift_predictor_size = None  # reconstructor resolution
opt.algo.ld.label_weight = 1.0
opt.algo.ld.shift_weight = 0.25
opt.algo.ld.truncation = None

# ---------------------------------------------------------------------------- #
# Options for Latent Discovery
# ---------------------------------------------------------------------------- #
opt.algo.cf = CN()
opt.algo.cf.topk = 10

# ---------------------------------------------------------------------------- #
# Options for StyleGAN
# ---------------------------------------------------------------------------- #

opt.structure = 'linear'
opt.loss = "logistic"
opt.drift = 0.001
opt.d_repeats = 1
opt.use_ema = False
opt.ema_decay = 0.999
opt.alpha = 1
opt.depth = 4

# ---------------------------------------------------------------------------- #
# Options for StyleGAN2
# ---------------------------------------------------------------------------- #
generator_kwargs = {
    "input_is_latent": True,
    "randomize_noise": False,
    "truncation": 0.8}

# ---------------------------------------------------------------------------- #
# Options for Generator
# ---------------------------------------------------------------------------- #
opt.model = CN()
opt.model.gen = CN()
opt.model.gen.latent_size = 512
# 8 in original paper
opt.model.gen.mapping_layers = 4
opt.model.gen.blur_filter = [1, 2, 1]
opt.model.gen.truncation_psi = 0.7
opt.model.gen.truncation_cutoff = 8

# ---------------------------------------------------------------------------- #
# Options for Discriminator
# ---------------------------------------------------------------------------- #
opt.model.dis = CN()
opt.model.dis.use_wscale = True
opt.model.dis.blur_filter = [1, 2, 1]

# ---------------------------------------------------------------------------- #
# Options for Generator Optimizer
# ---------------------------------------------------------------------------- #
opt.model.g_optim = CN()
opt.model.g_optim.learning_rate = 0.003
opt.model.g_optim.beta_1 = 0
opt.model.g_optim.beta_2 = 0.99
opt.model.g_optim.eps = 1e-8

# ---------------------------------------------------------------------------- #
# Options for Discriminator Optimizer
# ---------------------------------------------------------------------------- #
opt.model.d_optim = CN()
opt.model.d_optim.learning_rate = 0.003
opt.model.d_optim.beta_1 = 0
opt.model.d_optim.beta_2 = 0.99
opt.model.d_optim.eps = 1e-8

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
    assert opt.model.gen.latent_size == opt.algo.ld.latent_dim
    assert opt.encoder.latent_dimension == opt.algo.ld.directions_count


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
