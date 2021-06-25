"""
-------------------------------------------------
   File Name:    model_loader.py
   Author:       Adarsh k
   Date:         2021/04/25
   Description:  Load a pretrained generator as specified in config as well as required models for the given algorithm
                 Reference  : https://github.com/kadarsh22/GANLatentDiscovery
                 Contains pretrained generator and latent deformator
-------------------------------------------------
"""

from utils import *
# from models.stylegan2.models import Generator
from models.latent_deformator import LatentDeformator
from models.latent_shift_predictor import LeNetShiftPredictor, ResNetShiftPredictor
from loading import load_generator
import sys
from model import Lenet28

sys.path.insert(0, './models/')
from infogan_components import Generator, Discriminator
from InfoGAN import InfoGAN


def load(model, cpk_file):
    pretrained_dict = torch.load(cpk_file)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_model(opt):
    device = torch.device(opt.device + opt.device_id)
    gan_type = opt.gan_type
    if gan_type == 'StyleGAN2':
        config_gan = {"latent": 64 if opt.dataset == 'dsprites' else 512, "n_mlp": 3,
                      "channel_multiplier": 8 if opt.dataset != 'dsprites' else 1}
        G = Generator(
            size=64,
            style_dim=config_gan["latent"],
            n_mlp=config_gan["n_mlp"],
            small=True,
            channel_multiplier=config_gan["channel_multiplier"],
        )
        G.load_state_dict(torch.load(opt.pretrained_gen_path))
        G.eval().to(device)
        for p in G.parameters():
            p.requires_grad_(False)
    elif gan_type == 'SNGAN':
        G = load_generator({'gan_type': 'SNGAN'}, 'models/pretrained/generators/SN_MNIST/')
    elif gan_type == 'InfoGAN':
        c1_len = 10  # Multinomial
        c2_len = 0  # Gaussian
        c3_len = 0  # Bernoulli
        z_len = 64  # Noise vector length
        embedding_len = 128
        infogan_gen = Generator().to(device)
        infogan_dis = Discriminator().to(device)
        G = InfoGAN(infogan_gen, infogan_dis, embedding_len, z_len, c1_len, c2_len, c3_len, device)
        G.load('/home/adarsh/PycharmProjects/disentagled_latent_dirs/src/models/InfoGAN/')
    else:
        raise NotImplementedError

    if opt.algorithm == 'discrete_ld':
        deformator = LatentDeformator(shift_dim=G.dim_z,
                                      input_dim=opt.algo.discrete_ld.num_directions,
                                      # dimension of one-hot encoded vector
                                      out_dim=G.dim_z,
                                      type=opt.algo.discrete_ld.deformator_type,
                                      random_init=opt.algo.discrete_ld.deformator_randint).to(device)
        if opt.algo.discrete_ld.shift_predictor == 'ResNet':
            shift_predictor = ResNetShiftPredictor(deformator.input_dim, opt.algo.discrete_ld.shift_predictor_size,
                                                   channels=1 if opt.dataset == 'dsprites' or 'mnist' else 3).to(device)
            # shift_predictor = Lenet28()
            # shift_predictor.load_state_dict(torch.load('4.pth'))
        elif opt.algo.discrete_ld.shift_predictor == 'LeNet':
            shift_predictor = LeNetShiftPredictor(deformator.input_dim, 1).to(device)
        else:
            raise NotImplementedError

        deformator_opt = torch.optim.Adam(deformator.parameters(), lr=opt.algo.discrete_ld.deformator_lr)

        shift_predictor_opt = torch.optim.Adam(shift_predictor.parameters(),
                                               lr=opt.algo.discrete_ld.shift_predictor_lr)
        models = (G, deformator, shift_predictor, deformator_opt, shift_predictor_opt)
    elif opt.algorithm == 'infogan':
        models = G
    else:
        raise NotImplementedError
    return models
