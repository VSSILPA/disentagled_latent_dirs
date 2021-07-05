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
from models.stylegan2.models import Generator
from models.latent_deformator import LatentDeformator
from models.latent_shift_predictor import LeNetShiftPredictor, ResNetShiftPredictor
from models.cr_discriminator import ResNetRankPredictor
from models.gan_load import make_style_gan2

import sys

sys.path.insert(0, './models/')


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
    elif gan_type == 'StyleGAN2-Natural':
        G = make_style_gan2(opt.gan_resolution, opt.pretrained_gen_root, opt.w_shift)
    else:
        raise NotImplementedError

    if opt.algorithm == 'LD':
        deformator = LatentDeformator(shift_dim=G.style_dim,
                                      input_dim=opt.algo.ld.num_directions,  # dimension of one-hot encoded vector
                                      out_dim=G.style_dim,
                                      type=opt.algo.ld.deformator_type,
                                      random_init=opt.algo.ld.deformator_randint).to(device)
        if opt.algo.ld.shift_predictor == 'ResNet':
            shift_predictor = ResNetShiftPredictor(deformator.input_dim, opt.algo.ld.shift_predictor_size,
                                                   channels=1 if opt.dataset == 'dsprites' else 3).to(device)
        elif opt.algo.ld.shift_predictor == 'LeNet':
            shift_predictor = LeNetShiftPredictor(deformator.input_dim, 1).to(device)
        else:
            raise NotImplementedError

        deformator_opt = torch.optim.Adam(deformator.parameters(), lr=opt.algo.ld.deformator_lr)

        shift_predictor_opt = torch.optim.Adam(shift_predictor.parameters(), lr=opt.algo.ld.shift_predictor_lr)
        models = (G, deformator, shift_predictor, deformator_opt, shift_predictor_opt)
    elif opt.algorithm == 'CF':
        return G
    elif opt.algorithm == 'GS':
        return G
    elif opt.algorithm == 'ours-natural' or 'ours-synthetic':
        deformator = LatentDeformator(shift_dim=G.style_dim,
                                      input_dim=opt.algo.ours.num_directions,  # dimension of one-hot encoded vector
                                      out_dim=G.style_dim,
                                      type=opt.algo.ours.deformator_type,
                                      random_init=opt.algo.ours.deformator_randint).to(device)
        if opt.deformator_pretrained is not None:
            deformator.load_state_dict(torch.load(opt.deformator_pretrained))
        deformator_opt = torch.optim.Adam(deformator.parameters(), lr=opt.algo.ours.deformator_lr)
        cr_discriminator = ResNetRankPredictor(deformator.input_dim, opt.algo.ours.shift_predictor_size,
                                               channels=1 if opt.dataset == 'dsprites' else 3, num_dirs=opt.algo.ours.num_directions).to(device)
        cr_optimizer = torch.optim.Adam(cr_discriminator.parameters(), lr=opt.algo.ours.shift_predictor_lr)
        return G, deformator, deformator_opt, cr_discriminator, cr_optimizer

    elif opt.algorithm == 'linear_combo':
        deformator = LatentDeformator(shift_dim=G.style_dim,
                                      input_dim=opt.algo.linear_combo.num_directions,
                                      # dimension of one-hot encoded vector
                                      out_dim=G.style_dim,
                                      type=opt.algo.linear_combo.deformator_type,
                                      random_init=opt.algo.linear_combo.deformator_randint).to(device)
        models_dir = os.path.dirname(
            os.getcwd()) + f'/pretrained_models/' + opt.algo.linear_combo.file_name  # project root
        checkpoint = torch.load(models_dir)
        deformator.load_state_dict(checkpoint['deformator'])
        if opt.algo.linear_combo.shift_predictor == 'ResNet':
            shift_predictor = ResNetShiftPredictor(deformator.input_dim, opt.algo.linear_combo.shift_predictor_size,
                                                   channels=1 if opt.dataset == 'dsprites' else 3).to(device)
            cr_discriminator = ResNetRankPredictor(deformator.input_dim, opt.algo.linear_combo.shift_predictor_size,
                                                    channels=1 if opt.dataset == 'dsprites' else 3).to(device)
        elif opt.algo.linear_combo.shift_predictor == 'LeNet':
            shift_predictor = LeNetShiftPredictor(deformator.input_dim, 1).to(device)
        else:
            raise NotImplementedError
        shift_predictor.load_state_dict(checkpoint['shift_predictor'])

        deformator_opt = torch.optim.Adam(deformator.parameters(), lr=opt.algo.linear_combo.deformator_lr)
        cr_optimizer = torch.optim.Adam(cr_discriminator.parameters(), lr=opt.algo.linear_combo.shift_predictor_lr)

        shift_predictor_opt = torch.optim.Adam(shift_predictor.parameters(),
                                               lr=opt.algo.linear_combo.shift_predictor_lr)
        models = (G, deformator, shift_predictor, cr_discriminator, cr_optimizer, deformator_opt, shift_predictor_opt)
    else:
        raise NotImplementedError
    return models
