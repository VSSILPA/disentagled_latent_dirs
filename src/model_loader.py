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
from models.latent_deformator import LatentDeformator
from models.proggan_sefa import PGGANGenerator
from models.cr_discriminator import ResNetRankPredictor, IdentityPredictor
from models.gan_load import make_style_gan2
# from models.gan_load import make_proggan

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
    if gan_type == 'StyleGAN2-Natural':
        G = make_style_gan2(opt.gan_resolution, opt.pretrained_gen_root, opt.w_shift)

    elif gan_type == 'prog-gan':
        G = PGGANGenerator(resolution=opt.gan_resolution)
        checkpoint = torch.load(opt.pretrained_gen_root, map_location='cpu')
        if 'generator_smooth' in checkpoint:
            G.load_state_dict(checkpoint['generator_smooth'])
        else:
            G.load_state_dict(checkpoint['generator'])
    else:
        raise NotImplementedError
    G = G.cuda()
    G.eval()

    if opt.algorithm == 'CF':
        return G
    elif opt.algorithm == 'GS':
        return G
    elif opt.algorithm == 'ours-natural' or 'ours-synthetic':
        deformator = LatentDeformator(shift_dim=G.z_space_dim,
                                      input_dim=opt.algo.ours.num_directions,  # dimension of one-hot encoded vector
                                      out_dim=G.z_space_dim,
                                      type=opt.algo.ours.deformator_type,
                                      random_init=opt.algo.ours.deformator_randint,bias=True).to(device)
        deformator_opt = torch.optim.Adam(deformator.parameters(), lr=opt.algo.ours.deformator_lr)
        cr_discriminator = ResNetRankPredictor(deformator.input_dim, opt.algo.ours.shift_predictor_size,
                                               channels=1 if opt.dataset == 'dsprites' else 3,
                                               num_dirs=opt.algo.ours.num_directions).to(device)
        identity_discriminator = IdentityPredictor()
        # identity_discriminator.load_state_dict(torch.load(opt.classifier_pretrained_path)['cr_discriminator'])
        # identity_discriminator.cuda()
        # identity_discriminator.eval()
        cr_optimizer = torch.optim.Adam(cr_discriminator.parameters(), lr=opt.algo.ours.shift_predictor_lr)
        return G, deformator, deformator_opt, cr_discriminator, cr_optimizer, identity_discriminator
    else:
        raise NotImplementedError
    return models
