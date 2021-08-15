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
import torch
from models.rank_predictor import ResNetRankPredictor
from models.closedform.utils import load_generator as load_cf_generator
from models.closedform.utils import load_deformator as load_cf_deformator
from models.latentdiscovery.utils import load_generator as load_ld_generator
from models.latentdiscovery.utils import load_deformator as load_ld_deformator


def get_model(opt):
    if opt.algo.ours.initialisation == 'closed_form':
        generator = load_cf_generator(opt)
        deformator, layers = load_cf_deformator(opt) ##TODO first eigenvectos normalisation check

    elif opt.algo.ours.initialisation == 'latent_discovery':
        layers = None
        generator = load_ld_generator(opt)
        deformator = load_ld_deformator(opt)

    deformator_opt = torch.optim.Adam(deformator.parameters(), lr=opt.algo.ours.deformator_lr)
    rank_predictor = ResNetRankPredictor(num_dirs=opt.algo.ours.num_directions).cuda()
    rank_predictor_opt = torch.optim.Adam(rank_predictor.parameters(), lr=opt.algo.ours.rank_predictor_lr)

    return generator, deformator, deformator_opt, rank_predictor, rank_predictor_opt ,layers
