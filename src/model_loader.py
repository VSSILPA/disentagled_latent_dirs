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
from models.ganspace.utils import load_gs_deformator as load_gs_deformator
from ganspace.notebooks.notebook_init import *

out_root = Path('out/figures/steerability_comp')
makedirs(out_root, exist_ok=True)
rand = lambda : np.random.randint(np.iinfo(np.int32).max)


def get_model(opt):
    if opt.algo.ours.initialisation == 'closed_form':
        generator = load_cf_generator(opt)
        deformator = load_cf_deformator(opt)  ##TODO first eigenvectos normalisation check

    elif opt.algo.ours.initialisation == 'latent_discovery':
        generator = load_ld_generator(opt)
        deformator = load_ld_deformator(opt)

    elif opt.algo.ours.initialisation == 'ganspace':
        inst = get_instrumented_model('StyleGAN', 'celebahq', 'g_mapping', device, use_w=True, inst=None)
        generator = inst.model

        pc_config = Config(components=128, n=1_000_000, use_w=True,
                           layer='g_mapping', model=opt.gan_type, output_class='celebahq')
        # dump_name = get_or_compute(pc_config, inst)
        dump_name = '/home/adarsh/PycharmProjects/disentagled_latent_dirs/src/ganspace/cache/components/stylegan-celebahq_g_mapping_ipca_c128_n1000000_w.npz'
        print(dump_name)
        with np.load(dump_name) as data:
            lat_comp = data['lat_comp']
            lat_mean = data['lat_mean']
        deformator = load_gs_deformator(opt)
        d_ours_pose, d_ours_smile, d_ours_gender, d_ours_glasses = lat_comp[7], lat_comp[14], lat_comp[1], lat_comp[5]


    deformator_opt = torch.optim.Adam(deformator.parameters(), lr=opt.algo.ours.deformator_lr)
    rank_predictor = ResNetRankPredictor(num_dirs=opt.algo.ours.num_directions).cuda()
    rank_predictor_opt = torch.optim.Adam(rank_predictor.parameters(), lr=opt.algo.ours.rank_predictor_lr)

    return generator, deformator, deformator_opt, rank_predictor, rank_predictor_opt
