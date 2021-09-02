import os
import torch
from .gan_load import make_big_gan, make_proggan, make_sngan
from .latent_deformator import LatentDeformator
from models.closedform.closedform_directions import CfLinear,CfOrtho, CfProjection

GEN_CHECKPOINT_DIR = '../pretrained_models/generators/LatentDiscovery'
DEFORMATOR_CHECKPOINT_DIR = '../pretrained_models/deformators/LatentDiscovery'


def load_generator(opt, model_name='', gan_type=''):
    try:
        model_name = opt.algo.ours.model_name
        gan_type = opt.gan_type
    except AttributeError:
        model_name = model_name
        gan_type = gan_type
    G_weights = os.path.join(GEN_CHECKPOINT_DIR, model_name + '.pkl')
    if gan_type == 'BigGAN':
        G = make_big_gan(G_weights, [239]).eval()  ##TODO 239 class
    elif gan_type in ['ProgGAN']:
        G_weights = os.path.join(GEN_CHECKPOINT_DIR, model_name + '.pth')
        G = make_proggan(G_weights)
    elif 'StyleGAN2' in gan_type:
        from gan_load import make_style_gan2
        G = make_style_gan2(1024, G_weights, True)
    else:
        G = make_sngan(G_weights)

    G.cuda().eval()
    return G

def load_deformator(opt, G):
    model_name = opt.algo.ours.model_name
    # directions = torch.load(os.path.join(DEFORMATOR_CHECKPOINT_DIR, model_name, 'deformator_0.pt'), map_location=torch.device('cpu'))
    checkpoint = torch.load('/home/adarsh/PycharmProjects/disentagled_latent_dirs/results/celeba_hq/latent_discovery_ours/models/22000_model.pkl')
    deformator = CfOrtho(opt.algo.ours.num_directions, opt.algo.ours.latent_dim).cuda()
    deformator.ortho_mat.data = checkpoint['deformator']['ortho_mat']
    deformator.cuda()
    return deformator
