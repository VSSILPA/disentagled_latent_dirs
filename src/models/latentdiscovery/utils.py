import os
import torch
from .gan_load import make_big_gan, make_proggan, make_sngan
from .latent_deformator import LatentDeformator

GEN_CHECKPOINT_DIR = '../pretrained_models/generators/LatentDiscovery'
DEFORMATOR_CHECKPOINT_DIR = '../pretrained_models/deformators/LatentDiscovery'


def load_generator(opt):
    model_name = opt.algo.ours.model_name
    gan_type = opt.gan_type
    G_weights = os.path.join(GEN_CHECKPOINT_DIR, model_name + '.pkl')
    if gan_type == 'BigGAN':
        G = make_big_gan(G_weights, [239]).eval()  ##TODO 239 class
    elif gan_type in ['ProgGAN']:
        G = make_proggan(G_weights)
    elif 'StyleGAN2' in gan_type:
        from gan_load import make_style_gan2
        G = make_style_gan2(1024, G_weights, True)
    else:
        G = make_sngan(G_weights)

    G.cuda().eval()
    return G

def load_deformator(opt):
    model_name = opt.algo.ours.model_name
    deformator = LatentDeformator(shift_dim=opt.algo.ours.latent_dim,
                                  input_dim=opt.algo.ours.num_directions,
                                  out_dim=opt.latent_dim,
                                  type=opt.algo.ours.deformator_type,
                                  random_init=True).cuda()
    deformator.load_state_dict(torch.load(os.path.join(DEFORMATOR_CHECKPOINT_DIR, model_name, 'deformator_0.pt'), map_location=torch.device('cpu')))
    deformator.cuda()
    return deformator