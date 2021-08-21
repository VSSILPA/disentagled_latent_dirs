import os
import torch
from .gan_load import make_big_gan, make_proggan, make_sngan
from models.closedform.closedform_directions import CfLinear,CfOrtho, CfProjection

GEN_CHECKPOINT_DIR = '../pretrained_models/generators/LatentDiscovery'
DEFORMATOR_CHECKPOINT_DIR = '../pretrained_models/deformators/LatentDiscovery'


def load_generator(opt):
    model_name = opt.algo.ours.model_name
    gan_type = opt.gan_type
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

def load_deformator(opt):
    model_name = opt.algo.ours.model_name
    deformator_pretrained_weights = torch.load(os.path.join(DEFORMATOR_CHECKPOINT_DIR, model_name, 'deformator_0.pt'), map_location=torch.device('cpu'))
    deformator_pretrained_weights['linear.weight'] = deformator_pretrained_weights['linear.weight'][:,:200]
    if opt.algo.ours.deformator_type == 'linear':
        deformator = CfLinear(opt.algo.ours.num_directions, opt.algo.ours.latent_dim, bias=True)
        deformator.load_state_dict(deformator_pretrained_weights)
    elif opt.algo.ours.deformator_type == 'ortho':
        deformator = CfOrtho(opt.algo.ours.num_directions, opt.algo.ours.latent_dim)
        deformator.ortho_mat.data = deformator_pretrained_weights['linear.weight']
    elif opt.algo.ours.deformator_type == 'projection':
        deformator = CfProjection(opt.algo.ours.num_directions, opt.algo.ours.latent_dim,bias=True)
        deformator.load_state_dict(deformator_pretrained_weights)
    deformator.cuda()
    return deformator