import sys
sys.path.insert(0, './models/')
from utils import *
from models.gan_load import make_big_gan, make_proggan, make_gan, make_style_gan2
from models.latent_deformator import LatentDeformator
from models.latent_shift_predictor import LeNetShiftPredictor, ResNetShiftPredictor


def get_model(config):
    device = torch.device('cuda:' + str(config['device_id']))

    ## Loading generator
    gan_type = config['gan_type']
    if gan_type == 'BigGAN':
        G_weights = 'models/pretrained/generators/BigGAN/G_ema.pth',
        G = make_big_gan(G_weights, config['target_class']).eval()
        print_network(G.big_gan)
    elif gan_type == 'ProgGAN':
        G_weights = 'models/pretrained/generators/ProgGAN/100_celeb_hq_network-snapshot-010403.pth'
        G = make_proggan(G_weights)
    elif gan_type == 'StyleGAN2':
        G_weights =  'models/pretrained/StyleGAN2/stylegan2-car-config-f.pt',
        G = make_style_gan2(config['gan_resolution'], G_weights, config['w_shift'])
    elif gan_type == 'SNGAN':
        G_weights = 'models/pretrained/generators/SN_MNIST'
        G = make_gan(G_weights)
    else:
        raise NotImplementedError

    ## Loading Deformator
    deformator = LatentDeformator(shift_dim=G.dim_shift,
                                  input_dim=config['num_interpretable_dir'],
                                  out_dim=G.dim_z,
                                  type=config['deformator_type'],
                                  random_init=config['deformator_random_init']).to(device)
    if config['shift_predictor'] == 'ResNet':
        shift_predictor = ResNetShiftPredictor(deformator.input_dim, config['shift_predictor_size']).to(device)
    elif config['shift_predictor'] == 'LeNet':
        shift_predictor = LeNetShiftPredictor(deformator.input_dim, 1 ).to(device)

    deformator_opt = torch.optim.Adam(deformator.parameters(), lr=config['deformator_lr']) \
        if config['deformator_type'] not in ['id','random'] else None

    shift_predictor_opt = torch.optim.Adam(shift_predictor.parameters(), lr=config['shift_predictor_lr'])

    return G , deformator , shift_predictor ,deformator_opt ,shift_predictor_opt