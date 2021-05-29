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
from models.gan_load import make_big_gan, make_proggan, make_gan, make_style_gan2
from models.StyleGAN.GAN import StyleGAN
from models.stylegan2.models import Generator
from models.latent_deformator import LatentDeformator
from models.latent_shift_predictor import LeNetShiftPredictor, ResNetShiftPredictor

import sys
sys.path.insert(0, './models/')


def load(model, cpk_file):
    pretrained_dict = torch.load(cpk_file)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_model(config, opt):
    device = torch.device(opt.device + opt.device_id)
    gan_type = opt.gan_type
    if gan_type == 'BigGAN':
        G_weights = 'models/pretrained/generators/BigGAN/G_ema.pth',
        G = make_big_gan(G_weights, config['target_class']).eval()
    elif gan_type == 'StyleGAN':
        style_gan = StyleGAN(structure='linear',
                             resolution=64,
                             num_channels=1,
                             latent_size=opt.model.gen.latent_size,
                             g_args=opt.model.gen,
                             d_args=opt.model.dis,
                             g_opt_args=opt.model.g_optim,
                             d_opt_args=opt.model.d_optim,
                             loss=opt.loss,
                             drift=opt.drift,
                             d_repeats=opt.d_repeats,
                             use_ema=opt.use_ema,
                             ema_decay=opt.ema_decay,
                             device=device)
        load(style_gan.gen, 'models/pretrained/stylegan_dsprites/GAN_GEN_4_9.pth')
        G = style_gan.gen
    elif gan_type == 'ProgGAN':
        G_weights = 'models/pretrained/generators/ProgGAN/100_celeb_hq_network-snapshot-010403.pth'
        G = make_proggan(G_weights)
    elif gan_type == 'StyleGAN2':
        config_gan = {"latent": 64, "n_mlp": 3, "channel_multiplier": 1}
        G = Generator(
            size=64,
            style_dim=config_gan["latent"],
            n_mlp=config_gan["n_mlp"],
            small=True,
            channel_multiplier=config_gan["channel_multiplier"],
        )
        G.load_state_dict(torch.load('/media/adarsh/DATA/checkpoints_old/200000.pt')["g"])
        G.eval().to(device)
        for p in G.parameters():
            p.requires_grad_(False)
    elif gan_type == 'SNGAN':
        G_weights = 'models/pretrained/generators/SN_MNIST'
        G = make_gan(G_weights)
    elif gan_type == 'DCGAN':
        G = Generator()
        G.load_state_dict(torch.load('/home/adarsh/PycharmProjects/disentagled_latent_dirs/src/models/pretrained/generators/new_generators/new_generators/dsprites/infogan_mig_0.22.pkl')['gen_state_dict'])
        G.eval()
        G.cuda()
        # z = torch.rand(100, 5).cuda()* 2 - 1
        # c_cond = torch.rand(100, 5).cuda() * 2 - 1
        # z = torch.cat((z, c_cond), dim=1)
        # images = G(z.cuda())
        # from torchvision.utils import save_image
        # save_image(images.detach(),
        #            '/home/adarsh/PycharmProjects/disentagled_latent_dirs/results/stabilsation' + '/visualisations/gen_images.jpeg',
        #            nrow=int(np.sqrt(len(images))), normalize=True,
        #            scale_each=True, pad_value=128, padding=1)
    else:
        raise NotImplementedError

    if opt.algorithm == 'LD':
        deformator = LatentDeformator(shift_dim=opt.model.gen.latent_size,
                                      input_dim=opt.algo.ld.directions_count,  # dimension of one-hot encoded vector
                                      out_dim=opt.model.gen.latent_size,
                                      type=opt.algo.ld.deformator_type,
                                      random_init=opt.algo.ld.deformator_randint).to(device)
        if opt.algo.ld.shift_predictor == 'ResNet':
            shift_predictor = ResNetShiftPredictor(deformator.input_dim, opt.algo.ld.shift_predictor_size).to(device)
        elif opt.algo.ld.shift_predictor == 'LeNet':
            shift_predictor = LeNetShiftPredictor(deformator.input_dim, 1).to(device)

        deformator_opt = torch.optim.Adam(deformator.parameters(), lr=opt.algo.ld.deformator_lr)

        shift_predictor_opt = torch.optim.Adam(shift_predictor.parameters(), lr=opt.algo.ld.shift_predictor_lr)
        models = (G, deformator, shift_predictor, deformator_opt, shift_predictor_opt)
    elif opt.algorithm == 'CF' or 'GS':
        return G
    else:
        raise NotImplementedError
    return models
