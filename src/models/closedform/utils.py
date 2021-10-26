import os
import torch
import subprocess
from models.closedform.model_zoo import MODEL_ZOO
from models.closedform.closedform_directions import CfLinear, CfOrtho
from models.closedform.pggan_generator import PGGANGenerator
from models.closedform.stylegan_generator import StyleGANGenerator
from models.closedform.stylegan2_generator import StyleGAN2Generator

GEN_CHECKPOINT_DIR = '../pretrained_models/generators/ClosedForm'
DEFORMATOR_CHECKPOINT_DIR = '../pretrained_models/deformators/ClosedForm'


def build_generator(gan_type, resolution, **kwargs):
    """Builds generator by GAN type.

    Args:
        gan_type: GAN type to which the generator belong.
        resolution: Synthesis resolution.
        **kwargs: Additional arguments to build the generator.

    Raises:
        ValueError: If the `gan_type` is not supported.
        NotImplementedError: If the `gan_type` is not implemented.
    """

    if gan_type == 'pggan':
        return PGGANGenerator(resolution, **kwargs)
    if gan_type == 'stylegan':
        return StyleGANGenerator(resolution, **kwargs)
    if gan_type == 'stylegan2':
        return StyleGAN2Generator(resolution, **kwargs)
    raise NotImplementedError(f'Unsupported GAN type `{gan_type}`!')


def load_generator(opt, model_name=''):
    try:
        model_name = opt.algo.ours.model_name
    except AttributeError:
        model_name = model_name
    """Loads pre-trained generator.

    Args:
        model_name: Name of the model. Should be a key in `models.MODEL_ZOO`.

    Returns:
        A generator, which is a `torch.nn.Module`, with pre-trained weights
            loaded.

    Raises:
        KeyError: If the input `model_name` is not in `models.MODEL_ZOO`.
    """

    model_config = MODEL_ZOO[model_name].copy()
    url = model_config.pop('url')  # URL to download model if needed.

    # Build generator.
    print(f'Building generator for model `{model_name}` ...')
    generator = build_generator(**model_config)
    print(f'Finish building generator.')

    # Load pre-trained weights.
    os.makedirs(GEN_CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(GEN_CHECKPOINT_DIR, model_name + '.pth')
    print(f'Loading checkpoint from `{checkpoint_path}` ...')
    if not os.path.exists(checkpoint_path):
        print(f'  Downloading checkpoint from `{url}` ...')
        subprocess.call(['wget', '--quiet', '-O', checkpoint_path, url])
        print(f'  Finish downloading checkpoint.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'generator_smooth' in checkpoint:
        generator.load_state_dict(checkpoint['generator_smooth'])
    else:
        generator.load_state_dict(checkpoint['generator'])
    generator = generator.cuda()
    generator.eval()
    print(f'Finish loading checkpoint.')
    return generator


def load_deformator(opt):
    model_name = opt.algo.ours.model_name
    deformator_type = opt.algo.ours.deformator_type
    _, directions, _ = torch.load(os.path.join(DEFORMATOR_CHECKPOINT_DIR, model_name, model_name + '.pkl'))
    directions_T = directions.T  ## Sefa returns eigenvectors as rows, so transpose required
    if deformator_type == 'linear':
        deformator = CfLinear(opt.algo.ours.latent_dim, opt.algo.ours.num_directions)
        deformator.linear.weight.data = torch.FloatTensor(directions_T)
    elif deformator_type == 'ortho':
        deformator = CfOrtho(opt.algo.ours.latent_dim, opt.algo.ours.num_directions)
        deformator.ortho_mat.data = torch.FloatTensor(directions_T)
    deformator.cuda()
    return deformator


def load_gs_deformator(opt):
    model_name = opt.algo.ours.model_name
    deformator_type = opt.algo.ours.deformator_type
    if deformator_type == 'linear':
        deformator = CfLinear(opt.algo.ours.latent_dim, opt.algo.ours.num_directions)
    elif deformator_type == 'ortho':
        deformator = CfOrtho(opt.algo.ours.latent_dim, opt.algo.ours.num_directions)
    deformator.cuda()
    return deformator
