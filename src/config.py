import argparse
import json
import os
import sys

experiment_name = input("Enter experiment name ")
if experiment_name == '':
    print('enter valid experiment name')
    sys.exit()

experiment_description = input("Enter description of experiment ")
if experiment_description == '':
    print('enter proper description')
    sys.exit()

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default=experiment_name)
parser.add_argument('--experiment_description', type=str, default=experiment_description)

# general configuration
parser.add_argument('--gan_type', type=str, default='SNGAN', choices=['BigGAN', 'ProgGAN', 'StyleGAN2','SNGAN'],
                    help='architecture of model')
parser.add_argument('--dataset', type=str, default='Mnist', choices=['celebA', 'Mnist', 'ImageNet',
                                                                        'anime_face','dsprites'], help='name of the dataset')
parser.add_argument('--logging_freq', type=int, default=10, help='Frequency at which result  should be logged')
parser.add_argument('--saving_freq', type=int, default=2000, help='Frequency at which result  should be logged')
parser.add_argument('--evaluation', type=bool, default=False, help='whether to run in evaluation mode or not')
parser.add_argument('--file_name', type=str, default='25_gan.pkl', help='name of the model to be loaded')
parser.add_argument('--device_id', type=int, default=0, help='Device id of gpu')
parser.add_argument('--random_seed', type=int, default=2, help='Random seeds to run for ')


# GAN configurations
parser.add_argument('--num_steps', type=int, default=int(1e+5), help='The number of epochs to run')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--max_latent_dim', type=int, default=64, help='max number of latent dimensions')
parser.add_argument('--num_interpretable_dir', type=int, default=64, help='number of interpretable directions')
parser.add_argument('--deformator_lr', type=float, default=0.0001, help='learning rate of deformator')
parser.add_argument('--shift_predictor_lr', type=float, default=0.0001, help='learning rate of shift_predictor')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 optimizer')
parser.add_argument('--deformator_random_init', type=bool, default=True)



parser.add_argument('--deformator_type', type=str, default='proj', choices=['fc' , 'linear','id','ortho','proj','random'])
parser.add_argument('--shift_predictor', type=str, choices=['ResNet', 'LeNet'], default='LeNet')
parser.add_argument('--shift_distribution', type=str, choices=['normal','uniform'],default='uniform')
parser.add_argument('--shift_scale', type=int,default=6)
parser.add_argument('--min_shift', type=int,default=0.5)
parser.add_argument('--directions_count', type=int, default=64, help='number of directions')
parser.add_argument('--label_weight', type=float,default=1.0)
parser.add_argument('--shift_weight', type=float,default=0.25)
parser.add_argument('--truncation', type=int,default=None)

## Model specfic parameters
parser.add_argument('--w_shift', type=bool, default=True,
                    help='latent directions search in w-space for StyleGAN2')
parser.add_argument('--gan_resolution', type=int, default=1024,
                    help='generator out images resolution. Required only for StyleGAN2')

def get_config(inputs):
    config = parser.parse_args(inputs)
    return config.__dict__


def save_config(config):
    exp_name = config['experiment_name']
    cwd = os.path.dirname(os.getcwd()) + f'/results/{exp_name}'  # project root
    models_dir = cwd + '/models'  # models directory
    visualisations_dir = cwd + '/visualisations'  # directory in which images and plots are saved
    os.makedirs(cwd, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(visualisations_dir, exist_ok=True)
    with open(f'{cwd}/config.json', 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)
    return


def str2bool(v):
    return v.lower() in ('true', '1')
