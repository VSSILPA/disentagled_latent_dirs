from utils import *
from torch.utils.data import DataLoader, Dataset
from models.latent_deformator import LatentDeformator
import collections
import numpy as np
import random
import json
from models.proggan_sefa import PGGANGenerator
import torch.nn.functional as F
from torchvision.models import resnet18
import torch.nn as nn


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(1234)


def get_resnet():
    net = resnet18()
    modified_net = nn.Sequential(*list(net.children())[:-1])  # fetch all of the layers before the last fc.
    return modified_net


def get_classifier(pretrain_path, device):
    classifier = ClassifyModel().to(device)
    classifier.load_state_dict(torch.load(pretrain_path))
    return classifier


class ClassifyModel(nn.Module):
    def __init__(self, n_class=2):
        super(ClassifyModel, self).__init__()
        self.backbone = get_resnet()
        self.extra_layer = nn.Linear(512, n_class)

    def forward(self, x):
        out = self.backbone(x)
        out = torch.flatten(out, 1)
        out = self.extra_layer(out)
        return out


gan_type = 'prog-gan-sefa'
num_directions = 512

visualisation_data_path = '/media/adarsh/DATA/CelebA-Analysis/'
result_path = os.path.join(visualisation_data_path, 'Closed-Form-Analysis-results')
os.makedirs(result_path, exist_ok=True)

G = PGGANGenerator(resolution=1024)
checkpoint = torch.load('models/pretrained/ProgGAN/pggan_celebahq1024.pth', map_location='cpu')
if 'generator_smooth' in checkpoint:
    G.load_state_dict(checkpoint['generator_smooth'])
else:
    G.load_state_dict(checkpoint['generator'])
G = G.cuda()
G.eval()

pretrained_model = torch.load(visualisation_data_path + 'models/cf_model.pkl', map_location='cpu')
deformator = LatentDeformator(shift_dim=G.z_space_dim, input_dim=num_directions,
                              out_dim=G.z_space_dim, type='linear', random_init=True, bias=False) ##TODO Change bias True

deformator.load_state_dict(pretrained_model['deformator'])
deformator.cuda()
deformator.eval()


class NoiseDataset(Dataset):
    def __init__(self, num_samples, z_dim):
        self.num_samples = num_samples
        self.z_dim = z_dim
        self.data = torch.randn(num_samples, z_dim)

    def __getitem__(self, index):
        x = self.data[index]
        return x

    def __len__(self):
        return len(self.data)


num_samples = 2000
batch_size = 5
num_batches = int(num_samples / batch_size)
z = NoiseDataset(num_samples=num_samples, z_dim=G.z_space_dim)
z_loader = DataLoader(z, batch_size=batch_size, shuffle=False)
torch.save(z, os.path.join(result_path, 'z_analysis.pkl'))
shift = 10
attr_list = ['pose', 'eyeglasses', 'male', 'pose', 'smiling', 'young']
attr_var_dict = collections.OrderedDict()
for attr_selected in attr_list:
    print('Attribute : ', attr_selected)
    predictor = get_classifier(
        os.path.join(visualisation_data_path, "pretrain/classifier", attr_selected, "weight.pkl"),
        'cpu')
    predictor.cuda()
    predictor.eval()

    dir_dict = {}
    for dir in range(num_directions):
        attr_variation = 0
        for noise in z_loader:
            noise = noise.cuda()
            image = G(noise)
            image = F.avg_pool2d(image, 4, 4)
            img_score = torch.softmax(predictor(image), dim=1)[0][0]
            latent_shift = deformator(one_hot(deformator.input_dim, shift, dir).cuda())
            image_shifted = G(noise + latent_shift.cuda())
            image_shifted = F.avg_pool2d(image_shifted, 4, 4)
            img_shift_score = torch.softmax(predictor(image_shifted), dim=1)[0][0]
            attr_variation = attr_variation + (abs(img_shift_score.detach() - img_score.detach())).mean()
            del image
            del image_shifted
        attr_variation = attr_variation / num_batches
        dir_dict['Direction ' + str(dir)] = attr_variation.item()
        if dir % 50 == 0:
            print('Direction ' + str(dir) + ' completed')
            # sorted_dict = sorted(dir_dict.items(), key=lambda x: x[1], reverse=True)
            # sorted_dict = collections.OrderedDict(sorted_dict)
            # attr_var_dict[attr_selected] = sorted_dict
            # print('\n Saving JSON File!')
            # with open(os.path.join(result_path, 'Attribute_variation_dictionary.json'), 'w') as fp:
            #     json.dump(attr_var_dict, fp)

    sorted_dict = sorted(dir_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_dict = collections.OrderedDict(sorted_dict)
    attr_var_dict[attr_selected] = sorted_dict

print('Computed Attribute variation scores for all attributes')
print('\n Saving JSON File!')
with open(os.path.join(result_path, 'Attribute_variation_dictionary.json'), 'w') as fp:
    json.dump(attr_var_dict, fp)
print('Completed')

## Load JSON Code
## with open(os.path.join(result_path, 'Attribute_variation_dictionary.json'), 'r') as f:
##   data = json.load(f)
