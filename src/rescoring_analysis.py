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
total_directions = 512
num_samples = 2000
batch_size = 5
shift = 10
direction_indices = [12, 1, 9, 82, 107]  ##TODO Random directions
num_directions = 5
num_classifiers = 5
num_batches = int(num_samples / batch_size)

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
deformator = LatentDeformator(shift_dim=G.z_space_dim, input_dim=total_directions,
                              out_dim=G.z_space_dim, type='linear', random_init=True,
                              bias=False)  ##TODO Change bias True

deformator.load_state_dict(pretrained_model['deformator'])
deformator.cuda()
deformator.eval()

attr_list = ['pose', 'eyeglasses', 'male', 'smiling', 'young']
predictor_list = []
for attr_selected in attr_list:
    predictor = get_classifier(
        os.path.join(visualisation_data_path, "pretrain/classifier", attr_selected, "weight.pkl"),
        'cpu')
    predictor.cuda()
    predictor.eval()
    predictor_list.append(predictor)

# z = NoiseDataset(num_samples=num_samples, z_dim=G.z_space_dim)
# torch.save(z, os.path.join(result_path, 'z_analysis.pkl'))
#
z = torch.load(os.path.join(result_path, 'z_analysis.pkl'))
z_loader = DataLoader(z, batch_size=batch_size, shuffle=False)

ref_image_scores = torch.zeros([num_classifiers, num_batches])
with torch.no_grad():
    for batch_idx, noise in z_loader:
        noise = noise.cuda()
        image = G(noise)
        image = F.avg_pool2d(image, 4, 4)
        for predictor_idx, predictor in enumerate(predictor_list):
            ref_image_scores[predictor_idx, batch_idx] = torch.softmax(predictor(image), dim=1)[0][0].item()
ref_image_scores = ref_image_scores.unsqueeze(0).repeat(num_directions, 1, 1)

shifted_image_scores = torch.zeros([num_directions, num_classifiers, num_batches])
with torch.no_grad():
    for dir in range(len(direction_indices)):
        latent_shift = deformator(one_hot(deformator.input_dim, shift, dir).cuda())
        for batch_idx, noise in enumerate(z_loader):
            image_shifted = G(noise + latent_shift.cuda())
            image_shifted = F.avg_pool2d(image_shifted, 4, 4)
            for predictor_idx, predictor in enumerate(predictor_list):
                shifted_image_scores[dir, predictor_idx, batch_idx] = torch.softmax(predictor(image), dim=1)[0][
                    0].item()

difference_matrix = torch.abs(shifted_image_scores - ref_image_scores)
rescoring_analysis_matrix = torch.mean(difference_matrix, dim=-1)
