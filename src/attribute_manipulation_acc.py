from utils import *
from torch.utils.data import DataLoader, Dataset
from models.latent_deformator import LatentDeformator
import numpy as np
import torch
import random
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


gan_type = 'prog-gan-sefa'
total_directions = 512
num_samples = 100  ##TODO 2000
batch_size = 2  ##TODO 10
shift = 10
direction_indices = [12, 1]  ##TODO Random directions
num_directions = 2
num_classifiers = 2
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

attr_list = ['pose', 'eyeglasses', 'male', 'smiling',
             'young']  ##TODO ['pose', 'eyeglasses', 'male', 'smiling', 'young']
predictor_list = []
for attr_selected in attr_list:
    predictor = get_classifier(
        os.path.join(visualisation_data_path, "pretrain/classifier", attr_selected, "weight.pkl"),
        'cpu')
    predictor.cuda()
    predictor.eval()
    predictor_list.append(predictor)


def get_attribute_manipulation_acc(dir, predictor, z_loader, G, deformator):
    attr_manipulation_acc = torch.LongTensor().cuda()
    with torch.no_grad():
        for batch_idx, noise in enumerate(z_loader):
            latent_shift = deformator(one_hot(deformator.input_dim, shift, dir).cuda())
            image_shifted = G(noise.cuda() + latent_shift.cuda())
            image_shifted = F.avg_pool2d(image_shifted, 4, 4)
            scores = torch.argmax(torch.softmax(predictor(image_shifted), dim=1), dim=-1)
            attr_manipulation_acc = torch.cat((attr_manipulation_acc, scores))

    attr_manipulation_acc = attr_manipulation_acc.sum() / num_samples
    return attr_manipulation_acc


z = NoiseDataset(num_samples=num_samples, z_dim=G.z_space_dim)  ##TODO Change
z_loader = DataLoader(z, batch_size=batch_size, shuffle=False)

predictor_index  = 2  ##TODO Change
dir = 6  ##TODO Change
predictor = predictor_list[predictor_index]

attr_1_acc = get_attribute_manipulation_acc(dir, predictor, z_loader, G, deformator)
print(attr_list[pred ] + ' Direction Accuracy : ' + str(attr_1_acc))

