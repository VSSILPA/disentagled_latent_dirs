from utils import *
from torch.utils.data import DataLoader, Dataset
from models.latent_deformator import LatentDeformator
import numpy as np
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
num_samples = 10 ##TODO 2000
batch_size = 2 ##TODO 10
shift = 10
direction_indices = [12, 1]  ##TODO Random directions
num_directions = 2
num_classifiers =2
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

attr_list = ['pose', 'eyeglasses'] ##TODO ['pose', 'eyeglasses', 'male', 'smiling', 'young']
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
z = torch.load(os.path.join(result_path, 'z_analysis.pkl'))[0:10] ##TODO CHange
z_loader = DataLoader(z, batch_size=batch_size, shuffle=False)

ref_image_scores = torch.zeros([num_classifiers, num_samples])
with torch.no_grad():
    for batch_idx, noise in enumerate(z_loader):
        noise = noise.cuda()
        image = G(noise)
        image = F.avg_pool2d(image, 4, 4)
        for predictor_idx, predictor in enumerate(predictor_list):
            ref_image_scores[predictor_idx, batch_idx*batch_size:(batch_idx+1)*batch_size] = torch.softmax(predictor(image), dim=1)[:,0]
ref_image_scores = ref_image_scores.unsqueeze(0).repeat(num_directions, 1, 1)

shifted_image_scores = torch.zeros([num_directions, num_classifiers, num_samples])
with torch.no_grad():
    for dir in range(len(direction_indices)):
        latent_shift = deformator(one_hot(deformator.input_dim, shift, dir).cuda())
        for batch_idx,noise in enumerate(z_loader):
            image_shifted = G(noise.cuda() + latent_shift.cuda())
            image_shifted = F.avg_pool2d(image_shifted, 4, 4)
            for predictor_idx, predictor in enumerate(predictor_list):
                shifted_image_scores[dir, predictor_idx, batch_idx*batch_size:(batch_idx+1)*batch_size] = torch.softmax(predictor(image_shifted), dim=1)[:,0]

difference_matrix = torch.abs(shifted_image_scores - ref_image_scores)
rescoring_analysis_matrix = torch.mean(difference_matrix, dim=-1).numpy()

import matplotlib
import matplotlib.pyplot as plt

attr_list = ['pose', 'eyeglasses']
directions = [str(x) for x in range(2)]

fig, ax = plt.subplots()
im = ax.imshow(rescoring_analysis_matrix)

ax.set_xticks(np.arange(len(attr_list)))
ax.set_yticks(np.arange(len(directions)))
# ... and label them with the respective list entries
ax.set_xticklabels(attr_list)
ax.set_yticklabels(directions)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(attr_list)):
    for j in range(len(directions)):
        text = ax.text(j, i, rescoring_analysis_matrix[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Rescoring analysis Closed form")
fig.tight_layout()
plt.show()
plt.savefig('/home/adarsh/PycharmProjects/disentagled_latent_dirs/src/test.png')

print("hello")