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
from logger import PerfomanceLogger


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


perf_logger = PerfomanceLogger()

gan_type = 'prog-gan-sefa'
num_directions = 512

visualisation_data_path = '../results/'
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

pretrained_model = torch.load('../pretrained_models/CelebAAnalysis/cf_model.pkl', map_location='cpu')
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
batch_size = 1
num_batches = int(num_samples / batch_size)
z = NoiseDataset(num_samples=num_samples, z_dim=G.z_space_dim)
z_loader = DataLoader(z, batch_size=batch_size, shuffle=False)
torch.save(z, os.path.join(result_path, 'z_analysis.pkl'))
shift = 10
attr_list = ['pose', 'eyeglasses', 'male', 'smiling', 'young']
attr_var_dict = collections.OrderedDict()


pose_predictor = get_classifier(
        os.path.join("../pretrained_models/classifier", 'pose', "weight.pkl"),
        'cpu')
glass_predictor = get_classifier(
        os.path.join("../pretrained_models/classifier", 'eyeglasses', "weight.pkl"),
        'cpu')
gender_predictor = get_classifier(
        os.path.join("../pretrained_models/classifier", 'male', "weight.pkl"),
        'cpu')
smile_predictor = get_classifier(
        os.path.join("../pretrained_models/classifier", 'smiling', "weight.pkl"),
        'cpu')
age_predictor = get_classifier(
        os.path.join("../pretrained_models/classifier", 'young', "weight.pkl"),
        'cpu')
pose_predictor.cuda()
pose_predictor.eval()
glass_predictor.cuda()
glass_predictor.eval()
gender_predictor.cuda()
gender_predictor.eval()
smile_predictor.cuda()
smile_predictor.eval()
age_predictor.cuda()
age_predictor.eval()


attribute_scores_ref = []
with torch.no_grad():
    for noise in z_loader:
        noise = noise.cuda()
        image = G(noise)
        image = F.avg_pool2d(image, 4, 4)
        img_score_1 = torch.softmax(pose_predictor(image), dim=1)[0][0].detach()
        img_score_2 = torch.softmax(glass_predictor(image), dim=1)[0][0].detach()
        img_score_3 = torch.softmax(gender_predictor(image), dim=1)[0][0].detach()
        img_score_4 = torch.softmax(smile_predictor(image), dim=1)[0][0].detach()
        img_score_5 = torch.softmax(age_predictor(image), dim=1)[0][0].detach()
        img_score = [img_score_1, img_score_2, img_score_3, img_score_4, img_score_5]
        attribute_scores_ref.append(img_score)

dir_dict = {}
with torch.no_grad():
    for dir in range(num_directions):
        attr_variation_1 = 0
        attr_variation_2 = 0
        attr_variation_3 = 0
        attr_variation_4 = 0
        attr_variation_5 = 0
        perf_logger.start_monitoring("Direction " + str(dir) + " completed")
        for i, noise in enumerate(z_loader):
            latent_shift = deformator(one_hot(deformator.input_dim, shift, dir).cuda())
            image_shifted = G(noise + latent_shift.cuda())
            image_shifted = F.avg_pool2d(image_shifted, 4, 4)
            img_shift_score_1 = torch.softmax(pose_predictor(image_shifted), dim=1)[0][0].detach()
            img_shift_score_2 = torch.softmax(glass_predictor(image_shifted), dim=1)[0][0].detach()
            img_shift_score_3 = torch.softmax(gender_predictor(image_shifted), dim=1)[0][0].detach()
            img_shift_score_4 = torch.softmax(smile_predictor(image_shifted), dim=1)[0][0].detach()
            img_shift_score_5 = torch.softmax(age_predictor(image_shifted), dim=1)[0][0].detach()
            attr_variation_1 = attr_variation_1 + (abs(img_shift_score_1.detach() - attribute_scores_ref[i][0])).mean()
            attr_variation_2 = attr_variation_2 + (abs(img_shift_score_2.detach() - attribute_scores_ref[i][1])).mean()
            attr_variation_3 = attr_variation_3 + (abs(img_shift_score_3.detach() - attribute_scores_ref[i][2])).mean()
            attr_variation_4 = attr_variation_4 + (abs(img_shift_score_4.detach() - attribute_scores_ref[i][3])).mean()
            attr_variation_5 = attr_variation_5 + (abs(img_shift_score_5.detach() - attribute_scores_ref[i][4])).mean()
            del image_shifted
        attr_variation_1 = attr_variation_1 / num_batches
        attr_variation_2 = attr_variation_2 / num_batches
        attr_variation_3 = attr_variation_3 / num_batches
        attr_variation_4 = attr_variation_4 / num_batches
        attr_variation_5 = attr_variation_5 / num_batches
        dir_dict['Direction ' + str(dir)] = [attr_variation_1.item(), attr_variation_2.item(), attr_variation_3.item(), attr_variation_4.item(), attr_variation_5.item()]
        # if dir % 5 == 0 or dir == (num_directions- 1):
        #
        #     sorted_dict = sorted(dir_dict.items(), key=lambda x: x[1], reverse=True)
        #     sorted_dict = collections.OrderedDict(sorted_dict)
        #     attr_var_dict[attr_selected] = sorted_dict
        #     print('\n Saving JSON File (Intermediate)!')

        perf_logger.stop_monitoring("Direction " + str(dir) + " completed")
        with open(os.path.join(result_path, 'Attribute_variation_dictionary.json'), 'w') as fp:
            json.dump(dir_dict, fp)

## Load JSON Code
## with open(os.path.join(result_path, 'Attribute_variation_dictionary.json'), 'r') as f:
##   data = json.load(f)
