
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import torch
import os
import json
from utils import load_generator
from PIL import Image
from models import parse_gan_type
from utils import to_tensor
import torchvision.transforms as transforms
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

def postprocess(images, min_val=-1.0, max_val=1.0):
    """Post-processes images from `torch.Tensor` to `numpy.ndarray`.

    Args:
        images: A `torch.Tensor` with shape `NCHW` to process.
        min_val: The minimum value of the input tensor. (default: -1.0)
        max_val: The maximum value of the input tensor. (default: 1.0)

    Returns:
        A `numpy.ndarray` with shape `NHWC` and pixel range [0, 255].
    """
    assert isinstance(images, torch.Tensor)
    images = images.detach().cpu().numpy()
    images = (images - min_val) * 255 / (max_val - min_val)
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)
    return images

class NoiseDataset(Dataset):
    def __init__(self, latent_codes, num_samples, z_dim):
        self.num_samples = num_samples
        self.z_dim = z_dim
        self.data = latent_codes

    def __getitem__(self, index):
        x = self.data[index]
        return x

    def __len__(self):
        return len(self.data)

visualisation_data_path = 'all_pretrained_models'

set_seed(1234)

perf_logger = PerfomanceLogger()

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

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

pose_predictor = get_classifier(
        os.path.join(visualisation_data_path, "classifier", 'pose', "weight.pkl"),
        'cpu')
glass_predictor = get_classifier(
        os.path.join(visualisation_data_path, "classifier", 'eyeglasses', "weight.pkl"),
        'cpu')
gender_predictor = get_classifier(
        os.path.join(visualisation_data_path, "classifier", 'male', "weight.pkl"),
        'cpu')
smile_predictor = get_classifier(
        os.path.join(visualisation_data_path, "classifier", 'smiling', "weight.pkl"),
        'cpu')
age_predictor = get_classifier(
        os.path.join(visualisation_data_path, "classifier", 'young', "weight.pkl"),
        'cpu')
pose_predictor.cuda().eval()
glass_predictor.cuda().eval()
gender_predictor.cuda().eval()
smile_predictor.cuda().eval()
age_predictor.cuda().eval()

generator = load_generator('stylegan_celebahq1024')
gan_type = parse_gan_type(generator)

num_directions = 512
num_samples = 2000 ##TODO
batch_size  = 2 ##TODO
num_batches = int(num_directions / num_samples)


layers, cf_eigenvectors, _ = torch.load(os.path.join(visualisation_data_path ,'sefa-model-stylegan-celebA.pkl'), map_location= 'cpu')
codes = torch.randn(num_samples, generator.z_space_dim).cuda()
if gan_type == 'pggan':
    codes = generator.layer0.pixel_norm(codes)
elif gan_type in ['stylegan', 'stylegan2']:
    codes = generator.mapping(codes)['w']
    codes = generator.truncation(codes,
                                 trunc_psi=0.7,
                                 trunc_layers=8)
codes = codes.detach()
z = NoiseDataset(latent_codes=codes, num_samples=num_samples, z_dim=generator.z_space_dim)
z_loader = DataLoader(z, batch_size=batch_size, shuffle=False)
torch.save(z, os.path.join(visualisation_data_path, 'z_analysis.pkl'))


epsilon = 2

attribute_scores_ref = []
with torch.no_grad():
    for w in z_loader:
        w = w.cuda()

        images = generator.synthesis(w)['image']
        images = postprocess(images)
        transformed_images = [transform(Image.fromarray(img)).unsqueeze(0) for img in images]
        predict_images = torch.cat(transformed_images, dim=0).cuda()

        img_score_1 = pose_predictor(predict_images)[:, 0].detach()
        img_score_2 = glass_predictor(predict_images)[:, 0].detach()
        img_score_3 = gender_predictor(predict_images)[:, 0].detach()
        img_score_4 = smile_predictor(predict_images)[:, 0].detach()
        img_score_5 = age_predictor(predict_images)[:, 0].detach()
        img_score = [img_score_1, img_score_2, img_score_3, img_score_4, img_score_5]
        attribute_scores_ref.append(img_score)

dir_dict = {}
with torch.no_grad():
    for dir in range(4,5):
        attr_variation_1 = 0
        attr_variation_2 = 0
        attr_variation_3 = 0
        attr_variation_4 = 0
        attr_variation_5 = 0
        for i, w in enumerate(z_loader):
            perf_logger.start_monitoring("Batch" + str(i) + " completed")
            w_shift = w[:, layers, :].detach().cpu().numpy() + cf_eigenvectors[dir: dir+1] * epsilon
            images_shifted = generator.synthesis(to_tensor(w_shift))['image']
            images_shifted = postprocess(images_shifted)
            transformed_images = [transform(Image.fromarray(img)).unsqueeze(0) for img in images_shifted]
            predict_images = torch.cat(transformed_images, dim=0).cuda()

            img_shift_score_1 = pose_predictor(predict_images)[:, 0].detach()
            img_shift_score_2 = glass_predictor(predict_images)[:,0].detach()
            img_shift_score_3 = gender_predictor(predict_images)[:, 0].detach()
            img_shift_score_4 = smile_predictor(predict_images)[:, 0].detach()
            img_shift_score_5 = age_predictor(predict_images)[:, 0].detach()
            attr_variation_1 = attr_variation_1 + (
                abs(img_shift_score_1.detach() - attribute_scores_ref[i][0])).mean()
            attr_variation_2 = attr_variation_2 + (
                abs(img_shift_score_2.detach() - attribute_scores_ref[i][1])).mean()
            attr_variation_3 = attr_variation_3 + (
                abs(img_shift_score_3.detach() - attribute_scores_ref[i][2])).mean()
            attr_variation_4 = attr_variation_4 + (
                abs(img_shift_score_4.detach() - attribute_scores_ref[i][3])).mean()
            attr_variation_5 = attr_variation_5 + (
                abs(img_shift_score_5.detach() - attribute_scores_ref[i][4])).mean()
            del predict_images
            perf_logger.start_monitoring("Batch" + str(i) + " completed")
        attr_variation_1 = attr_variation_1 / num_batches
        attr_variation_2 = attr_variation_2 / num_batches
        attr_variation_3 = attr_variation_3 / num_batches
        attr_variation_4 = attr_variation_4 / num_batches
        attr_variation_5 = attr_variation_5 / num_batches
        dir_dict['Direction ' + str(dir)] = [attr_variation_1.item(), attr_variation_2.item(),
                                             attr_variation_3.item(), attr_variation_4.item(),
                                             attr_variation_5.item()]
        # if dir % 5 == 0 or dir == (num_directions- 1):
        #
        #     sorted_dict = sorted(dir_dict.items(), key=lambda x: x[1], reverse=True)
        #     sorted_dict = collections.OrderedDict(sorted_dict)
        #     attr_var_dict[attr_selected] = sorted_dict
        #     print('\n Saving JSON File (Intermediate)!')

        perf_logger.stop_monitoring("Direction " + str(dir) + " completed")
        with open(os.path.join(visualisation_data_path, 'Attribute_variation_dictionary.json'), 'w') as fp:
            json.dump(dir_dict, fp)
