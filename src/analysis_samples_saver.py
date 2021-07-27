from utils import *
from torch.utils.data import DataLoader, Dataset
from models.latent_deformator import LatentDeformator
import numpy as np
import random
from models.proggan_sefa import PGGANGenerator
import torch.nn.functional as F
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

perf_logger = PerfomanceLogger()

gan_type = 'prog-gan-sefa'
num_directions = 512

# visualisation_data_path = '/media/adarsh/DATA/CelebA-Analysis/'
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

pretrained_model = torch.load('../pretrained_models/cf_model.pkl', map_location='cpu')
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


class ImageDataset(Dataset):
    def __init__(self,images, images_shifted):
        self.images = images
        self.images_shifted = images_shifted

    def __getitem__(self, index):
        return self.images[index] , self.images_shifted[index]

    def __len__(self):
        return len(self.images)

num_samples = 2000 ##TODO 2000
batch_size = 5 ##TODO 5
num_batches = int(num_samples / batch_size)
z = NoiseDataset(num_samples=num_samples, z_dim=G.z_space_dim)
z_loader = DataLoader(z, batch_size=batch_size, shuffle=False)
torch.save(z, os.path.join(result_path, 'z_analysis.pkl'))
images_ = []
images_shifted_ = []
shift = 10
for noise in z_loader:
    noise = noise.cuda()
    image = G(noise)
    image = F.avg_pool2d(image, 4, 4)
    latent_shift = deformator(one_hot(deformator.input_dim, shift, dir).cuda())
    image_shifted = G(noise + latent_shift.cuda())
    image_shifted = F.avg_pool2d(image_shifted, 4, 4)
    images_.append((image.detach().cpu()))
    images_shifted_.append((image_shifted.detach().cpu()))

images_final = torch.stack(images_)
torch.save(images_final, os.path.join(result_path, 'images_eval.pkl'))
images_shift = torch.stack(images_shifted_)
torch.save(images_shift,os.path.join(result_path, 'images_shift.pkl'))



