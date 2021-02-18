import random
from utils import *
from models.latent_deformator import normal_projection_stat
import torch.nn as nn

log = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.cross_entropy = nn.CrossEntropyLoss()
        self.logit_loss = 0
        self.shift_loss = 0
        self.loss = 0
        self.set_seed(self.config['random_seed'])

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def train_gan(self, generator, deformator, shift_predictor, deformator_opt, shift_predictor_opt):

        # should_gen_classes = is_conditional(generator)

        generator.zero_grad()
        deformator.zero_grad()
        shift_predictor.zero_grad()

        z = self.make_noise(self.config['batch_size'], generator.dim_z,truncation=self.config['truncation']).cuda()
        target_indices, shifts, basis_shift = self.make_shifts(deformator.input_dim)

        shift = deformator(basis_shift)

        imgs = generator(z)
        imgs_shifted = generator.gen_shifted(z, shift)

        logits, shift_prediction = shift_predictor(imgs, imgs_shifted)
        logit_loss = self.cross_entropy(logits, target_indices.cuda())
        shift_loss = self.config['shift_weight'] * torch.mean(torch.abs(shift_prediction - shifts))

        loss = logit_loss + shift_loss
        loss.backward()

        deformator_opt.step()
        shift_predictor_opt.step()

        self.logit_loss = self.logit_loss + logit_loss.item()
        self.shift_loss = self.shift_loss + shift_loss.item()
        self.loss = self.loss + loss.item()

        return  deformator, shift_predictor, deformator_opt, shift_predictor_opt, (self.loss ,self.logit_loss, self.shift_loss)

    def make_noise(self, batch, dim, truncation=None):
        if isinstance(dim, int):
            dim = [dim]
        if truncation is None or truncation == 1.0:
            return torch.randn([batch] + dim)
        else:
            return torch.from_numpy(self.truncated_noise([batch] + dim, truncation)).to(torch.float)

    def truncated_noise(self,size, truncation=1.0):
        return truncnorm.rvs(-truncation, truncation, size=size)

    def make_shifts(self, latent_dim):
        target_indices = torch.randint(0, self.config['directions_count'], [self.config['batch_size']],device='cuda')
        if self.config['shift_distribution'] == "normal":
            shifts = torch.randn(target_indices.shape, device='cuda')
        elif self.config['shift_distribution'] == "uniform":
            shifts = 2.0 * torch.rand(target_indices.shape, device='cuda') - 1.0

        shifts = self.config['shift_scale'] * shifts
        shifts[(shifts < self.config['min_shift']) & (shifts > 0)] = self.config['min_shift']
        shifts[(shifts > -self.config['min_shift']) & (shifts < 0)] = -self.config['min_shift']

        try:
            latent_dim[0]
            latent_dim = list(latent_dim)
        except Exception:
            latent_dim = [latent_dim]
        z_shift = torch.zeros([self.config['batch_size']] + latent_dim, device='cuda')
        for i, (index, val) in enumerate(zip(target_indices, shifts)):
            z_shift[i][index] += val

        return target_indices, shifts, z_shift


