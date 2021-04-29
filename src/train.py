import random
from utils import *
from models.latent_deformator import normal_projection_stat
import torch.nn as nn

log = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, config ,opt):
        super(Trainer, self).__init__()
        self.config = config
        self.opt = opt
        self.cross_entropy = nn.CrossEntropyLoss()


    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def train_latent_discovery(self, generator, deformator, shift_predictor, deformator_opt, shift_predictor_opt):

        # should_gen_classes = is_conditional(generator)

        generator.zero_grad()
        deformator.zero_grad()
        shift_predictor.zero_grad()

        z = make_noise(self.opt.batch_size, generator.latent_size,truncation=self.opt.algo.ld.truncation).cuda()
        target_indices, shifts, basis_shift = self.make_shifts(deformator.input_dim)

        shift = deformator(basis_shift)
        imgs , _ = generator(z,self.opt.depth,self.opt.alpha)
        imgs_shifted, _ = generator(z + shift,self.opt.depth,self.opt.alpha)

        logits, shift_prediction = shift_predictor(imgs, imgs_shifted)
        logit_loss = self.cross_entropy(logits, target_indices.cuda())
        shift_loss = self.opt.algo.ld.shift_weight * torch.mean(torch.abs(shift_prediction - shifts))

        loss = logit_loss + shift_loss
        loss.backward()

        deformator_opt.step()
        shift_predictor_opt.step()


        return  deformator, shift_predictor, deformator_opt, shift_predictor_opt, (loss.item() ,logit_loss.item(), shift_loss.item())

    @staticmethod
    def make_noise(batch, dim, truncation=None):
        if isinstance(dim, int):
            dim = [dim]
        if truncation is None or truncation == 1.0:
            return torch.randn([batch] + dim)
        else:
            return torch.from_numpy(truncated_noise([batch] + dim, truncation)).to(torch.float)

    @staticmethod
    def truncated_noise(size, truncation=1.0):
        return truncnorm.rvs(-truncation, truncation, size=size)

    def make_shifts(self, latent_dim):
        target_indices = torch.randint(0, self.opt.algo.ld.directions_count, [self.opt.batch_size],device='cuda')
        if self.opt.algo.ld.shift_distribution == "normal":
            shifts = torch.randn(target_indices.shape, device='cuda')
        elif self.opt.algo.ld.shift_distribution == "uniform":
            shifts = 2.0 * torch.rand(target_indices.shape, device='cuda') - 1.0

        shifts = self.opt.algo.ld.shift_scale * shifts
        shifts[(shifts < self.opt.algo.ld.min_shift) & (shifts > 0)] = self.opt.algo.ld.min_shift
        shifts[(shifts > -self.opt.algo.ld.min_shift) & (shifts < 0)] = -self.opt.algo.ld.min_shift

        try:
            latent_dim[0]
            latent_dim = list(latent_dim)
        except Exception:
            latent_dim = [latent_dim]
        z_shift = torch.zeros([self.opt.batch_size] + latent_dim, device='cuda')
        for i, (index, val) in enumerate(zip(target_indices, shifts)):
            z_shift[i][index] += val

        return target_indices, shifts, z_shift


