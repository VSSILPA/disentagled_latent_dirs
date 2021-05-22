import random
from utils import *
from models.latent_deformator import normal_projection_stat
import torch.nn as nn
from config import generator_kwargs

log = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)


class Trainer(object):

    def __init__(self, config, opt):
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

        z = torch.randn(self.opt.algo.ld.batch_size, generator.style_dim).cuda()
        target_indices, shifts, z_shift = self.make_shifts(deformator.input_dim)

        shift = deformator(z_shift)
        w = generator.style(z)
        imgs, _ = generator([w], **generator_kwargs)
        imgs_shifted, _ = generator([w + shift], **generator_kwargs)

        logits, shift_prediction = shift_predictor(imgs, imgs_shifted)
        logit_loss = self.cross_entropy(logits, target_indices.cuda())
        shift_loss = self.opt.algo.ld.shift_weight * torch.mean(torch.abs(shift_prediction - shifts))

        loss = logit_loss + shift_loss
        loss.backward()

        deformator_opt.step()
        shift_predictor_opt.step()

        return deformator, shift_predictor, deformator_opt, shift_predictor_opt, (
        loss.item(), logit_loss.item(), shift_loss.item())

    def train_ganspace(self, generator):
        z = torch.randn(self.opt.algo.gs.num_samples, generator.style_dim).cuda()
        feats = generator.get_latent(z)
        V = torch.svd(feats - feats.mean(0)).V.detach().cpu().numpy()
        deformator = V[:, :self.opt.algo.gs.num_directions]
        deformator_layer = torch.nn.Linear(self.opt.algo.cf.num_directions, 512)
        deformator_layer.weight.data = torch.FloatTensor(deformator)
        return deformator_layer

    def train_closed_form(self,generator):
        modulate = {
            k: v
            for k, v in generator.state_dict().items()
            if "modulation" in k and "to_rgbs" not in k and "weight" in k
        }
        weight_mat = []
        for k, v in modulate.items():
            weight_mat.append(v)
        W = torch.cat(weight_mat, 0)
        V = torch.svd(W).V.detach().cpu().numpy()
        deformator = V[:, :self.opt.algo.cf.num_directions]
        deformator_layer = torch.nn.Linear(self.opt.algo.cf.num_directions, 512)
        deformator_layer.weight.data = torch.FloatTensor(deformator)
        return deformator_layer

    def make_shifts(self, latent_dim):
        target_indices = torch.randint(0, self.opt.algo.ld.directions_count, [self.opt.algo.ld.batch_size],
                                       device='cuda')
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
        z_shift = torch.zeros([self.opt.algo.ld.batch_size] + latent_dim, device='cuda')
        for i, (index, val) in enumerate(zip(target_indices, shifts)):
            z_shift[i][index] += val

        return target_indices, shifts, z_shift
