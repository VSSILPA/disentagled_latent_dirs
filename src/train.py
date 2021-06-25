import random
from utils import *
from models.latent_deformator import normal_projection_stat
import torch.nn as nn


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

    def train_discrete_ld(self, generator, deformator, shift_predictor, deformator_opt, shift_predictor_opt):

        generator.zero_grad()
        deformator.zero_grad()
        shift_predictor.zero_grad()

        z = torch.randn(self.opt.algo.discrete_ld.batch_size, generator.dim_z).cuda()
        epsilon, targets = self.make_shifts_discrete_ld()

        shift = deformator(epsilon)
        # imgs_ref = generator(z)
        imgs_shifted = generator(z + shift)

        logits = shift_predictor(imgs_shifted)
        logit_loss = self.cross_entropy(logits, targets.cuda())
        logit_loss.backward()
        #
        shift_predictor_opt.step()
        deformator_opt.step()

        return deformator, shift_predictor, deformator_opt, shift_predictor_opt, (0, 0, 0)

    def train_ganspace(self, generator):

        z = torch.randn(self.opt.algo.gs.num_samples, generator.style_dim).cuda()
        feats = generator.get_latent(z)
        V = torch.svd(feats - feats.mean(0)).V.detach().cpu().numpy()
        deformator = V[:, :self.opt.algo.gs.num_directions]
        deformator_layer = torch.nn.Linear(self.opt.algo.cf.num_directions, V.shape[1])
        deformator_layer.weight.data = torch.FloatTensor(deformator)
        return deformator_layer

    def train_closed_form(self, generator):

        modulate = {
            k: v
            for k, v in generator.state_dict().items()
            if "modulation" in k and "to_rgbs" not in k and "weight" in k
        }
        weight_mat = []
        for k, v in modulate.items():
            weight_mat.append(v)
        W = torch.cat(weight_mat[:-1], 0)
        V = torch.svd(W).V.detach().cpu().numpy()
        deformator = V[:, :self.opt.algo.cf.num_directions]
        deformator_layer = torch.nn.Linear(self.opt.algo.cf.num_directions, V.shape[1])
        deformator_layer.weight.data = torch.FloatTensor(deformator)
        return deformator_layer

    def make_shifts(self, latent_dim):

        target_indices = torch.randint(0, self.opt.algo.linear_combo.num_directions,
                                       [self.opt.algo.linear_combo.batch_size]).cuda()
        if self.opt.algo.linear_combo.shift_distribution == "normal":
            shifts = torch.randn(target_indices.shape)
        elif self.opt.algo.linear_combo.shift_distribution == "uniform":
            shifts = 2.0 * torch.rand(target_indices.shape).cuda() - 1.0

        shifts = self.opt.algo.linear_combo.shift_scale * shifts
        shifts[(shifts < self.opt.algo.linear_combo.min_shift) & (shifts > 0)] = self.opt.algo.linear_combo.min_shift
        shifts[(shifts > -self.opt.algo.linear_combo.min_shift) & (shifts < 0)] = -self.opt.algo.linear_combo.min_shift

        try:
            latent_dim[0]
            latent_dim = list(latent_dim)
        except Exception:
            latent_dim = [latent_dim]
        z_shift = torch.zeros([self.opt.algo.linear_combo.batch_size] + latent_dim).cuda()
        for i, (index, val) in enumerate(zip(target_indices, shifts)):
            z_shift[i][index] += val

        return target_indices, shifts, z_shift

    def make_shifts_discrete_ld(self):

        # directions_count = list(range(self.opt.algo.linear_combo.num_directions)) sampled_directions_batch = [
        # random.sample(directions_count,self.opt.algo.linear_combo.combo_dirs) for x in range(
        # self.opt.algo.linear_combo.batch_size)] ground_truth_idx = torch.Tensor(np.array(
        # sampled_directions_batch)).cuda() selected_directions = torch.zeros((self.opt.algo.linear_combo.batch_size,
        # self.opt.algo.linear_combo.num_directions)).cuda() for idx,nonzero_idx in enumerate(
        # sampled_directions_batch): for i in nonzero_idx: selected_directions[idx][i] = 1
        target = torch.randint(0, self.opt.algo.discrete_ld.num_directions, (self.opt.algo.discrete_ld.batch_size,))
        epsilon = torch.nn.functional.one_hot(target,
                                              num_classes=10).cuda()
        epsilon = epsilon.type(torch.float32)
        # noise = torch.FloatTensor(self.opt.algo.discrete_ld.batch_size,
        #                             1).uniform_(-6, 6).cuda()
        # epsilon = epsilon * noise

        # z_shift = selected_directions * epsilon z_shift[(z_shift < self.opt.algo.linear_combo.min_shift) & (z_shift
        # > 0)] = self.opt.algo.linear_combo.min_shift z_shift[(z_shift > -self.opt.algo.linear_combo.min_shift) & (
        # z_shift < 0)] = -self.opt.algo.linear_combo.min_shift ground_truths = z_shift.gather(dim=1,
        # index = ground_truth_idx.long())

        return epsilon, target
