import random
from utils import *
from models.latent_deformator import normal_projection_stat
import torch.nn as nn
from config import generator_kwargs


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

        generator.zero_grad()
        deformator.zero_grad()
        shift_predictor.zero_grad()

        z = torch.randn(self.opt.algo.linear_combo.batch_size, generator.style_dim).cuda()
        target_indices,_, z_shift=self.make_shifts()

        shift = deformator(z_shift)
        w = generator.style(z)
        imgs, _ = generator([w], **generator_kwargs)
        imgs_shifted, _ = generator([w + shift], **generator_kwargs)
        logits, _ = shift_predictor(imgs.detach(), imgs_shifted.detach())
        logit_loss = self.cross_entropy(logits, target_indices.cuda())
        logit_loss.backward()
        shift_predictor_opt.step()


        generator.zero_grad()
        deformator.zero_grad()
        shift_predictor.zero_grad()

        z = torch.randn(self.opt.algo.linear_combo.batch_size, generator.style_dim).cuda()
        epsilon  = self.make_shifts_linear_combo()

        shift = deformator(epsilon)
        w = generator.style(z)
        imgs, _ = generator([w], **generator_kwargs)
        imgs_shifted, _ = generator([w + shift], **generator_kwargs)

        logits , shift_prediction = shift_predictor(imgs, imgs_shifted)
        shift_loss = torch.mean(torch.abs(shift_prediction - epsilon))
        loss = logit_loss + 0.1*shift_loss

        loss.backward()
#        shift_predictor_opt.step()

        deformator_opt.step()


        return deformator, shift_predictor, deformator_opt, shift_predictor_opt, (
            0, 0, loss.item())

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

        target_indices = torch.randint(0, self.opt.algo.ld.num_directions, [self.opt.algo.ld.batch_size]).cuda()
        if self.opt.algo.ld.shift_distribution == "normal":
            shifts = torch.randn(target_indices.shape)
        elif self.opt.algo.ld.shift_distribution == "uniform":
            shifts = 2.0 * torch.rand(target_indices.shape).cuda() - 1.0

        shifts = self.opt.algo.ld.shift_scale * shifts
        shifts[(shifts < self.opt.algo.ld.min_shift) & (shifts > 0)] = self.opt.algo.ld.min_shift
        shifts[(shifts > -self.opt.algo.ld.min_shift) & (shifts < 0)] = -self.opt.algo.ld.min_shift

        try:
            latent_dim[0]
            latent_dim = list(latent_dim)
        except Exception:
            latent_dim = [latent_dim]
        z_shift = torch.zeros([self.opt.algo.ld.batch_size] + latent_dim).cuda()
        for i, (index, val) in enumerate(zip(target_indices, shifts)):
            z_shift[i][index] += val

        return target_indices, shifts, z_shift

    def make_shifts_linear_combo(self):

#        directions_count = list(range(self.opt.algo.linear_combo.num_directions))
#        sampled_directions_batch = [random.sample(directions_count,self.opt.algo.linear_combo.combo_dirs) for x in range(self.opt.algo.linear_combo.batch_size)]
#        ground_truth_idx = torch.Tensor(np.array(sampled_directions_batch)).cuda()
#        selected_directions = torch.zeros((self.opt.algo.linear_combo.batch_size, self.opt.algo.linear_combo.num_directions)).cuda()
#        for idx,nonzero_idx in enumerate(sampled_directions_batch):
#            for i in nonzero_idx:
#                selected_directions[idx][i] = 1
        epsilon = torch.FloatTensor(self.opt.algo.linear_combo.batch_size, self.opt.algo.linear_combo.num_directions).uniform_(-1, 1).cuda()
#        z_shift = selected_directions * epsilon
#        z_shift[(z_shift < self.opt.algo.linear_combo.min_shift) & (z_shift > 0)] = self.opt.algo.linear_combo.min_shift
#        z_shift[(z_shift > -self.opt.algo.linear_combo.min_shift) & (z_shift < 0)] = -self.opt.algo.linear_combo.min_shift
#        ground_truths = z_shift.gather(dim=1, index = ground_truth_idx.long())

        return epsilon
