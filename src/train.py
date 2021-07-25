import random
from utils import *
from models.latent_deformator import normal_projection_stat
import torch.nn as nn
from config import generator_kwargs

import logging


class Trainer(object):

    def __init__(self, config, opt):
        super(Trainer, self).__init__()
        self.config = config
        self.opt = opt
        self.cross_entropy = nn.CrossEntropyLoss()
        self.ranking_loss = nn.BCEWithLogitsLoss()
        self.y_real_, self.y_fake_ = torch.ones(int(self.opt.algo.ours.batch_size / 2), 1, device="cuda"), torch.zeros(
            int(self.opt.algo.ours.batch_size / 2),
            1,
            device="cuda")

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def train_ours(self, generator, deformator, deformator_opt, cr_discriminator, cr_optimizer, identity_discriminator):
        generator.zero_grad()
        deformator.zero_grad()

        z_ = torch.randn(int(self.opt.algo.ours.batch_size / 2), generator.z_space_dim).cuda()
        z = torch.cat((z_, z_), dim=0)
        epsilon, ground_truths = self.make_shifts_rank()
        shift_epsilon = deformator(epsilon)
        imgs = generator(z + shift_epsilon)
        logits = cr_discriminator(imgs.detach())

        epsilon1, epsilon2 = torch.split(logits, int(self.opt.algo.ours.batch_size / 2))
        epsilon_diff = epsilon1 - epsilon2
        ranking_loss = self.ranking_loss(epsilon_diff, ground_truths)
        ranking_loss.backward()
        cr_optimizer.step()
        del imgs

        generator.zero_grad()
        deformator.zero_grad()

        z_ = torch.randn(int(self.opt.algo.ours.batch_size / 2), generator.z_space_dim).cuda()
        z = torch.cat((z_, z_), dim=0)
        epsilon, ground_truths = self.make_shifts_rank()
        shift_epsilon = deformator(epsilon)

        imgs_ref = generator(z)
        imgs = generator(z + shift_epsilon)
        logits = cr_discriminator(imgs)
        identity = identity_discriminator(torch.cat((imgs_ref[:int(self.opt.algo.ours.batch_size / 2)],
                                                     imgs[:int(self.opt.algo.ours.batch_size / 2)]), dim=1))

        epsilon1, epsilon2 = torch.split(logits, int(self.opt.algo.ours.batch_size / 2))
        epsilon_diff = epsilon1 - epsilon2
        ranking_loss = self.ranking_loss(epsilon_diff, ground_truths)
        identity_loss = self.ranking_loss(identity, self.y_fake_)
        loss = self.opt.algo.ours.ranking_weight * ranking_loss + self.opt.algo.ours.identity_weight * identity_loss

        loss.backward()

        deformator_opt.step()

        return deformator, deformator_opt, cr_discriminator, cr_optimizer, loss.item()

    def train_ganspace(self, generator):
        z = torch.randn(self.opt.algo.gs.num_samples, generator.style_dim).cuda()
        feats = generator.get_latent(z)
        V = torch.svd(feats - feats.mean(0)).V.detach().cpu().numpy()
        deformator = V[:, :self.opt.algo.gs.num_directions]
        deformator_layer = torch.nn.Linear(self.opt.algo.cf.num_directions, V.shape[1], bias=False)
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
        deformator = V[:, :self.opt.algo.ours.num_directions]
        deformator_layer = torch.nn.Linear(self.opt.algo.ours.num_directions, V.shape[1], bias=False)
        deformator_layer.weight.data = torch.FloatTensor(deformator)
        return deformator_layer

    def make_shifts_rank(self):
        epsilon = torch.FloatTensor(int(self.opt.algo.ours.batch_size),
                                    self.opt.algo.ours.num_directions).uniform_(-self.opt.algo.ours.shift_min, self.opt.algo.ours.shift_max).cuda()

        epsilon_1, epsilon_2 = torch.split(epsilon, int(self.opt.algo.ours.batch_size / 2))
        ground_truths = (epsilon_1 > epsilon_2).type(torch.float32).cuda()
        epsilon = torch.cat((epsilon_1, epsilon_2), dim=0)
        return epsilon, ground_truths
