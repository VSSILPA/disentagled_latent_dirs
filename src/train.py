import random
from utils import *
import torch.nn as nn
import numpy as np


class Trainer(object):

    def __init__(self, config, opt):
        super(Trainer, self).__init__()
        self.config = config
        self.opt = opt
        self.normalize = lambda t: t / np.sqrt(np.sum(t.reshape(-1) ** 2))
        self.ranking_loss = nn.BCEWithLogitsLoss()

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def train_ganspace(self, generator):
        z = torch.randn(self.opt.algo.gs.num_samples, self.opt.algo.ours.latent_dim)
        feats = generator.mapping(z)['w']
        V = torch.svd(feats - feats.mean(0)).V.detach().cpu().numpy()
        deformator = V[:, :self.opt.algo.ours.num_directions]
        deformator_layer = torch.nn.Linear(self.opt.algo.ours.num_directions, V.shape[1])
        deformator_layer.weight.data = torch.FloatTensor(deformator)
        return deformator

    def train_ours(self, generator, deformator, deformator_opt, rank_predictor, rank_predictor_opt):
        generator.zero_grad()
        deformator.zero_grad()
        rank_predictor_opt.zero_grad()
        rank_predictor.train()

        deformator.ortho_mat.data = deformator.ortho_mat / torch.norm(deformator.ortho_mat, dim=0, p=2)
        w_ = generator.sample_latent(int(self.opt.algo.ours.batch_size / 2))
        w = torch.cat((w_, w_), dim=0)

        epsilon, ground_truths = self.make_shifts_rank()
        shift_epsilon = deformator(epsilon)

        w = w.unsqueeze(1).repeat(1, 18, 1)
        shift_epsilon = shift_epsilon.unsqueeze(1).repeat(1, 18, 1)
        w[:, :4, :] = w[:, :4, :] + shift_epsilon[:, :4, :]

        imgs = generator.sample_np(w)
        logits = rank_predictor(imgs.detach())

        epsilon1, epsilon2 = torch.split(logits, int(self.opt.algo.ours.batch_size / 2))
        epsilon_diff = epsilon1 - epsilon2
        rank_predictor_loss = self.ranking_loss(epsilon_diff, ground_truths)
        rank_predictor_loss.backward()
        rank_predictor_opt.step()
        del imgs

        generator.zero_grad()
        deformator.zero_grad()
        rank_predictor.eval()

        z_ = torch.randn(int(self.opt.algo.ours.batch_size / 2), self.opt.algo.ours.latent_dim).cuda()
        z = torch.cat((z_, z_), dim=0)
        epsilon, ground_truths = self.make_shifts_rank()
        shift_epsilon = deformator(epsilon)

        imgs = generator(z + shift_epsilon)
        logits = rank_predictor(imgs)

        epsilon1, epsilon2 = torch.split(logits, int(self.opt.algo.ours.batch_size / 2))
        epsilon_diff = epsilon1 - epsilon2
        deformator_ranking_loss = self.ranking_loss(epsilon_diff, ground_truths)

        deformator_ranking_loss.backward()

        deformator_opt.step()

        return deformator, deformator_opt, rank_predictor, rank_predictor_opt, rank_predictor_loss.item(), deformator_ranking_loss.item()

    def make_shifts_rank(self):
        epsilon = torch.FloatTensor(int(self.opt.algo.ours.batch_size),
                                    self.opt.algo.ours.num_directions).uniform_(-self.opt.algo.ours.shift_min,
                                                                                self.opt.algo.ours.shift_min)

        epsilon_1, epsilon_2 = torch.split(epsilon, int(self.opt.algo.ours.batch_size / 2))
        ground_truths = (epsilon_1 < epsilon_2).type(torch.float32)
        epsilon_1[:, self.opt.algo.ours.num_out_units :] = 0
        epsilon_2[:, self.opt.algo.ours.num_out_units : ] = 0
        ground_truths = ground_truths[:, :4]
        epsilon = torch.cat((epsilon_1, epsilon_2), dim=0)
        return epsilon, ground_truths
