import random
from utils import *
from models.latent_deformator import normal_projection_stat
import torch.nn as nn
import torch.functional as F


class Trainer(object):

    def __init__(self, config, opt):
        super(Trainer, self).__init__()
        self.config = config
        self.opt = opt
        self.cross_entropy = nn.CrossEntropyLoss()
        self.adversarial_loss = torch.nn.BCELoss()
        self.similarity_loss = nn.TripletMarginLoss()

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def train_discrete_ld(self, generator, discriminator, disc_opt, deformator, shift_predictor, deformator_opt,
                          shift_predictor_opt, train_loader):

        generator.zero_grad()
        deformator.zero_grad()

        label_fake = torch.full((self.opt.algo.discrete_ld.batch_size,), 0, dtype=torch.float32).cuda()
        label_real = torch.full((self.opt.algo.discrete_ld.batch_size,), 1, dtype=torch.float32).cuda()

        z = torch.randn(self.opt.algo.discrete_ld.batch_size, generator.dim_z).cuda()
        z_pos = torch.randn(self.opt.algo.discrete_ld.batch_size, generator.dim_z).cuda()
        images = generator(z)
        epsilon_ref, epsilon_neg, targets = self.make_shifts_discrete_ld()

        z_final = torch.cat((z, z_pos, z), dim=0)
        epsilon = torch.cat((epsilon_ref, epsilon_ref, epsilon_neg), dim=0)

        shift = deformator(epsilon)

        disc_opt.zero_grad()
        prob_real, _, _ = discriminator(images.detach().cuda())
        loss_D_real = self.adversarial_loss(prob_real.view(-1), label_real)
        loss_D_real.backward()

        imgs_shifted = generator(z_final + shift)

        prob_fake_D, _, _ = discriminator(imgs_shifted[:self.opt.algo.discrete_ld.batch_size].detach())

        loss_D_fake = self.adversarial_loss(prob_fake_D.view(-1), label_fake)
        loss_D_fake.backward()

        disc_opt.step()

        generator.zero_grad()
        deformator.zero_grad()

        prob_fake, logits, similarity = discriminator(imgs_shifted)
        # logits, z_rec = shift_predictor(imgs_shifted)

        loss_G = self.adversarial_loss(prob_fake.view(-1)[:self.opt.algo.discrete_ld.batch_size], label_real)
        loss = loss_G + self.cross_entropy(logits[:self.opt.algo.discrete_ld.batch_size], targets[
                                                                                          :self.opt.algo.discrete_ld.batch_size].cuda()) + self.similarity_loss(
            similarity[:self.opt.algo.discrete_ld.batch_size],
            similarity[self.opt.algo.discrete_ld.batch_size:2 * self.opt.algo.discrete_ld.batch_size],
            similarity[2 * self.opt.algo.discrete_ld.batch_size:])
        loss.backward()

        # shift_predictor_opt.step()
        deformator_opt.step()

        return deformator, discriminator, disc_opt, shift_predictor, deformator_opt, shift_predictor_opt, (0, 0, 0)

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

        target_indices = torch.randint(0, self.opt.algo.discrete_ld.num_directions,
                                       [self.opt.algo.discrete_ld.batch_size]).cuda()
        if self.opt.algo.discrete_ld.shift_distribution == "normal":
            shifts = torch.randn(target_indices.shape)
        elif self.opt.algo.discrete_ld.shift_distribution == "uniform":
            shifts = 2.0 * torch.rand(target_indices.shape).cuda() - 1.0

        shifts = self.opt.algo.discrete_ld.shift_scale * shifts
        shifts[(shifts < self.opt.algo.discrete_ld.min_shift) & (shifts > 0)] = self.opt.algo.discrete_ld.min_shift
        shifts[(shifts > -self.opt.algo.discrete_ld.min_shift) & (shifts < 0)] = -self.opt.algo.discrete_ld.min_shift

        try:
            latent_dim[0]
            latent_dim = list(latent_dim)
        except Exception:
            latent_dim = [latent_dim]
        z_shift = torch.zeros([self.opt.algo.discrete_ld.batch_size] + latent_dim).cuda()
        for i, (index, val) in enumerate(zip(target_indices, shifts)):
            z_shift[i][index] += val

        return target_indices, shifts, z_shift

    def get_latent_triplets(self, z, epsilon):

        z_pos = torch.randn(self.opt.algo.discrete_ld.batch_size, self.opt.algo.discrete_ld.latent_dim).cuda()
        epsilon_pos = epsilon
        z_neg = z
        # eps_neg =
        # for i in range
        # epsilon_neg =

    def make_shifts_discrete_ld(self):

        target = torch.randint(0, self.opt.algo.discrete_ld.num_directions, (self.opt.algo.discrete_ld.batch_size,))
        epsilon_neg = []
        for i in range(target.shape[0]):
            target_set = list(range(10))
            target_set.remove(target[i].item())
            epsilon_neg.append(random.choice(target_set))

        epsilon_ref = torch.nn.functional.one_hot(target, num_classes=10).cuda()
        epsilon_ref = epsilon_ref.type(torch.float32)
        epsilon_neg = torch.nn.functional.one_hot(torch.LongTensor(epsilon_neg), num_classes=10).cuda()
        epsilon_neg = epsilon_neg.type(torch.float32)

        return epsilon_ref, epsilon_neg, target
        # directions_count = list(range(self.opt.algo.linear_combo.num_directions)) sampled_directions_batch = [
        # random.sample(directions_count,self.opt.algo.linear_combo.combo_dirs) for x in range(
        # self.opt.algo.linear_combo.batch_size)] ground_truth_idx = torch.Tensor(np.array(
        # sampled_directions_batch)).cuda() selected_directions = torch.zeros((self.opt.algo.linear_combo.batch_size,
        # self.opt.algo.linear_combo.num_directions)).cuda() for idx,nonzero_idx in enumerate(
        # sampled_directions_batch): for i in nonzero_idx: selected_directions[idx][i] = 1
