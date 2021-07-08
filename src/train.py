import random
from utils import *
from models.latent_deformator import normal_projection_stat
import torch.nn as nn
from config import generator_kwargs
import PIL
import Augmentor
import torchvision
from torchvision import transforms


class Trainer(object):

    def __init__(self, config, opt):
        super(Trainer, self).__init__()
        self.config = config
        self.opt = opt
        self.cross_entropy = nn.CrossEntropyLoss()
        self.ranking_loss = nn.BCEWithLogitsLoss()
        p = Augmentor.Pipeline()
        p.rotate(probability=1, max_left_rotation=20, max_right_rotation=20)
        p.zoom(probability=1, min_factor=0.9, max_factor=1.1)  # TODO PAY ATTENTION to zoom factor
        p.random_distortion(probability=1, grid_width=1, grid_height=1, magnitude=10)
        self.torch_transform = torchvision.transforms.Compose(
            [transforms.ToPILImage(), p.torch_transform(), transforms.ToTensor()])
        self.y_real_, self.y_fake_ = torch.ones(self.opt.batch_size, 1, device="cuda"), torch.zeros(self.opt.batch_size,
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

    def train_ours(self, generator, deformator, deformator_opt, cr_discriminator, cr_optimizer):

        generator.zero_grad()
        deformator.zero_grad()

        z_ = torch.randn(int(self.opt.algo.ours.batch_size / 2), generator.dim_z[0], generator.dim_z[1],
                         generator.dim_z[2]).cuda()
        z = torch.cat((z_, z_), dim=0)

        epsilon, ground_truths = self.make_shifts_rank()
        shift_epsilon = deformator(epsilon)
        shift_epsilon = shift_epsilon.unsqueeze(2)
        shift_epsilon = shift_epsilon.unsqueeze(3)
        imgs, _ = generator(z + shift_epsilon)
        logits, identity = cr_discriminator(imgs.detach())

        epsilon1, epsilon2 = torch.split(logits, int(self.opt.algo.ours.batch_size / 2))
        identity1, identity2 = torch.split(identity, int(self.opt.algo.ours.batch_size / 2))
        epsilon_diff = epsilon1 - epsilon2
        identity_diff = identity1 - identity2
        ranking_loss = self.ranking_loss(epsilon_diff, ground_truths)
        identity_loss = self.ranking_loss(identity_diff, self.y_real_)
        loss = ranking_loss + identity_loss
        loss.backward()
        cr_optimizer.step()

        # augmentation based training
        cr_optimizer.zero_grad()
        augmented_image = self.torch_transform(imgs[:int(self.opt.algo.ours.batch_size / 2)].detach())
        real_aug = torch.cat((imgs[:int(self.opt.algo.ours.batch_size / 2)].detach(), augmented_image), dim=0).cuda()
        _, identity = cr_discriminator(real_aug.detach())
        identity1, identity2 = torch.split(identity, int(self.opt.algo.ours.batch_size / 2))
        identity_diff = identity1 - identity2 ##real -augmented
        ranking_loss = self.ranking_loss(identity_diff, self.y_fake_)
        ranking_loss.backward()
        cr_optimizer.step()
        del real_aug
        del augmented_image

        generator.zero_grad()
        deformator.zero_grad()

        z_ = torch.randn(int(self.opt.algo.ours.batch_size / 2), generator.dim_z[0], generator.dim_z[1],
                         generator.dim_z[2]).cuda()
        z = torch.cat((z_, z_), dim=0)
        epsilon, ground_truths = self.make_shifts_rank()
        shift_epsilon = deformator(epsilon)
        shift_epsilon = shift_epsilon.unsqueeze(2)
        shift_epsilon = shift_epsilon.unsqueeze(3)
        z_total = torch.cat((z+shift_epsilon, z), dim=0)

        imgs, _ = generator(z_total)
        logits, identity = cr_discriminator(imgs)

        logits = logits[:self.opt.algo.ours.batch_size]
        identity1, identity2 = torch.split(identity, int(self.opt.algo.ours.batch_size / 2))
        epsilon1, epsilon2 = torch.split(logits, int(self.opt.algo.ours.batch_size / 2))
        epsilon_diff = epsilon1 - epsilon2
        identity_diff = identity2 - identity1 ## z - (z+shift_epsilon)
        ranking_loss = self.ranking_loss(epsilon_diff, ground_truths) + 0.5*self.ranking_loss(identity_diff, self.y_fake_)

        ranking_loss.backward()

        deformator_opt.step()

        return deformator, deformator_opt, cr_discriminator, cr_optimizer, ranking_loss.item()

    def train_latent_discovery(self, generator, deformator, shift_predictor, cr_discriminator, cr_optimizer,
                               deformator_opt,
                               shift_predictor_opt):

        generator.zero_grad()
        deformator.zero_grad()
        shift_predictor.zero_grad()

        z_ = torch.randn(int(self.opt.algo.linear_combo.batch_size / 2), generator.style_dim).cuda()
        z = torch.cat((z_, z_), dim=0)

        epsilon, ground_truths = self.make_shifts_rank()
        shift_epsilon = deformator(epsilon)

        w = generator.style(z)

        imgs, _ = generator([w + shift_epsilon], **generator_kwargs)
        logits = cr_discriminator(imgs.detach())

        epsilon1, epsilon2 = torch.split(logits, int(self.opt.algo.linear_combo.batch_size / 2))
        epsilon_diff = epsilon1 - epsilon2
        ranking_loss = self.ranking_loss(epsilon_diff, ground_truths)
        ranking_loss.backward()
        cr_optimizer.step()

        generator.zero_grad()
        deformator.zero_grad()
        shift_predictor.zero_grad()

        z = torch.randn(self.opt.algo.linear_combo.batch_size, generator.style_dim).cuda()
        epsilon = self.make_shifts_linear_combo()

        shift = deformator(epsilon)
        w = generator.style(z)
        imgs, _ = generator([w], **generator_kwargs)
        imgs_shifted, _ = generator([w + shift], **generator_kwargs)

        _, shift_prediction = shift_predictor(imgs, imgs_shifted)
        shift_loss = torch.mean(torch.abs(shift_prediction - epsilon))

        z_ = torch.randn(int(self.opt.algo.linear_combo.batch_size / 2), generator.style_dim).cuda()
        z = torch.cat((z_, z_), dim=0)
        epsilon, ground_truths = self.make_shifts_rank()
        shift_epsilon = deformator(epsilon)

        w = generator.style(z)

        imgs, _ = generator([w + shift_epsilon], **generator_kwargs)
        logits = cr_discriminator(imgs)

        epsilon1, epsilon2 = torch.split(logits, int(self.opt.algo.linear_combo.batch_size / 2))
        epsilon_diff = epsilon1 - epsilon2
        ranking_loss = self.ranking_loss(epsilon_diff, ground_truths)

        loss = ranking_loss + shift_loss

        loss.backward()

        deformator_opt.step()
        shift_predictor_opt.step()

        return deformator, shift_predictor, cr_discriminator, cr_optimizer, deformator_opt, shift_predictor_opt, (
            0, 0, loss.item())

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

    def make_shifts_rank(self):

        epsilon = torch.FloatTensor(int(self.opt.algo.ours.batch_size),
                                    self.opt.algo.ours.num_directions).uniform_(-1, 1).cuda()

        epsilon_1, epsilon_2 = torch.split(epsilon, int(self.opt.algo.ours.batch_size / 2))
        ground_truths = (epsilon_1 > epsilon_2).type(torch.float32).cuda()
        epsilon = torch.cat((epsilon_1, epsilon_2), dim=0)
        return epsilon, ground_truths

    def make_shifts_linear_combo(self):

        # directions_count = list(range(self.opt.algo.linear_combo.num_directions)) sampled_directions_batch = [
        # random.sample(directions_count,self.opt.algo.linear_combo.combo_dirs) for x in range(
        # self.opt.algo.linear_combo.batch_size)] ground_truth_idx = torch.Tensor(np.array(
        # sampled_directions_batch)).cuda() selected_directions = torch.zeros((self.opt.algo.linear_combo.batch_size,
        # self.opt.algo.linear_combo.num_directions)).cuda() for idx,nonzero_idx in enumerate(
        # sampled_directions_batch): for i in nonzero_idx: selected_directions[idx][i] = 1
        epsilon = torch.FloatTensor(self.opt.algo.linear_combo.batch_size,
                                    self.opt.algo.linear_combo.num_directions).uniform_(-1, 1).cuda()
        # z_shift = selected_directions * epsilon z_shift[(z_shift < self.opt.algo.linear_combo.min_shift) & (z_shift
        # > 0)] = self.opt.algo.linear_combo.min_shift z_shift[(z_shift > -self.opt.algo.linear_combo.min_shift) & (
        # z_shift < 0)] = -self.opt.algo.linear_combo.min_shift ground_truths = z_shift.gather(dim=1,
        # index = ground_truth_idx.long())

        return epsilon
