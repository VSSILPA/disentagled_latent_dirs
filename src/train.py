import os
import random
from utils import *
from models.latent_deformator import normal_projection_stat
import torch.nn as nn
import random
import torch.functional as F
from torch.utils.data.dataset import random_split
from copy import deepcopy
from CustomDataset import NewDataset
from torch.utils.data.dataset import random_split
import torchvision
import torchvision.transforms as transforms
from math import floor


class Trainer(object):

    def __init__(self, config, opt):
        super(Trainer, self).__init__()
        self.config = config
        self.opt = opt
        self.cross_entropy = nn.CrossEntropyLoss()
        self.adversarial_loss = torch.nn.BCELoss()
        self.similarity_loss = nn.TripletMarginLoss()
        self.real_images = self._get_real_data()

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
        disc_opt.zero_grad()

        label_fake = torch.full((self.opt.algo.discrete_ld.batch_size,), 0, dtype=torch.float32).cuda()
        label_real = torch.full((self.opt.algo.discrete_ld.batch_size,), 1, dtype=torch.float32).cuda()

        z = torch.randn(self.opt.algo.discrete_ld.batch_size, generator.z_dim).cuda()
        label = torch.zeros([1, generator.c_dim])
        images = generator(z,label)
        images = torch.clamp(images, -1, 1)
        prob_real, _, _ = discriminator(images.detach().cuda(),label)
        loss_D_real = self.adversarial_loss(prob_real.view(-1), label_real)
        loss_D_real.backward()

        epsilon_ref, pos, neg, targets = self._get_samples()
        postive_images = self.real_images[pos]
        negative_images = self.real_images[neg]
        shift = deformator(epsilon_ref)
        imgs_shifted = generator(z + shift,label)
        imgs_shifted = torch.clamp(imgs_shifted, -1, 1)
        prob_fake_D, _, _ = discriminator(imgs_shifted.detach(),label)

        loss_D_fake = self.adversarial_loss(prob_fake_D.view(-1), label_fake)
        loss_D_fake.backward()

        disc_opt.step()

        generator.zero_grad()
        deformator.zero_grad()
        imgs_final = torch.cat((imgs_shifted,postive_images.cuda(),negative_images.cuda()),dim=0)

        prob_fake, logits, similarity = discriminator(imgs_final,label)
        # logits, z_rec = shift_predictor(imgs_shifted)

        loss_G = self.adversarial_loss(prob_fake.view(-1)[:self.opt.algo.discrete_ld.batch_size], label_real)
        loss = loss_G + 0.1*self.cross_entropy(logits[:self.opt.algo.discrete_ld.batch_size], torch.LongTensor(targets).cuda())\
               + 0.1*self.similarity_loss(
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

    def _get_samples(self):

        anchor = list(range(100))
        neg = []
        negative_idx = []
        positive_idx = []
        for i in range(10):
            anchor_list = list(range(i * 10, (i + 1) * 10))
            random.shuffle(anchor_list)
            positive_idx = positive_idx + anchor_list
            neg = random.choices(list(set(anchor) - set(anchor_list)), k=10)
            negative_idx = negative_idx + neg
        idx = random.choices(list(range(100)), k=self.opt.algo.discrete_ld.batch_size)
        target = [floor(anchor[i] / 10) for i in idx]
        epsilon_ref = torch.nn.functional.one_hot(torch.LongTensor(target), num_classes=10).cuda()
        epsilon_ref = epsilon_ref.type(torch.float32)
        pos = [positive_idx[i] for i in idx]
        neg = [negative_idx[i] for i in idx]
        return epsilon_ref, pos, neg, target

    def _get_real_data(self):
        data_dir = os.path.join(os.getcwd(), 'data')
        os.makedirs(data_dir, exist_ok=True)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])
        train_dataset = torchvision.datasets.CIFAR10(root=f'{data_dir}/', download=True, train=True, transform=transform)

        temp_list_data = [train_dataset[i][0] for i in range(len(train_dataset))]
        temp_list_data = torch.stack(temp_list_data)

        temp_list_labels = [train_dataset.targets[i] for i in range(len(train_dataset))]
        temp_list_labels = torch.LongTensor(temp_list_labels)

        train_dataset = NewDataset(temp_list_data, temp_list_labels)
        split_data = random_split(train_dataset, [40000, 10000])

        temp_train_dataset = deepcopy(split_data[0])
        validation_dataset = deepcopy(split_data[1])

        train_idx = temp_train_dataset.indices
        train_dataset.data = train_dataset.data[train_idx]
        train_dataset.targets = train_dataset.targets[train_idx]

        numpy_labels = np.asarray(train_dataset.targets)
        sort_labels = np.sort(numpy_labels)
        sort_index = np.argsort(numpy_labels)
        unique, start_index = np.unique(sort_labels, return_index=True)
        training_index = []
        for s in start_index:
            for i in range(10):
                training_index.append(sort_index[s + i])

        real_images = torch.stack([temp_train_dataset[i][0] for i in training_index])
        return real_images

        # directions_count = list(range(self.opt.algo.linear_combo.num_directions)) sampled_directions_batch = [
        # random.sample(directions_count,self.opt.algo.linear_combo.combo_dirs) for x in range(
        # self.opt.algo.linear_combo.batch_size)] ground_truth_idx = torch.Tensor(np.array(
        # sampled_directions_batch)).cuda() selected_directions = torch.zeros((self.opt.algo.linear_combo.batch_size,
        # self.opt.algo.linear_combo.num_directions)).cuda() for idx,nonzero_idx in enumerate(
        # sampled_directions_batch): for i in nonzero_idx: selected_directions[idx][i] = 1
