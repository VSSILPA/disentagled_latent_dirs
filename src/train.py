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
from reassigned_dataset import ReassignedDataset
import faiss
import time


class Trainer(object):

    def __init__(self, config, opt):
        super(Trainer, self).__init__()
        self.config = config
        self.opt = opt
        self.cross_entropy = nn.CrossEntropyLoss()
        self.adversarial_loss = torch.nn.BCELoss()
        # self.similarity_loss = nn.TripletMarginLoss()
        # self.real_images = self._get_real_data()
        self.n = 50000
        self.k = 10

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
                          shift_predictor_opt):

        image_z = []
        features_list = []

        generator.eval()
        shift_predictor.eval()
        for i in range(500):
            z = torch.randn(self.opt.algo.discrete_ld.batch_size, generator.z_dim).cuda()
            w = generator.mapping(z, 0)
            images = generator.synthesis(w)
            images = torch.clamp(images, -1, 1)
            _, _, features = shift_predictor(images)
            image_z.append(z)
            features_list.append(features.data.cpu())
        images_z = torch.stack(image_z).view(-1, 512)
        features = torch.stack(features_list).view(-1, 512).numpy()

        clustering_loss = self.cluster(features, False)
        del features
        del images
        del features_list
        data = []
        labels_list = []
        for i, labels in enumerate(self.images_lists):
            data.append(images_z[labels])
            labels_list.append([i]*len(labels))

        labels = [item for sublist in labels_list for item in sublist]
        train_data = [item for sublist in data for item in sublist]
        dataset = NewDataset(torch.stack(train_data),torch.stack(labels),generator,transform=)

        shift_predictor.type_estimator.weight.data.normal_(0, 0.01)
        shift_predictor.type_estimator.bias.data.zero_()

        shift_predictor.train()
        optimizer_tl = torch.optim.SGD(shift_predictor.type_estimator.parameters(), lr=0.05, weight_decay=10 ** -5,)




        return deformator, discriminator, disc_opt, shift_predictor, deformator_opt, shift_predictor_opt, (0, 0, 0)

    def run_kmeans(self,x, nmb_clusters, verbose=False):
        """Runs kmeans on 1 GPU.
        Args:
            x: data
            nmb_clusters (int): number of clusters
        Returns:
            list: ids of data in each cluster
        """
        n_data, d = x.shape

        # faiss implementation of k-means
        clus = faiss.Clustering(d, nmb_clusters)

        # Change faiss seed at each k-means so that the randomly picked
        # initialization centroids do not correspond to the same feature ids
        # from an epoch to another.
        clus.seed = np.random.randint(1234)

        clus.niter = 20
        clus.max_points_per_centroid = 10000000
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        flat_config.device = 0
        index = faiss.GpuIndexFlatL2(res, d, flat_config)

        # perform the training
        clus.train(x, index)
        _, I = index.search(x, 1)
        stats = clus.iteration_stats
        losses = np.array([
            stats.at(i).obj for i in range(stats.size())
        ])
        if verbose:
            print('k-means loss evolution: {0}'.format(losses))

        return [int(n[0]) for n in I], losses[-1]


    def cluster(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = self.preprocess_features(data)

        # cluster the data
        I, loss = self.run_kmeans(xb,10 , False)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss

    def preprocess_features(self,npdata, pca=256):
        """Preprocess an array of features.
        Args:
            npdata (np.array N * ndim): features to preprocess
            pca (int): dim of output
        Returns:
            np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
        """
        _, ndim = npdata.shape
        npdata = npdata.astype('float32')

        # Apply PCA-whitening with Faiss
        mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
        mat.train(npdata)
        assert mat.is_trained
        npdata = mat.apply_py(npdata)

        # L2 normalization
        row_sums = np.linalg.norm(npdata, axis=1)
        npdata = npdata / row_sums[:, np.newaxis]

        return npdata

    def cluster_assign(self,images_lists, dataset):
        """Creates a dataset from clustering, with clusters as labels.
        Args:
            images_lists (list of list): for each cluster, the list of image indexes
                                        belonging to this cluster
            dataset (list): initial dataset
        Returns:
            ReassignedDataset(torch.utils.data.Dataset): a dataset with clusters as
                                                         labels
        """
        assert images_lists is not None
        pseudolabels = []
        image_indexes = []
        for cluster, images in enumerate(images_lists):
            image_indexes.extend(images)
            pseudolabels.extend([cluster] * len(images))

        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        t = transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize])

        return ReassignedDataset(image_indexes, pseudolabels, dataset, t)

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
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        train_dataset = torchvision.datasets.CIFAR10(root=f'{data_dir}/', download=True, train=True,
                                                     transform=transform)

        temp_list_data = [train_dataset[i][0] for i in range(len(train_dataset))]
        temp_list_data = torch.stack(temp_list_data)

        temp_list_labels = [train_dataset.targets[i] for i in range(len(train_dataset))]
        temp_list_labels = torch.LongTensor(temp_list_labels)

        train_dataset = NewDataset(temp_list_data, temp_list_labels)
        split_data = random_split(train_dataset, [10000, 40000])

        temp_train_dataset = deepcopy(split_data[0])

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
