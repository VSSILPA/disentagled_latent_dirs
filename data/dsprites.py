import numpy as np
import torch
import torchvision
import os
from utils import cartesian_product
import matplotlib.pyplot as plt

SCREAM_PATH = "/home/adarsh/Documents/data/scream/scream.jpg"
dsprites_path = "../data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
npy_path = '/home/adarsh/PycharmProjects/disentagled_latent_dirs/data/dsprites'


class DSprites(object):
    def __init__(self, config, opt):
        self.config = config
        self.opt = opt
        self.exp_name = config['experiment_name']
        self.images = np.load(npy_path + '/imgs.npy', mmap_mode='r+', encoding="latin1", allow_pickle=True)
        self.images = self.images.reshape(-1, 1, 64, 64)*255  # data in range of [0,255]
        self.labels = cartesian_product(np.arange(3),
                                        np.arange(6),
                                        np.arange(40),
                                        np.arange(32),
                                        np.arange(32))
        self.num_factors = 5
        self.latents_sizes = np.array([3, 6, 40, 32, 32])
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))
        self.show_images_grid()

    def show_images_grid(self, nrows=10):
        path = self.opt.result_dir + '/visualisations/input.jpeg'
        index = np.random.choice(self.images.shape[0], nrows * nrows, replace=False)
        batch_tensor = torch.from_numpy(self.images[index])
        grid_img = torchvision.utils.make_grid(batch_tensor.view(-1, 1, 64, 64), nrow=10, padding=5, pad_value=1)
        grid = grid_img.permute(1, 2, 0).type(torch.FloatTensor)
        plt.imsave(path, grid.numpy())

    def sample_latent(self, size=1):
        """
        Generate a vector with size of ground truth factors and random fill each column with values from range
        :param size:
        :return: latents
        """

        samples = np.zeros((size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)
        return samples

    def sample_images_from_latent(self, latent):
        indices_sampled = self.latent_to_index(latent)
        imgs_sampled = self.images[indices_sampled]
        imgs_sampled = 2 * (imgs_sampled/255) - 1  # normalising data to -1,1
        return imgs_sampled

    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)
