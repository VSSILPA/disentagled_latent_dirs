import numpy as np
import torch
import torchvision
import os
import matplotlib.pyplot as plt
from utils import cartesian_product


class Cars3D(object):
    def __init__(self, config, opt):
        self.config = config
        self.opt = opt
        self.exp_name = config['experiment_name']
        self.images = np.load('../data/cars/car3d.npy')
        self.images = self.images.reshape(-1, 3, 64, 64)*255  # data is range of [0,255]
        self.labels = cartesian_product(np.arange(183),
                                        np.arange(24),
                                        np.arange(4))
        self.num_factors = 3
        self.latents_sizes = np.array([183, 24, 4])
        self.latent_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))
        self.show_images_grid()

    def show_images_grid(self, nrows=10):
        file_location = self.opt.result_dir + '/visualisations'
        index = np.random.choice(self.images.shape[0], nrows * nrows, replace=False)
        batch_tensor = torch.from_numpy(self.images[index])/255
        grid_img = torchvision.utils.make_grid(batch_tensor.reshape(-1, 3, 64, 64), nrow=10, padding=5, pad_value=1)
        grid = grid_img.permute(1, 2, 0).type(torch.FloatTensor)
        plt.imsave(file_location + '/input.jpeg', grid.numpy())

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
        return np.dot(latents, self.latent_bases).astype(int)
