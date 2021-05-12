import numpy as np
import torch
import torchvision
import os
import matplotlib.pyplot as plt
from utils import cartesian_product


class MPI3D(object):
	def __init__(self, config):
		self.config = config
		self.exp_name = config['experiment_name']
		os.chdir("..")
		path  = os.getcwd()+"/data/mpi3d_toy/mpi3d_toy.npy"
		self.images = np.load(path,mmap_mode='r')
		self.labels = cartesian_product(
			np.arange(6),
			np.arange(6),
			np.arange(2),
			np.arange(3),
			np.arange(3),
			np.arange(40),
			np.arange(40),
		)
		self.num_factors = 7
		self.images = np.transpose(self.images, [0, 3, 1, 2])
		self.latents_sizes = np.array([6,6,2,3,3,40,40])
		self.latent_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))

	def show_images_grid(self, nrows=10):
		path = os.getcwd() + f'/results/{self.exp_name}' + '/visualisations/input.jpeg'
		index = np.random.choice(self.images.shape[0], nrows * nrows, replace=False)
		batch_tensor = torch.from_numpy(self.images[index])
		grid_img = torchvision.utils.make_grid(batch_tensor.reshape(-1, 3, 64, 64), nrow=10, padding=5, pad_value=1)
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
		imgs_sampled = 2 * (imgs_sampled/255) - 1
		return imgs_sampled

	def latent_to_index(self, latents):
		return np.dot(latents, self.latent_bases).astype(int)

	# def sample_latent_values(self, latents_sampled):
	# 	indices_sampled = self.latent_to_index(latents_sampled)
	# 	latent_values = self.latents_values[indices_sampled]
	# 	return latent_values
