import numpy as np
import torch
import torchvision
import os
import matplotlib.pyplot as plt
from utils import cartesian_product
from scipy.io import loadmat
import numpy as np
from torchvision.transforms import Resize

class Cars3D(object):
	def __init__(self, config):
		self.config = config
		self.exp_name = config['experiment_name']
		os.chdir("..")
		path  = os.getcwd()+"/data/cars"
		paths = np.loadtxt(path+'/list.txt', dtype="str")
		paths = [os.path.join(path, "{}.mat".format(each)) for each in paths]
		self.images = np.load('../data/cars/car3d.npy')
		self.images = self.images.reshape(-1,3,64,64)
		self.labels = cartesian_product(np.arange(183), np.arange(24), np.arange(4))
		self.latents_sizes = np.array([183,24,4])
		self.latent_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))

	def show_images_grid(self, nrows=10):
		path = os.getcwd() + f'/results/{self.exp_name}' + '/visualisations/input.jpeg'
		index = np.random.choice(self.images.shape[0], nrows * nrows, replace=False)
		batch_tensor = torch.from_numpy(self.images[index]/255)
		grid_img = torchvision.utils.make_grid(batch_tensor.reshape(-1, 3, 128, 128), nrow=10, padding=5, pad_value=1)
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
		imgs_sampled = 2 * (imgs_sampled / 255.0) - 1
		return imgs_sampled

	def latent_to_index(self, latents):
		return np.dot(latents, self.latent_bases).astype(int)

	# def sample_latent_values(self, latents_sampled):
	# 	indices_sampled = self.latent_to_index(latents_sampled)
	# 	latent_values = self.latents_values[indices_sampled]
	# 	return latent_values
