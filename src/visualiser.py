import os
import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from utils import *
matplotlib.use("Agg")
import itertools
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import io
from PIL import Image


def add_border(tensor):
	border = 3
	for ch in range(tensor.shape[0]):
		color = 1.0 if ch == 0 else -1
		tensor[ch, :border, :] = color
		tensor[ch, -border:, ] = color
		tensor[ch, :, :border] = color
		tensor[ch, :, -border:] = color
	return tensor

class Visualiser(object):
	def __init__(self, config):
		self.config = config
		self.experiment_name = config['experiment_name']

	def to_image(self,tensor, adaptive=False):
		if len(tensor.shape) == 4:
			tensor = tensor[0]
		if adaptive:
			tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
			return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))
		else:
			tensor = (tensor + 1) / 2
			tensor.clamp(0, 1)
			return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))

	@torch.no_grad()
	def interpolate(self, generator, z, shifts_r, shifts_count, dim, deformator=None, with_central_border=False):
		shifted_images = []
		for shift in np.arange(-shifts_r, shifts_r + 1e-9, shifts_r / shifts_count):
			if deformator is not None:
				z_deformed = z.cuda() + deformator(one_hot(z.shape[1:], shift, dim).cuda())
			else:
				z_deformed = z.cuda() + one_hot(z.shape[1:], shift, dim).cuda()
			shifted_image = generator(z_deformed).cpu()[0]
			if shift == 0.0 and with_central_border:
				shifted_image = add_border(shifted_image)

			shifted_images.append(shifted_image)
		return shifted_images


	def make_interpolation_chart(self, step ,generator, deformator=None, z=None,
								 shift_r=10, shifts_count = 5, dims = None, dims_count = 10, texts = None, **kwargs):

		file_location = os.path.dirname(os.getcwd()) + f'/results/{self.experiment_name}' + '/visualisations/latent_traversal/'
		if not os.path.exists(file_location):
			os.makedirs(file_location)
		path = file_location + str(step) + '.png'

		if deformator is not None:
			deformator.eval()

		z = z if z is not None else make_noise(1, generator.dim_z)

		original_img = generator(z.cuda()).cpu()

		imgs = []
		if dims is None:
			dims = range(dims_count)
		for i in dims:
			imgs.append(self.interpolate(generator, z, shift_r, shifts_count, i, deformator))

		rows_count = len(imgs) + 1
		fig, axs = plt.subplots(rows_count, **kwargs)

		axs[0].axis('off')
		axs[0].imshow(self.to_image(original_img, True))

		if texts is None:
			texts = dims
		for ax, shifts_imgs, text in zip(axs[1:], imgs, texts):
			ax.axis('off')
			plt.subplots_adjust(left=0.5)
			ax.imshow(self.to_image(make_grid(shifts_imgs, nrow=(2 * shifts_count + 1), padding=1), True))
			ax.text(-20, 21, str(text), fontsize=10)
		plt.savefig(path)

	def fig_to_image(fig):
		buf = io.BytesIO()
		fig.savefig(buf)
		buf.seek(0)
		return Image.open(buf)


	def generate_plot_save_results(self, results, plot_type):
		file_location = os.path.dirname(os.getcwd())+ f'/results/{self.experiment_name}' + '/visualisations/plots/'
		if not os.path.exists(file_location):
			os.makedirs(file_location)
		plt.figure()
		for name, values in results.items():
			x_axis = [self.config['logging_freq'] * i for i in range(len(values))]
			plt.plot(x_axis, values, label=name)
		plt.legend(loc="upper right")
		path = file_location + str(plot_type) + '.jpeg'
		plt.savefig(path)










