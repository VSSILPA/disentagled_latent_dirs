import os
import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from utils import *
import itertools
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToPILImage
import io
from PIL import Image
from utils import make_noise
from config import generator_kwargs

matplotlib.use("Agg")


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
    def __init__(self, config, opt):
        self.config = config
        self.experiment_name = config['experiment_name']
        self.opt = opt

    def to_image(self, tensor, adaptive=False):
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
        for shift in np.arange(-shifts_r, shifts_r, shifts_r / shifts_count):
            if deformator is not None:
                z_deformed = z.cuda() + deformator(one_hot(self.opt.algo.ld.directions_count, shift, dim).cuda())
            else:
                z_deformed = z.cuda() + one_hot(z.shape[1:], shift, dim).cuda()
            shifted_image = generator(z_deformed, self.opt.depth, self.opt.alpha)[0].cpu()
            if shift == 0.0 and with_central_border:
                shifted_image = add_border(shifted_image)

            shifted_images.append(shifted_image)
        return torch.stack(shifted_images)

    def make_interpolation_chart(self, step, generator, deformator=None, shift_r=10, shifts_count=5, dims_count=10):

        file_location = self.opt.result_dir + '/visualisations/latent_traversal/'
        if not os.path.exists(file_location):
            os.makedirs(file_location)
        path = file_location + str(step) + '.png'

        deformator.eval()
        generator.eval()

        z = make_noise(self.opt.num_images_grid, generator.latent_size, truncation=self.opt.algo.ld.truncation).cuda()
        imgs = []
        for i in range(dims_count):
            imgs.append(self.interpolate(generator, z, shift_r, shifts_count, i, deformator))

        batch_tensor = torch.stack(imgs).view(-1, 1, 64, 64)

        save_image(batch_tensor.view(-1, 1, 64, 64), path, nrow=10, normalize=True, scale_each=True, pad_value=128,
                   padding=1)

    def fig_to_image(fig):
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        return Image.open(buf)

    def generate_plot_save_results(self, results, plot_type):
        file_location = os.path.dirname(os.getcwd()) + f'/results/{self.experiment_name}' + '/visualisations/plots/'
        if not os.path.exists(file_location):
            os.makedirs(file_location)
        plt.figure()
        for name, values in results.items():
            x_axis = [self.config['logging_freq'] * i for i in range(len(values))]
            plt.plot(x_axis, values, label=name)
        plt.legend(loc="upper right")
        path = file_location + str(plot_type) + '.jpeg'
        plt.savefig(path)

    def plot_generated_images(self, opt, generator):
        z = torch.randn(50, generator.style_dim).cuda()
        w = generator.style(z)
        imgs = generator([w], **generator_kwargs)[0]
        save_image(imgs, opt.result_dir + '/visualisations/generated_images.jpeg', nrow=int(np.sqrt(len(imgs))),
                   normalize=True, scale_each=True, pad_value=128, padding=1)
