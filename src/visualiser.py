import matplotlib
import matplotlib.pyplot as plt
from utils import *
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToPILImage
import io
from PIL import Image
from config import generator_kwargs

from torch_tools.visualization import to_image

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


def plot_generated_images(opt, generator):
    z = torch.randn(50, generator.style_dim).cuda()
    w = generator.style(z)
    imgs = generator([w], **generator_kwargs)[0]
    save_image(imgs, opt.result_dir + '/visualisations/generated_images.jpeg', nrow=int(np.sqrt(len(imgs))),
               normalize=True, scale_each=True, pad_value=128, padding=1)


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)


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
    def interpolate(self, generator, z, shifts_r, shifts_count, dim, directions, with_central_border=True):
        shifted_images = []
        directions.cuda()
        for shift in np.arange(-shifts_r, shifts_r, shifts_r / shifts_count): # TODO change traversal in z space to wspace
            if directions is not None:
                latent_shift = directions(one_hot(directions.input_dim, shift, dim).cuda())
            else:
                latent_shift = one_hot(z.shape[1:], shift, dim).cuda()
            w = generator.style(z.cuda())
            shifted_image = generator([w+latent_shift], **generator_kwargs)[0]
            if shift == 0.0 and with_central_border:
                shifted_image = add_border(shifted_image)
            shifted_images.append(shifted_image)
        return torch.stack(shifted_images)

    @torch.no_grad()
    def make_interpolation_chart(self, step, generator, directions, shift_r=10, shifts_count=5,texts=None,**kwargs):

        file_location = self.opt.result_dir + '/visualisations/latent_traversal/'
        if not os.path.exists(file_location):
            os.makedirs(file_location)

        directions.eval()
        generator.eval()

        z = torch.randn(1, generator.style_dim)
        w = generator.style(z.cuda())
        original_img = generator([w], **generator_kwargs)[0]
        imgs = []
        if self.opt.algorithm == 'LD':
            num_directions = self.opt.algo.ld.num_directions
        elif self.opt.algorithm == 'linear_combo':
            num_directions = self.opt.algo.linear_combo.num_directions
        elif self.opt.algorithm == 'GS':
            num_directions = self.opt.algo.gs.num_directions
        elif self.opt.algorithm == 'CF' :
            num_directions = self.opt.algo.cf.num_directions
        elif self.opt.algorithm == 'ours' or 'ours-natural':
            num_directions = self.opt.algo.ours.num_directions
        num_directions = [6,15,24,49,50,53,57,58,69,70,71,78]

        for i in num_directions:
            imgs.append(self.interpolate(generator, z, shift_r, shifts_count, i, directions))

        rows_count = len(imgs) + 1
        fig, axs = plt.subplots(rows_count, **kwargs)

        axs[0].axis('off')
        axs[0].imshow(to_image(original_img, True))

        if texts is None:
            texts =num_directions
        for ax, shifts_imgs, text in zip(axs[1:], imgs, texts):
            ax.axis('off')
            plt.subplots_adjust(left=0.5)
            ax.imshow(to_image(make_grid(shifts_imgs.squeeze(1), nrow=(2 * shifts_count + 1), padding=1), True))
            ax.text(-20, 21, str(text), fontsize=10)
        fig_to_image(fig).convert("RGB").save(
            os.path.join(file_location, '{}_{}.jpg'.format('fixed', step)))

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
