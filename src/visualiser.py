import matplotlib
import matplotlib.pyplot as plt
from utils import *
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToPILImage
import io
from PIL import Image
import torchvision
from torch.autograd import Variable
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
    def interpolate(self, generator,z, shifts_r, shifts_count, dim, directions, with_central_border=False):
        n_cols = 10
        z = z[dim*10:(dim+1)*10]
        directions.cuda()
        dir_required = directions.weight[:,dim]
        dir_required = dir_required.repeat(n_cols, 1)
        z_deformed = z.cuda() + dir_required.cuda()
        images = generator(z_deformed)

        return images

    def make_interpolation_chart(self, step, z,generator, directions, shift_r=10, shifts_count=5):

        file_location = self.opt.result_dir + '/visualisations/latent_traversal/'
        if not os.path.exists(file_location):
            os.makedirs(file_location)
        path = file_location + str(step) + '.png'

        directions.eval()
        generator.eval()

        imgs = []
        if self.opt.algorithm == 'LD':
            num_directions = self.opt.algo.ld.num_directions
        elif self.opt.algorithm == 'discrete_ld':
            num_directions = self.opt.algo.discrete_ld.num_directions
        elif self.opt.algorithm =='GS':
            num_directions = self.opt.algo.gs.num_directions
        elif self.opt.algorithm == 'CF' :
            num_directions = self.opt.algo.cf.num_directions
        for i in range(num_directions):
            imgs.append(self.interpolate(generator, z, shift_r, shifts_count, i, directions))

        batch_tensor = torch.stack(imgs).view(-1, self.opt.num_channels, self.opt.image_size, self.opt.image_size)
        batch_tensor = torch.clamp(batch_tensor, -1, 1)

        save_image(batch_tensor.view(-1, self.opt.num_channels, self.opt.image_size, self.opt.image_size), path, nrow=10, normalize=True, scale_each=True, pad_value=128,
                   padding=1)

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

    def plot_infogan_grid(self, model):
        z_dict = model.get_z(10 * 100, sequential=True)
        gan_input = torch.cat([z_dict[k] for k in z_dict.keys()], dim=1)
        gan_input = Variable(gan_input, requires_grad=True).cuda()
        imgs = model.gen(gan_input)
        grid_img = torchvision.utils.make_grid(imgs[:100], nrow=10, normalize=True)
        plt.imshow(grid_img.permute(1, 2, 0).cpu().data)
        plt.savefig(self.opt.result_dir+'/visualisations/gan_generate_grid.png')


