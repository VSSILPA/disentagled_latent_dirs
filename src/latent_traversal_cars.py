#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torchvision.transforms import Resize
from utils import *
from models.closedform.utils import load_generator
from torchvision.utils import save_image
import numpy as np
import random
import torch.nn.functional as F
from torchvision.models import resnet18
import torch.nn as nn
import matplotlib.pylab as plt
import torchvision
from torch_tools.visualization import to_image
from torchvision.utils import make_grid
import cv2
from IPython import display
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


# ## Configurations

# In[3]:


random_seed = 1234
set_seed(random_seed)
load_codes = False
algo = 'ortho'
root_dir= '/home/adarsh/PycharmProjects/disentagled_latent_dirs'
result_path = os.path.join(root_dir,  'results/cars/closed_form_ours/qualitative_analysis')


# ## Model Selection

# In[4]:


deformator_path = os.path.join(root_dir, 'pretrained_models/deformators/ClosedForm/stylegan_car512/stylegan_car512.pkl')
layers, cf_deformator, _ = torch.load(deformator_path, map_location='cpu')
cf_deformator = torch.FloatTensor(cf_deformator).cuda()

deformator_path = os.path.join(root_dir, 'results/cars/models/50000_model.pkl')
deformator = torch.load(deformator_path)['deformator']['ortho_mat']
dse_deformator = deformator.T
        
generator = load_generator(None, model_name='stylegan_car512')

if load_codes:
    codes = torch.load(os.path.join(root_dir, 'results/cars/closed_form_ours/quantitative_analysis/z_analysis.pkl'))
else:
    num_samples = 1000
    codes = torch.randn(num_samples, 512).cuda()
    w = generator.mapping(codes)['w']
    codes  = generator.truncation(w, trunc_psi = 0.7, trunc_layers = 8)


# In[5]:


def add_border(tensor):
    border = 3
    for ch in range(tensor.shape[0]):
        color = 1.0 if ch == 0 else -1
        tensor[ch, :border, :] = color
        tensor[ch, -border:,] = color
        tensor[ch, :, :border] = color
        tensor[ch, :, -border:] = color
    return tensor

@torch.no_grad()
def interpolate(G, z, shifts_r, shifts_count, dim, deformator=None, with_central_border=False):
    shifted_images = []
    for shift in np.linspace(-shifts_r,shifts_r,shifts_count):
        shifted_image = G.synthesis(z + (deformator[dim:dim + 1] * shift).unsqueeze(1).repeat(1,len(layers),1))
        if shift == 0.0 and with_central_border:
            shifted_image = add_border(shifted_image)
        shifted_images.append(shifted_image)
    shifted_images = torch.stack(shifted_images).squeeze(dim=1)
    return shifted_images

@torch.no_grad()
def make_interpolation_chart(G, deformator=None, z=None,
                             shifts_r=10.0, shifts_count=5,
                             dims=None, dims_count=10, texts=None, **kwargs):


    original_img = G.synthesis(z).cpu()
    imgs = []
    if dims is None:
        dims = range(dims_count)
    for i in dims:
        imgs.append(interpolate(G, z, shifts_r, shifts_count, i, deformator))

    rows_count = len(imgs) + 1
    fig, axs = plt.subplots(rows_count, **kwargs)

    axs[0].axis('off')
    axs[0].imshow(to_image(original_img, True))
    
    

    if texts is None:
        texts = dims
    for ax, shifts_imgs, text in zip(axs[1:], imgs, texts):
        ax.axis('off')
        plt.subplots_adjust(left=0.5)
        ax.imshow(to_image(make_grid(shifts_imgs.clamp(min=-1, max=1), nrow=(2 * shifts_count + 1),scale_each=True, normalize=True, padding=1), True))
        ax.text(-20, 21, str(text), fontsize=10)


    return fig


@torch.no_grad()
def inspect_all_directions(G, deformator, out_dir, zs=None, num_z=3, shifts_r=8.0):
    os.makedirs(out_dir, exist_ok=True)

    step = 5
    max_dim = 128
    codes = zs.cuda()
    w = G.mapping(codes)['w']
    zs  = G.truncation(w, trunc_psi = 0.7, trunc_layers = 8)
    shifts_count = zs.shape[0]

    for start in range(0, max_dim - 1, step):
        imgs = []
        dims = range(start, min(start + step, max_dim))
        for z in zs:
            z = z.unsqueeze(0)
            fig = make_interpolation_chart(
                G, deformator=deformator, z=z,
                shifts_count=5, dims=dims, shifts_r=shifts_r,
                dpi=250, figsize=(int(shifts_count * 4.0), int(0.5 * step) + 2))
            fig.canvas.draw()
            plt.close(fig)
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # crop borders
            nonzero_columns = np.count_nonzero(img != 255, axis=0)[:, 0] > 0
            img = img.transpose(1, 0, 2)[nonzero_columns].transpose(1, 0, 2)
            imgs.append(img)

        out_file = os.path.join(out_dir, '{}_{}.jpg'.format(dims[0], dims[-1]))
        print('saving chart to {}'.format(out_file))
        Image.fromarray(np.hstack(imgs)).save(out_file)
        
# z = torch.load('codes.pkl').cuda()
z = torch.randn(3, 512)
# z = torch.load(os.path.join(result_path,'temp','code_1.pkl'))
out_dir = '/home/adarsh/PycharmProjects/disentagled_latent_dirs/results/cars/closedform_ours/qualitative_analysis/inspect_all_dirs_dse'
inspect_all_directions(generator, dse_deformator,out_dir,zs=z, shifts_r=2)


# In[ ]:


def postprocess_images(images):
        """Post-processes images from `torch.Tensor` to `numpy.ndarray`."""
        images = images.detach().cpu().numpy()
        images = (images + 1) * 255 / 2
        images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
        images = images.transpose(0, 2, 3, 1)
        return images


def save_images(codes, shifts_r, shifts_count, cf_dir, dse_dir, generator, cf_deformator, dse_deformator):
        plt.figure(figsize=(30,30))
        for idx, z in enumerate(codes):
            print('Figure : ' + str(idx))
            z_shift_cf = []
            z_shift_dse = []
            for i, shift in enumerate(np.linspace(-shifts_r,shifts_r,shifts_count)):
                z_shift_cf.append(z + cf_deformator[cf_dir: cf_dir + 1] * shift)
                z_shift_dse.append(z + dse_deformator[dse_dir: dse_dir + 1] * shift)
            z_shift_cf = torch.stack(z_shift_cf).squeeze(dim=1)
            z_shift_dse = torch.stack(z_shift_dse).squeeze(dim=1)
            with torch.no_grad():
                cf_images= generator(z_shift_cf)
            torch.save(cf_images, os.path.join(result_path, 'temp', 'cf.pkl'))
            del cf_images
            with torch.no_grad():
                dse_images= generator(z_shift_dse)
            torch.save(dse_images, os.path.join(result_path, 'temp', 'dse.pkl'))
            del dse_images
            cf_images = torch.load(os.path.join(result_path, 'temp', 'cf.pkl'))
            dse_images = torch.load(os.path.join(result_path, 'temp', 'dse.pkl'))
            all_images = torch.cat((cf_images, dse_images), dim=0)
            grid = torchvision.utils.make_grid(all_images.clamp(min=-1, max=1),nrow=10, scale_each=True, normalize=True)
            display.display(plt.gcf())
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            del all_images
            del cf_images
            del dse_images
            del grid

    
z_min_index = 0
z_max_index = 5
cf_dir = 1
dse_dir = 1
shift_r = 10
shift_count = 10
all_images = save_images(codes[z_min_index:z_max_index], shift_r, shift_count, cf_dir, dse_dir, generator, cf_deformator, dse_deformator)                    

