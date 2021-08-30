
# coding: utf-8

# In[ ]:


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
get_ipython().magic('matplotlib inline')


# In[ ]:


def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


# ## Configurations

# In[ ]:


random_seed = 12
# set_seed(random_seed)
load_codes = False
algo = 'ortho'
root_dir= '/home/adarsh/PycharmProjects/disentagled_latent_dirs'
result_path = os.path.join(root_dir,  'results/cars/closedform_ours/qualitative_analysis')


# ## Model Selection

# In[ ]:


deformator_path = os.path.join(root_dir, 'pretrained_models/deformators/ClosedForm/stylegan_car512/stylegan_car512.pkl')
layers, cf_deformator, _ = torch.load(deformator_path, map_location='cpu')
cf_deformator = torch.FloatTensor(cf_deformator).cuda()

model_num = 60000

# loading ours
deformator_path = os.path.join(root_dir, 'results/cars/closedform_ours/models/'+str(model_num)+'_model.pkl')
deformator = torch.load(deformator_path)['deformator']['ortho_mat']
dse_deformator = deformator.T

# dse_deformator = cf_deformator

        
generator = load_generator(None, model_name='stylegan_car512')

if load_codes:
    codes = torch.load(os.path.join(root_dir, 'results/cars/closedform_ours/quantitative_analysis/z_analysis.pkl'))
else:
    num_samples = 1000
    codes = torch.randn(num_samples, 512).cuda()
    w = generator.mapping(codes)['w']
    codes  = generator.truncation(w, trunc_psi = 0.7, trunc_layers = 8)


# In[ ]:


# def add_border(tensor):
#     border = 3
#     for ch in range(tensor.shape[0]):
#         color = 1.0 if ch == 0 else -1
#         tensor[ch, :border, :] = color
#         tensor[ch, -border:,] = color
#         tensor[ch, :, :border] = color
#         tensor[ch, :, -border:] = color
#     return tensor

# @torch.no_grad()
# def interpolate(G, z, shifts_r, shifts_count, dim, deformator=None, with_central_border=False):
#     shifted_images = []
#     for shift in np.linspace(-shifts_r,shifts_r,shifts_count):
#         shifted_image = G.synthesis(z + (deformator[dim:dim + 1] * shift).unsqueeze(1).repeat(1,len(layers),1))
#         if shift == 0.0 and with_central_border:
#             shifted_image = add_border(shifted_image)
#         shifted_images.append(shifted_image)
#     shifted_images = torch.stack(shifted_images).squeeze(dim=1)
#     return shifted_images

# @torch.no_grad()
# def make_interpolation_chart(G, deformator=None, z=None,
#                              shifts_r=10.0, shifts_count=5,
#                              dims=None, dims_count=10, texts=None, **kwargs):


#     original_img = G.synthesis(z).cpu()
#     imgs = []
#     if dims is None:
#         dims = range(dims_count)
#     for i in dims:
#         imgs.append(interpolate(G, z, shifts_r, shifts_count, i, deformator))

#     rows_count = len(imgs) + 1
#     fig, axs = plt.subplots(rows_count, **kwargs)

#     axs[0].axis('off')
#     axs[0].imshow(to_image(original_img, True))
    
    

#     if texts is None:
#         texts = dims
#     for ax, shifts_imgs, text in zip(axs[1:], imgs, texts):
#         ax.axis('off')
#         plt.subplots_adjust(left=0.5)
#         ax.imshow(to_image(make_grid(shifts_imgs.clamp(min=-1, max=1), nrow=(2 * shifts_count + 1),scale_each=True, normalize=True, padding=1), True))
#         ax.text(-20, 21, str(text), fontsize=10)


#     return fig


# @torch.no_grad()
# def inspect_all_directions(G, deformator, out_dir, zs=None, num_z=3, shifts_r=8.0):
#     os.makedirs(out_dir, exist_ok=True)

#     step = 4
#     max_dim = 51
#     codes = zs.cuda()
#     w = G.mapping(codes)['w']
#     zs  = G.truncation(w, trunc_psi = 0.7, trunc_layers = 8)
#     shifts_count = zs.shape[0]

#     for start in range(0, max_dim - 1, step):
#         imgs = []
#         dims = range(start, min(start + step, max_dim))
#         for z in zs:
#             z = z.unsqueeze(0)
#             fig = make_interpolation_chart(
#                 G, deformator=deformator, z=z,
#                 shifts_count=5, dims=dims, shifts_r=shifts_r,
#                 dpi=250, figsize=(int(shifts_count * 4.0), int(0.5 * step) + 2))
#             fig.canvas.draw()
#             plt.close(fig)
#             img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#             img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

#             # crop borders
#             nonzero_columns = np.count_nonzero(img != 255, axis=0)[:, 0] > 0
#             img = img.transpose(1, 0, 2)[nonzero_columns].transpose(1, 0, 2)
#             imgs.append(img)

#         out_file = os.path.join(out_dir, '{}_{}_{}.jpg'.format(dims[0], dims[-1],model_num))
#         print('saving chart to {}'.format(out_file))
#         Image.fromarray(np.hstack(imgs)).save(out_file)
        
# # z = torch.load('codes.pkl').cuda()
# z = torch.randn(3, 512)
# # z = torch.load(os.path.join(result_path,'temp','code_1.pkl'))
# out_dir = '/home/adarsh/PycharmProjects/disentagled_latent_dirs/results/cars/closedform_ours/qualitative_analysis/inspect_all_dirs_dse_60k'
# inspect_all_directions(generator, dse_deformator,out_dir,zs=z, shifts_r=2)


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
        temp_path = os.path.join(result_path, 'temp')
        os.makedirs(temp_path, exist_ok=True)
        for idx, z in enumerate(codes):
            print('Figure : ' + str(idx))
            z_shift_cf = []
            z_shift_dse = []
            for i, shift in enumerate(np.linspace(-shifts_r,shifts_r,shifts_count)):
                cf_shift_epsilon = (cf_deformator[cf_dir: cf_dir + 1] * shift).unsqueeze(1).repeat(1,len(layers),1)
                z_shift_cf.append(z + cf_shift_epsilon)
                dse_shift_epsilon = (dse_deformator[dse_dir: dse_dir + 1] * shift).unsqueeze(1).repeat(1,len(layers),1)
                z_shift_dse.append(z + dse_shift_epsilon)
            z_shift_cf = torch.stack(z_shift_cf).squeeze(dim=1)
            z_shift_dse = torch.stack(z_shift_dse).squeeze(dim=1)
            with torch.no_grad():
                cf_images= generator.synthesis(z_shift_cf)
            torch.save(cf_images, os.path.join(temp_path, 'cf.pkl'))
            del cf_images
            with torch.no_grad():
                dse_images= generator.synthesis(z_shift_dse)
            torch.save(dse_images, os.path.join(temp_path, 'dse.pkl'))
            del dse_images
            cf_images = torch.load(os.path.join(temp_path, 'cf.pkl'))
            dse_images = torch.load(os.path.join(temp_path, 'dse.pkl'))
            all_images = torch.cat((cf_images, dse_images), dim=0)
            grid = torchvision.utils.make_grid(all_images.clamp(min=-1, max=1),nrow=5, scale_each=True, normalize=True)
            display.display(plt.gcf())
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            del all_images
            del cf_images
            del dse_images
            del grid

    
z_min_index = 0
z_max_index = 100
codes = torch.load(os.path.join(result_path,  'Car type/attr_codes.pkl'))
cf_dir = 0
dse_dir = 0
shift_r = 1
shift_count = 5
all_images = save_images(codes, shift_r, shift_count, cf_dir, dse_dir, generator, cf_deformator, dse_deformator)                    


# # Plot Results

# In[ ]:


def get_manipulated_images(z, shift_r, shift_count, cf_dir, dse_dir, generator, cf_deformator, dse_deformator):
    temp_path =  os.path.join(result_path, 'temp')
    os.makedirs(temp_path, exist_ok=True)
    z_shift_cf = []
    z_shift_dse = []
    for i, shift in enumerate(np.linspace(-shifts_r,shifts_r,shifts_count)):
        cf_shift_epsilon = (cf_deformator[cf_dir: cf_dir + 1] * shift).unsqueeze(1).repeat(1,len(layers),1)
        z_shift_cf.append(z + cf_shift_epsilon)
        dse_shift_epsilon = (dse_deformator[dse_dir: dse_dir + 1] * shift).unsqueeze(1).repeat(1,len(layers),1)
        z_shift_dse.append(z + dse_shift_epsilon)
    z_shift_cf = torch.stack(z_shift_cf).squeeze(dim=1)
    z_shift_dse = torch.stack(z_shift_dse).squeeze(dim=1)
    with torch.no_grad():
        cf_images= generator.synthesis(z_shift_cf)
    with torch.no_grad():
        dse_images= generator.synthesis(z_shift_dse)
    return cf_images, dse_images

        


# In[ ]:


root_dir= '/home/adarsh/PycharmProjects/disentagled_latent_dirs'
result_path = os.path.join(root_dir,  'results/cars/closedform_ours/qualitative_analysis')
attr_list = ['Car type', 'Color', 'Zoom']
z = []
for each_attr in attr_list:
    z.append(torch.load(os.path.join(result_path, each_attr + '/attr_codes.pkl')))


# In[ ]:



shifts_r = 1
shifts_count = 5
cf_dir = 0
dse_dir = 0
desired_idx  = 1



cf_type, dse_type = get_manipulated_images(z[0][desired_idx], shifts_r, shifts_count, cf_dir, dse_dir, generator, cf_deformator, dse_deformator)


shifts_r = 1
shifts_count = 5
cf_dir = 3
dse_dir = 3
desired_idx  = 0

cf_color, dse_color = get_manipulated_images(z[1][desired_idx], shifts_r, shifts_count, cf_dir, dse_dir, generator, cf_deformator, dse_deformator)


shifts_r = 1
shifts_count = 5
cf_dir = 4
dse_dir = 4
desired_idx  = 0

cf_zoom, dse_zoom = get_manipulated_images(z[2][desired_idx], shifts_r, shifts_count, cf_dir, dse_dir, generator, cf_deformator, dse_deformator)



# In[ ]:


cf = torch.stack((cf_type, cf_color, cf_zoom),dim=0)
dse = torch.stack((dse_type, dse_color, dse_zoom),dim=0)
all_images = [cf, dse]


# In[ ]:


algo = ['SeFa', 'SeFa + SRE']


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SMALL_SIZE = 8
plt.rc('axes', titlesize=22, labelsize=20)
plt.rcParams["figure.facecolor"] = 'w'

fig = plt.figure(figsize=(20, 5))
gs = gridspec.GridSpec(2, 3, wspace=0.1, hspace=0.01)
ax = np.zeros(6, dtype=object)
count = 0
for i in range(2):
    for j in range(3):
        ax[count] = fig.add_subplot(gs[i, j])
        grid = torchvision.utils.make_grid(all_images[i][j].clamp(min=-1, max=1),nrow=5, scale_each=True, normalize=True)
        ax[count].imshow(grid.permute(1, 2, 0).cpu().numpy())
        ax[count].grid(False)
        ax[count].set_xticks([])
        ax[count].set_yticks([])
        count = count + 1
        ax[j].title.set_text(attr_list[j])
ax[0].set_ylabel(algo[0], rotation=90)
ax[3].set_ylabel(algo[1], rotation=90)
plt.rcParams["font.family"] = "Times New Roman"

gs.tight_layout(fig)
plt.savefig('test.pdf', bbox_inches = 'tight')

