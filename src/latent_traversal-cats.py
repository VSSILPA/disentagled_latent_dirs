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
import cv2
from IPython import display
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
result_path = os.path.join(root_dir,  'results/cats/closed_form_ours/qualitative_analysis')


# ## Model Selection

# In[4]:


deformator_path = os.path.join(root_dir, 'pretrained_models/deformators/ClosedForm/stylegan_cat256/stylegan_cat256.pkl')
layers, cf_deformator, _ = torch.load(deformator_path, map_location='cpu')
cf_deformator = torch.FloatTensor(cf_deformator).cuda()

deformator_path = os.path.join(root_dir, 'results/cats/models/50000_model.pkl')
deformator = torch.load(deformator_path)['deformator']['ortho_mat']
dse_deformator = deformator.T
        
generator = load_generator(None, model_name='stylegan_cat256')

if load_codes:
#     codes = np.load(os.path.join(root_dir, 'pretrained_models/latent_codes/pggan_celebahq1024_latents.npy'))
#     codes = torch.from_numpy(codes).type(torch.FloatTensor).cuda()
    codes = torch.load(os.path.join(root_dir, 'results/celeba_hq/closed_form_ours/quantitative_analysis/z_analysis.pkl'))
else:
    num_samples = 1000
    codes = torch.randn(num_samples, 512).cuda()
    w = generator.mapping(codes)['w']
    codes  = generator.truncation(w, trunc_psi = 0.7, trunc_layers = 8)


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
            grid = torchvision.utils.make_grid(all_images.clamp(min=-1, max=1),nrow=3, scale_each=True, normalize=True)
            display.display(plt.gcf())
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            del all_images
            del cf_images
            del dse_images
            del grid

    
z_min_index = 0
z_max_index = 100
cf_dir = 5
dse_dir = 5
shift_r = 1
shift_count = 3
all_images = save_images(codes[z_min_index:z_max_index], shift_r, shift_count, cf_dir, dse_dir, generator, cf_deformator, dse_deformator)                    


# In[ ]:




