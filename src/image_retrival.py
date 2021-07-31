import torch
from celeba_dataset import CelebADataset
from torchvision import transforms
import torch.nn as nn
import os
import numpy as np
from sefa_master.utils import *
from utils import NoiseDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from custom_dataset import NewDataset
from models.cr_discriminator import ResNetRankPredictor
from PIL import Image
from torchvision.transforms import ToTensor


class ImageRetriever(object):
    """

    """

    def __init__(self, encoder, pool_size=10, model_type='pggan', batch_size=2):
        self.pool_size = pool_size
        self.top_k = 10
        self.model_type = model_type
        self.batch_size = batch_size
        self.similar_image_index = [1, 2, 3]
        self.encoder = torch.load(encoder) if isinstance(encoder, str) else encoder
        self.distance = nn.PairwiseDistance(p=2)
        self.img_folder = ''
        self.pool_loader = self._get_pool_loader()

    def get_similar_images(self, query_image, attribute_index):
        attributes_query_image = self.encoder(query_image)
        attributes_pool_image = self.get_attributes_pool_images()
        similar_images = self._retrive_similar_images(attributes_query_image, attributes_pool_image)
        return similar_images

    def _get_pool_loader(self, file_path='', source='generator'):
        if source == 'generator':
            if os.path.exists(file_path):
                image_array = np.load(file_path, mmap_mode='r')
            else:
                image_array = np.memmap('generated_images', dtype='float32', mode='w+',
                                        shape=(self.pool_size,3,256,256))
                generator = load_generator('pggan_celebahq1024')
                z = NoiseDataset(num_samples=self.poattributes_query_imageol_size, z_dim=generator.z_space_dim)
                z_loader = DataLoader(z, batch_size=self.batch_size, shuffle=False)
                with torch.no_grad():
                    for batch_idx, z in enumerate(z_loader):
                        image = generator(z.cuda())['image']
                        image = F.avg_pool2d(image, 4, 4)
                        image_array[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size] = image.cpu().numpy()
            celeba_dataset = NewDataset(image_array)
            pool_loader = torch.utils.data.DataLoader(celeba_dataset, batch_size=self.batch_size,
                                                           num_workers=0, pin_memory=True, shuffle=True)
        elif source == 'real_data':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            celeba_dataset = CelebADataset(self.img_folder, transform)
            pool_loader = torch.utils.data.DataLoader(celeba_dataset, batch_size=self.batch_size, num_workers=0,
                                                           pin_memory=True, shuffle=True)
        else:
            raise NotImplementedError
        return pool_loader


    def get_attributes_pool_images(self):

        attributes = torch.zeros((self.pool_size, self.encoder.shift_estimator.out_features))
        for batch_idx, images in enumerate(self.pool_loader):
            representation = self.encoder(images)
            attributes[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size] = representation
        return attributes

    def _retrive_similar_images(self, attributes_query_image, attributes_pool_image):

        selected_attributes_query_image = attributes_query_image.repeat(self.pool_size).view(-1,self.pool_size)
        selected_attributes_pool_image = attributes_pool_image
        distance = self.distance(selected_attributes_pool_image, selected_attributes_query_image)
        distance_query, similar_image_idx = torch.topk(distance, k=self.top_k, largest=False)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        celeba_dataset = CelebADataset(self.img_folder, transform)
        similar_images = celeba_dataset.__getitem__(similar_image_idx)
        return similar_images


# checkpoint = torch.load('../pretrained_models/70007_model.pkl')['shift_predictor']
# encoder = ResNetRankPredictor.load_state_dict(checkpoint)
encoder = ResNetRankPredictor(dim=10)
application = ImageRetriever(encoder)
query_image_path = '/media/adarsh/DATA/celebA-HQ-dataset-download-master/0.jpg'
raw_image = Image.open(query_image_path)
query_image = ToTensor()(raw_image).unsqueeze(0)
similar_images = application.get_similar_images(query_image,5)
