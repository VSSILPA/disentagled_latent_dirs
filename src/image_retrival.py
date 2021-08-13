import torch
from utils import CelebADataset
from torchvision import transforms
import torch.nn as nn
import os
import numpy as np
from sefa_master.utils import *
from utils import NoiseDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import NewDataset
from models.rank_predictor import ResNetRankPredictor
from PIL import Image
from torchvision.transforms import ToTensor
import matplotlib
import matplotlib.pyplot as plt


class ImageRetriever(object):
    """

    """

    def __init__(self, pool_size=30000, model_type='pggan', batch_size=2):
        self.pool_size = pool_size
        self.top_k = 5
        self.model_type = model_type
        self.batch_size = batch_size
        self.similar_image_index = [1, 2, 3]
        self.distance = nn.PairwiseDistance(p=2)
        self.attr_idx = 2
        self.img_folder = '/media/adarsh/DATA/data1024x1024/'
        self.pool_loader = self._get_pool_loader()

    def get_similar_images(self, query_image, encoder):
        attributes_query_image = encoder(query_image)
        attributes_pool_image = self.get_attributes_pool_images(encoder)
        similar_images = self._retrive_similar_images(attributes_query_image.view(1, -1), attributes_pool_image)
        return similar_images

    def _get_pool_loader(self, file_path='', source='real_data'):
        if source == 'generator':
            if os.path.exists(file_path):
                image_array = np.load(file_path, mmap_mode='r')
            else:
                image_array = np.memmap('generated_images', dtype='float32', mode='w+',
                                        shape=(self.pool_size, 3, 256, 256))
                generator = load_generator('pggan_celebahq1024')
                z = NoiseDataset(num_samples=self.attributes_query_image_size, z_dim=generator.z_space_dim)
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
                                                      pin_memory=True, shuffle=False, drop_last=True)
        else:
            raise NotImplementedError
        return pool_loader

    def get_attributes_pool_images(self, encoder):

        representations = []
        with torch.cuda.amp.autocast():
            for batch_idx, images in enumerate(self.pool_loader):
                representation = encoder(images.cuda())
                representations.append(representation.detach().cpu())
        attributes = torch.stack(representations)
        torch.save(representations, 'real_data_representations.pkl')
        return attributes

    def _retrive_similar_images(self, attributes_query_image, attributes_pool_image):

        selected_attributes_query_image = attributes_query_image.view(1, -1).repeat(attributes_pool_image.shape[0],
                                                                                    1)  ##todo
        selected_attributes_pool_image = attributes_pool_image
        distance = self.distance(selected_attributes_pool_image, selected_attributes_query_image.cpu())
        distance_query, similar_image_idx = torch.topk(distance.double(), k=self.top_k, largest=False)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        celeba_dataset = CelebADataset(self.img_folder, transform)
        similar_images = [celeba_dataset.__getitem__(idx) for idx in similar_image_idx]
        similar_images = torch.stack(similar_images)
        return similar_images

    def get_nearest_neigbours(self, query_image):
        distance = []
        for i, images in enumerate(self.pool_loader):
            distance.append(self.distance(images.view(self.batch_size, 3 * 1024 * 1024),
                                          query_image.repeat(self.batch_size, 1, 1, 1).view(self.batch_size,
                                                                                            3 * 1024 * 1024)))
        distance = torch.stack(distance).view(-1)

        distance_query, similar_image_idx = torch.topk(distance.double(), k=self.top_k, largest=False)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        celeba_dataset = CelebADataset(self.img_folder, transform)
        similar_images = [celeba_dataset.__getitem__(idx) for idx in similar_image_idx]
        similar_images = torch.stack(similar_images)
        return similar_images

    def generate_plots(self, query_image, similar_image_ours, similar_images_knn):

        images = [similar_image_ours[:5], None, similar_images_knn[:5]]
        gs = matplotlib.gridspec.GridSpec(3, 6)

        fig = plt.figure()
        ax1 = fig.add_subplot(gs[1, 0])
        plt.imshow(query_image.squeeze(0).permute(1, 2, 0))
        for i in range(0, 3, 2):
            for j in range(1, 6):
                ax1 = fig.add_subplot(gs[i, j])
                plt.imshow(images[i][j - 1].squeeze(0).permute(1, 2, 0))

        plt.savefig('test.png')

        # fig = plt.figure()
        # ax1 = fig.add_subplot(gs[1, 0])
        #
        # # fig = plt.figure()
        # # ax1 = fig.add_subplot(gs[1, 0])
        # # plt.imshow(query_image.squeeze(0).permute(1, 2, 0))
        # # axes_list = [[[0.25, 0.25, 0.2, 0.2], [0.35, 0.35, 0.3, 0.3], [0.45, 0.45, 0.4, 0.4], [0.25, 0.25, 0.6, 0.6],
        # #               [0.25, 0.25, 0.7, 0.7]], None,
        # #              [[0.15, 0.15, 0.2, 0.2], [0.15, 0.15, 0.4, 0.4], [0.15, 0.15, 0.5, 0.5], [0.25, 0.15, 0.6, 0.6],
        # #               [0.15, 0.15, 0.7, 0.7]]
        # #              ]
        # for i in range(0, 3, 2):
        #     for j in range(1, 6):
        #         ax1 = fig.add_axes(axes_list[i][j - 1])
        #         plt.imshow(images[i][j - 1].squeeze(0).permute(1, 2, 0))
        # plt.savefig('/home/adarsh/PycharmProjects/disentagled_latent_dirs/src/test.png')
        # plt.imshow(query_image.squeeze(0).permute(1, 2, 0))
        # for i in range(0, 3, 2):
        #     for j in range(1, 6):
        #         ax1 = fig.add_axes([0.1, 0.1, 0.85, 0.85])
        #         plt.imshow(images[i][j - 1].squeeze(0).permute(1, 2, 0))


checkpoint = torch.load('../pretrained_models/best_model_ours/celeba_hq/18000_model.pkl')
rank_predictor = ResNetRankPredictor(num_dirs=512)
rank_predictor.load_state_dict(checkpoint['rank_predictor'])
rank_predictor.cuda().eval()
application = ImageRetriever(batch_size=4)
query_image_path = '/media/adarsh/DATA/celebA-HQ-dataset-download-master/0.jpg'
raw_image = Image.open(query_image_path)
query_image = ToTensor()(raw_image).unsqueeze(0)
# similar_images_knn = application.get_nearest_neigbours(query_image)
similar_images_ours = application.get_similar_images(query_image, 5, rank_predictor)

application.generate_plots(query_image, similar_images_ours, similar_images_knn)
