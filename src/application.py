import torch
from utils import CelebADataset
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from models.rank_predictor import ResNetRankPredictor
from PIL import Image
import os
from models.attribute_predictors.attribute_predictor import get_classifier
import matplotlib
import random
import torch.nn.functional as F
from torchmetrics import RetrievalMAP
import torchvision
import matplotlib.pyplot as plt
import itertools


class ImageRetriever(object):
    """

    """

    def __init__(self, rank_predictor, transform, pool_size=1000, batch_size=64):
        self.pool_size = pool_size
        self.top_k = 6
        self.batch_size = batch_size
        self.distance = nn.PairwiseDistance(p=1)
        self.attr_idx = 24
        self.transform = transform
        self.eval_transform = transforms.Compose([transforms.ToTensor()])
        self.img_folder = '/media/adarsh/DATA/data1024x1024/'
        self.pool_loader = self._get_pool_loader()
        self.num_eval = 15
        self.encoder = rank_predictor

    def get_similar_images(self, query_image,i):
        attributes_query_image = self.encoder(query_image).view(-1, 512)[:, self.attr_idx]
        attributes_pool_image = self.get_attributes_pool_images(self.encoder).view(-1, 512)[:, self.attr_idx]
        similar_images, similar_image_idx = self._retrieve_similar_images(attributes_query_image.view(1, -1),
                                                                          attributes_pool_image,i)
        return similar_images, similar_image_idx

    def _get_pool_loader(self):
        celeba_dataset = CelebADataset(self.img_folder, self.transform)
        pool_loader = torch.utils.data.DataLoader(celeba_dataset, batch_size=self.batch_size, num_workers=0,
                                                  pin_memory=True, shuffle=False, drop_last=True)
        return pool_loader

    def get_attributes_pool_images(self, cache=True):
        if True:
            return torch.load('../pretrained_models/representations/celeba_hq/to_tensor_representation.pkl')
        representations = []
        for batch_idx, images in enumerate(self.pool_loader):
            representation = self.encoder(images.cuda())
            representations.append(representation.detach().cpu())
            if batch_idx % 1000 == 0:
                print(batch_idx)
        attributes = torch.stack(representations)
        return attributes

    def _retrieve_similar_images(self, attributes_query_image, attributes_pool_image,i):

        selected_attributes_query_image = attributes_query_image.view(-1).repeat(attributes_pool_image.shape[0],
                                                                                 1).view(-1)
        selected_attributes_pool_image = attributes_pool_image
        distance = self.distance(selected_attributes_pool_image.view(-1, 1),
                                 selected_attributes_query_image.view(-1, 1).cpu())
        distance_query, similar_image_idx = torch.topk(distance.double(), k=self.top_k, largest=False)
        celeba_dataset = CelebADataset(self.img_folder, self.transform)
        similar_images = [celeba_dataset.__getitem__(idx) for idx in similar_image_idx]
        similar_images = torch.stack(similar_images)
        grid_img = torchvision.utils.make_grid(similar_images[:self.top_k], nrow=10, padding=10, pad_value=1)
        grid = grid_img.permute(1, 2, 0).type(torch.FloatTensor)
        plt.imsave('similar_images'+str(i)+'.svg', grid.data.numpy())
        return similar_images, similar_image_idx

    def compute_evaluation_metrics(self):
        query_idx = [str(random.randrange(1, self.pool_size, 1)).zfill(5) for i in range(self.num_eval)]
        similar_images_indices = []
        for i, idx in enumerate(query_idx):
            query_image_path = self.img_folder + idx + '.jpg'
            raw_image = Image.open(query_image_path)
            query_image = self.transform(raw_image).unsqueeze(0)
            similar_images, similar_idx = self.get_similar_images(query_image)
            similar_images_indices.append(similar_idx[1:])
            if i % 20 == 0:
                print(i)

        ground_truth_query_images = self.get_attribute_labels(query_idx)
        ground_truth_retrieved_images = self.get_attribute_labels(torch.stack(similar_images_indices).view(-1).tolist())
        ground_truth_query_images = torch.Tensor(ground_truth_query_images).unsqueeze(1).repeat(1, self.top_k - 1).view(
            -1)
        ground_truth_retrieved_images = torch.FloatTensor(ground_truth_retrieved_images).view(-1, self.top_k - 1).view(
            -1)
        index = [[x] * (self.top_k - 1) for x in range(len(query_idx))]
        indexes = torch.LongTensor(list(itertools.chain(*index)))
        rmap = RetrievalMAP()
        metric = rmap(ground_truth_retrieved_images, ground_truth_query_images.bool(), indexes=indexes)
        print(metric)

    def get_classifier(self, attribute_name="male"):
        classifier = get_classifier(
            os.path.join("../pretrained_models", "classifiers", attribute_name, "weight.pkl"),
            'cpu')
        classifier.cuda().eval()
        return classifier

    def get_attribute_labels(self, image_idx):

        classifier = self.get_classifier()
        attribute_labels = []
        for idx in image_idx:
            image_path = self.img_folder + str(idx).zfill(5) + '.jpg'
            raw_image = Image.open(image_path)
            image = self.eval_transform(raw_image).unsqueeze(0)
            image = (image + 1) / 2
            image = F.avg_pool2d(image, 4, 4)
            predictions = classifier(image.cuda())
            attribute_labels.append(torch.argmax(torch.softmax(predictions, dim=1), dim=-1).item())
        return attribute_labels

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

# old_peoples = [1779,1734,1818,1817,2166,2250,2344,2744]

for i in range(4100,4125):

    checkpoint = torch.load('../results/celeba_hq/closed_form_ours/models/18000_model.pkl')
    rank_predictor = ResNetRankPredictor(num_dirs=512)
    rank_predictor.load_state_dict(checkpoint['rank_predictor'])
    rank_predictor.cuda().eval()

    transform = transforms.Compose([transforms.ToTensor()])
    application = ImageRetriever(rank_predictor, transform, batch_size=4)
    query_image_path = '/media/adarsh/DATA/data1024x1024/0'+str(i)+'.jpg'
    raw_image = Image.open(query_image_path)
    query_image = transform(raw_image).unsqueeze(0)

    # application.compute_evaluation_metrics()
    similar_images_ours, similar_images_idx = application.get_similar_images(query_image,i)
    application.generate_plots(query_image, similar_images_ours, similar_images_ours)
