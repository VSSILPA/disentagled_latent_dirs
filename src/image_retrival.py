import torch
from utils import CelebADataset
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from models.rank_predictor import ResNetRankPredictor
from PIL import Image
from models.attribute_predictors import attribute_utils
import matplotlib
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt


class ImageRetriever(object):
    """

    """

    def __init__(self,transform, pool_size=1000, batch_size=64):
        self.pool_size = pool_size
        self.top_k = 6
        self.batch_size = batch_size
        self.distance = nn.PairwiseDistance(p=2)
        self.attr_idx = 1
        self.transform = transform
        self.eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.img_folder = '/media/adarsh/DATA/data1024x1024/'
        self.pool_loader = self._get_pool_loader()
        self.num_eval = 2000

    def get_similar_images(self, query_image, encoder):
        # with torch.cuda.amp.autocast():
        attributes_query_image = encoder(query_image).view(-1,512)[:, self.attr_idx]
        attributes_pool_image = self.get_attributes_pool_images(encoder).view(-1,512)[:, self.attr_idx]
        similar_images = self._retrieve_similar_images(attributes_query_image.view(1, -1), attributes_pool_image)
        return similar_images

    def _get_pool_loader(self):
        celeba_dataset = CelebADataset(self.img_folder, self.transform)
        pool_loader = torch.utils.data.DataLoader(celeba_dataset, batch_size=self.batch_size, num_workers=0,
                                                  pin_memory=True, shuffle=False, drop_last=True)
        return pool_loader

    def get_attributes_pool_images(self, encoder, cache=True):
        if cache:
            return torch.load('../pretrained_models/pool_representations_to_tensor.pkl')
        representations = []

        for batch_idx, images in enumerate(self.pool_loader):
            representation = encoder(images.cuda())
            representations.append(representation.detach().cpu())
        attributes = torch.stack(representations)
        return attributes

    def _retrieve_similar_images(self, attributes_query_image, attributes_pool_image):

        selected_attributes_query_image = attributes_query_image.view(-1).repeat(attributes_pool_image.shape[0], 1).view(-1)
        selected_attributes_pool_image = attributes_pool_image
        distance = self.distance(selected_attributes_pool_image.view(-1, 1), selected_attributes_query_image.view(-1, 1).cpu())
        distance_query, similar_image_idx = torch.topk(distance.double(), k=self.top_k, largest=False)
        celeba_dataset = CelebADataset(self.img_folder, self.transform)
        similar_images = [celeba_dataset.__getitem__(idx) for idx in similar_image_idx]
        similar_images = torch.stack(similar_images)
        return similar_images

    def compute_evaluation_metrics(self):
        query_idx = [str(random.randrange(1, self.pool_size, 1)).zfill(5) for i in range(self.num_eval)]
        similar_images_indices = []
        for idx in query_idx:
            query_image_path = self.img_folder + idx + '.jpg'
            raw_image = Image.open(query_image_path)
            query_image = self.eval_transform(raw_image)
            query_image_pre_processed = F.avg_pool2d(query_image, 4, 4)
            similar_images, similar_idx = self.get_similar_images(query_image_pre_processed)
            application.generate_plots(query_image, similar_images, similar_images)
            similar_images_indices.append(similar_idx)

        ground_truth_query_images = self.get_attribute_labels(query_idx)
        ground_truth_retrieved_images = self.get_attribute_labels(torch.stack(similar_images_indices).view(-1).tolist())
        ground_truth_query_images = torch.Tensor(ground_truth_query_images).unsqueeze(1).repeat(1, self.top_k)
        ground_truth_retrieved_images = torch.FloatTensor(ground_truth_retrieved_images).view(-1, self.top_k)
        print("finish")





    def get_classifier(self, attribute_name="Male"):
        if attribute_name != 'pose':
            classifier = attribute_utils.ClassifierWrapper(attribute_name, device='cuda')
            classifier.cuda().eval()
        return classifier

    def get_attribute_labels(self, image_idx):

        classifier = self.get_classifier()
        attribute_labels = []
        for idx in image_idx:
            image_path = self.img_folder + str(idx).zfill(5) + '.jpg'
            raw_image = Image.open(image_path)
            image = self.transform(raw_image)
            predictions = classifier(image.cuda())
            attribute_labels.append(torch.argmax(torch.softmax(predictions,dim=1), dim=-1).item())
        return attribute_labels

    def get_nearest_neighbours(self, query_image):
        distance = []
        for i, images in enumerate(self.pool_loader):
            distance.append(self.distance(images.view(self.batch_size, 3 * 1024 * 1024),
                                          query_image.repeat(self.batch_size, 1, 1, 1).view(self.batch_size,
                                                                                            3 * 1024 * 1024)))
            if i == 50:
                break
        distance = torch.stack(distance).view(-1)

        distance_query, similar_image_idx = torch.topk(distance.double(), k=self.top_k, largest=False)
        celeba_dataset = CelebADataset(self.img_folder, self.transform)
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
        print("done")

        # fig = plt.figure()
        # ax1 = fig.add_subplot(gstransform = transforms.Compose(
        #             [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

checkpoint = torch.load('../pretrained_models/best_model_ours/celeba_hq/18000_model.pkl')
rank_predictor = ResNetRankPredictor(num_dirs=512)
rank_predictor.load_state_dict(checkpoint['rank_predictor'])
rank_predictor.cuda().eval()
# transform = transforms.Compose(
#             [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform = transforms.Compose([transforms.ToTensor()])
application = ImageRetriever(transform, batch_size=4)
query_image_path = '/media/adarsh/DATA/data1024x1024/00224.jpg'
raw_image = Image.open(query_image_path)
query_image = transform(raw_image).unsqueeze(0)
# similar_images_knn = application.get_nearest_neighbours(query_image)
similar_images_ours = application.get_similar_images(query_image, rank_predictor)
application.generate_plots(query_image, similar_images_ours, similar_images_ours)
