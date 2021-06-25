import torch
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt


class Evaluator(object):
    def __init__(self, config, opt):
        self.config = config
        self.device = torch.device('cuda:' + str(opt.device_id))
        self.opt = opt

    def compute_metrics(self, data, gan):

        train_loader , test_loader = data
        latent_rep = []
        labels_true = []
        self.generate_images(gan)
        for images, labels in test_loader:
            out_dis, hid = gan.dis(images.to(self.device))
            c1 = F.log_softmax(gan.Q_cat(hid))
            predicted_labels = torch.argmax(c1, dim=1)
            latent_rep.append(predicted_labels)
            labels_true.append(labels)
        latent_rep = torch.stack(latent_rep).view(-1).detach().cpu().numpy()
        labels_true = torch.stack(labels_true).view(-1).detach().cpu().numpy()

        # km = KMeans(n_clusters=max(self.opt.num_classes, len(np.unique(labels_true))), random_state=0).fit(latent_rep.detach().cpu().numpy())
        # labels_pred = km.labels_
        # purity = self._compute_purity(labels_pred, labels_true)
        ari = adjusted_rand_score(labels_true, latent_rep)
        nmi = normalized_mutual_info_score(labels_true, latent_rep)
        return {'NMI': nmi, 'ARI': ari, 'ACC': purity}

    def _compute_purity(self, labels_pred, labels_true):
        clusters = set(labels_pred)

        # find out what class is most frequent in each cluster
        cluster_classes = {}
        correct = 0
        for cluster in clusters:
            # get the indices of rows in this cluster
            indices = np.where(labels_pred == cluster)[0]

            cluster_labels = labels_true[indices]
            majority_label = np.argmax(np.bincount(cluster_labels))
            correct += np.sum(cluster_labels == majority_label)

        return float(correct) / len(labels_pred)

    def generate_images(self, model):
        z_dict = model.get_z(10 * 100, sequential=True)
        gan_input = torch.cat([z_dict[k] for k in z_dict.keys()], dim=1)
        gan_input = Variable(gan_input, requires_grad=True).to(self.device)
        imgs = model.gen(gan_input)
        grid_img = torchvision.utils.make_grid(imgs[:100], nrow=10, normalize=True)
        plt.imshow(grid_img.permute(1, 2, 0).cpu().data)
        plt.savefig('/home/adarsh/PycharmProjects/disentagled_latent_dirs/imgs.png')

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
