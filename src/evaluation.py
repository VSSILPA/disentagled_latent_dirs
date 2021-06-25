import torch
import numpy as np
import os
import random
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

        train_loader, test_loader = data
        latent_rep = []
        labels_true = []
        for images, labels in test_loader:
            out_dis, hid = gan.dis(images.to(self.device))
            c1 = F.log_softmax(gan.Q_cat(hid))
            predicted_labels = torch.argmax(c1, dim=1)
            latent_rep.append(predicted_labels)
            labels_true.append(labels)
        latent_rep = torch.stack(latent_rep).view(-1).detach().cpu().numpy()
        labels_true = torch.stack(labels_true).view(-1).detach().cpu().numpy()

        purity = self._compute_purity(latent_rep, labels_true)
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

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
