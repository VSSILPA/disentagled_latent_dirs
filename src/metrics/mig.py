import numpy as np
import logging
from sklearn.metrics import mutual_info_score
import torch

import sklearn


class MIG(object):
    """
            Implementation of the metric in: MIG
    """

    def __init__(self, dsprites, device_id,opt):
        super(MIG, self).__init__()
        self.data = dsprites
        self.device_id = device_id
        self.config = opt

    def compute_mig(self, model, num_train=10000, batch_size=64):

        score_dict = {}
        representations, ground_truth = self.generate_batch_factor_code(model, num_train, batch_size)

        normalized_representation = self.normalize_data(representations)
        normalized_ground_truth = self.normalize_data(ground_truth)

        ## discretize data
        discrete_representation = self.discretize_data(normalized_representation)
        discrete_ground_truth = self.discretize_data(normalized_ground_truth)

        m = self.discrete_mutual_info(discrete_representation, discrete_ground_truth)
        # m is [num_latents, num_factors]

        entropy = self.discrete_entropy(discrete_ground_truth)
        sorted_m = np.sort(m, axis=0)[::-1]
        dimension_wise_mig = np.divide((sorted_m[0, :] - sorted_m[1, :])[1:],entropy[1:])  # 1: skips the first latent code that is constant
        logging.info(dimension_wise_mig)
        score_dict["discrete_mig"] = np.mean(dimension_wise_mig[0:])
        return score_dict

    def discrete_entropy(self,ys):
        """Compute discrete mutual information."""
        num_factors = ys.shape[0]
        h = np.zeros(num_factors)
        for j in range(num_factors):
            h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
        return h

    def normalize_data(self, data, mean=None, stddev=None):
        if mean is None:
            mean = np.mean(data, axis=1)
        if stddev is None:
            stddev = np.std(data, axis=1)
        return (data - mean[:, np.newaxis]) / stddev[:, np.newaxis]

    def discretize_data(self,target, num_bins=20):
        """Discretization based on histograms."""
        target = np.nan_to_num(target)
        discretized = np.zeros_like(target)
        for i in range(target.shape[0]):
            discretized[i, :] = np.digitize(target[i, :], np.histogram(target[i, :], num_bins)[1][:-1])
        return discretized

    def discrete_mutual_info(self,z, v):
        """Compute discrete mutual information."""
        num_codes = z.shape[0]
        num_factors = v.shape[0]
        m = np.zeros([num_codes, num_factors])
        for i in range(num_codes):
            for j in range(num_factors):

                if num_factors > 1:
                    m[i, j] = sklearn.metrics.mutual_info_score(v[j, :], z[i, :])
                elif num_factors == 1:
                    m[i, j] = sklearn.metrics.mutual_info_score(np.squeeze(v), z[i, :])

        return m

    def generate_batch_factor_code(self, model, num_points, batch_size):

        representations = None
        factors = None
        i = 0
        while i < num_points:
            num_points_iter = min(num_points - i, batch_size)
            current_factors = self.data.sample_latent(num_points_iter)
            current_observations = torch.from_numpy(self.data.sample_images_from_latent(current_factors))
            current_representations = model(current_observations)
            current_representations = current_representations.data.cpu()
            if i == 0:
                factors = current_factors
                representations = current_representations
            else:
                factors = np.vstack((factors, current_factors))
                representations = np.vstack((representations, current_representations))
            i += num_points_iter
        return np.transpose(representations), np.transpose(factors)