import numpy as np
import logging
from sklearn.metrics import mutual_info_score
import torch

import sklearn

log = logging.getLogger(__name__)


class MIG(object):
    """
        Implementation of the metric in: MIG
    """

    def __init__(self, dsprites, device_id, opt):
        super(MIG, self).__init__()
        self.data = dsprites
        self.device_id = device_id
        self.config = opt

    def compute_mig(self, model, num_train=10000, batch_size=64):

        representations, ground_truth = self.generate_batch_factor_code(model, num_train, batch_size)
        mat, e = self._get_mi_matrix(representations, ground_truth, bins=20)
        sorted_m = np.sort(mat, axis=0)[::-1]
        logging.info("MIG element wise " + str(np.divide(sorted_m[0, :] - sorted_m[1, :], e[:])))
        mig = np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], e[:]))
        return mig

    def discrete_entropy(self, ys):
        """
        Compute discrete mutual information."""
        num_factors = ys.shape[0]
        h = np.zeros(num_factors)
        for j in range(num_factors):
            h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
        return h

    def discrete_mutual_info(self, z, v):
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

    def _get_mi_matrix(self, representations, ground_truth, bins=20):
        x = self._histogram_discretize(representations, num_bins=bins)
        f = ground_truth
        mat = self.discrete_mutual_info(x, f)
        e = self.discrete_entropy(f)
        return mat, e

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

    def _histogram_discretize(self, target, num_bins=20):
        discretized = np.zeros_like(target)
        for i in range(target.shape[0]):
            discretized[i, :] = np.digitize(
                target[i, :], np.histogram(target[i, :], num_bins)[1][:-1]
            )
        return discretized
