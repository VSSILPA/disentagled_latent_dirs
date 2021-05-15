from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
from utils import *
import scipy


class DCIMetric(object):
    """
        Implementation of the metric in:
        beta-VAE: Learning Basic Visual Concepts with a Constrained Variational
        Framework
    """

    def __init__(self, data, device_id):
        super(DCIMetric, self).__init__()
        self.data = data
        self.device_id = device_id

    def compute_dci(self, model, num_train=5000, num_test=5000, batch_size=1024):
        x_train, y_train = self.get_normalized_data(model, num_points=num_train, batch_size=batch_size)
        x_test, y_test = self.get_normalized_data(model, num_points=num_test, batch_size=batch_size)
        importance_matrix, train_err, test_err = self._compute_importance_gbt(x_train, y_train, x_test, y_test)
        disentanglement_vector = self._disentanglement(importance_matrix)
        completeness_vector = self._completeness(importance_matrix)
        return {'disentanglement_vector': disentanglement_vector,
                'completeness_vector': completeness_vector,
                'informativeness_vector': train_err,
                'disentanglement': np.sum(disentanglement_vector), 'completeness': np.sum(completeness_vector),
                'informativeness': np.mean(test_err)}

    def _disentanglement(self, importance_matrix):
        """Compute the disentanglement score of the representation."""
        per_code = 1.0 - scipy.stats.entropy(
            importance_matrix.T + 1e-11, base=importance_matrix.shape[1]
        )
        if importance_matrix.sum() == 0.0:
            importance_matrix = np.ones_like(importance_matrix)
        code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()
        disentaglement_vector = per_code * code_importance
        return disentaglement_vector

    def _completeness(self, importance_matrix):
        """"Compute completeness of the representation."""
        per_factor = 1.0 - scipy.stats.entropy(
            importance_matrix + 1e-11, base=importance_matrix.shape[0]
        )
        if importance_matrix.sum() == 0.0:
            importance_matrix = np.ones_like(importance_matrix)
        factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
        completness_vector = per_factor * factor_importance
        return completness_vector

    def _compute_importance_gbt(self, x_train, y_train, x_test, y_test):
        """Compute importance based on gradient boosted trees."""
        num_factors = y_train.shape[1]
        num_codes = x_train.shape[1]
        importance_matrix = np.zeros(shape=[num_codes, num_factors], dtype=np.float64)
        train_loss = []
        test_loss = []
        for i in range(num_factors):
            model = ensemble.GradientBoostingClassifier()
            model.fit(x_train, y_train[:,i])            #TODO using Gradient boosting classifier use lasso and check results as well use git your infogan repo
            importance_matrix[:, i] = np.abs(model.feature_importances_)
            train_loss.append(np.mean(model.predict(x_train) == y_train[:, i]))
            test_loss.append(np.mean(model.predict(x_test) == y_test[:, i]))
        return importance_matrix, train_loss, test_loss

    def get_normalized_data(self, model, num_points, batch_size):  # Note : Currently not normalising the data

        representations = None
        factors = None
        i = 0
        while i < num_points:
            num_points_iter = min(num_points - i, batch_size)
            current_factors = self.data.sample_latent(num_points_iter)
            current_observations = self.data.sample_images_from_latent(current_factors)
            current_representations = model(torch.from_numpy(current_observations))
            current_representations = current_representations.data.cpu()
            if i == 0:
                factors = current_factors
                representations = current_representations
            else:
                factors = np.vstack((factors, current_factors))
                representations = np.vstack((representations, current_representations))
            i += num_points_iter
        return representations, factors
