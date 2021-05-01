from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from utils import *


class DCIMetric(object):
    """ Impementation of the metric in:
        beta-VAE: Learning Basic Visual Concepts with a Constrained Variational
        Framework
    """

    def __init__(self, dsprites, device_id):
        super(DCIMetric, self).__init__()
        self.data = dsprites
        self.device_id = device_id

    def compute_dci(self, model):
        importance_matrix, train_errs, test_errs = self.get_importance_matrix(model)
        # disentanglement
        disent_scores = entropic_scores(importance_matrix.T)
        c_rel_importance = np.sum(importance_matrix, 1) / np.sum(
            importance_matrix)  # relative importance of each code variable
        disent_scores = np.nan_to_num(disent_scores, copy=True, nan=0.0, posinf=None, neginf=None)
        disentaglement_vector = np.array(disent_scores) * c_rel_importance
        disent_w_avg = np.sum(disentaglement_vector)

        # completeness
        complete_scores = entropic_scores(importance_matrix)
        completness_vector = complete_scores * (importance_matrix.sum(axis=0) / importance_matrix.sum())
        complete_avg = np.sum(completness_vector)

        # informativeness (append averages)
        train_errs[0, -1] = np.mean(train_errs[0, :-1])
        test_errs[0, -1] = np.mean(test_errs[0, :-1])
        return {'disentanglement_vector': disentaglement_vector,
                'completeness_vector': completness_vector,
                'informativeness_vector': train_errs[0, :-1],
                'disentanglement': disent_w_avg, 'completeness': complete_avg,
                'informativeness': test_errs[0, -1]}

    def get_importance_matrix(self, model, num_train=10000, num_test=5000, batch_size=1024):
        x_train, y_train = self.get_normalized_data(model, num_points=num_train, batch_size=batch_size)
        x_test, y_test = self.get_normalized_data(model, num_points=num_test, batch_size=batch_size)
        num_factors = y_train.shape[1]
        train_errs = np.zeros((1, num_factors + 1))
        test_errs = np.zeros((1, num_factors + 1))
        importance_matrix = []
        for j in range(num_factors):
            parameters = {"alpha": [0.0001, 0.005, 0.001, 0.01, 0.1, 1]}
            model_xg = Lasso()
            model = GridSearchCV(model_xg, parameters, cv=5)
            model.fit(x_train, y_train[:, j])
            final_model = Lasso(alpha=model.best_params_['alpha'])
            final_model.fit(x_train, y_train[:, j])
            train_errs[0, j] = nrmse(final_model.predict(x_train), y_train[:, j])
            test_errs[0, j] = nrmse(final_model.predict(x_test), y_test[:, j])
            r = getattr(final_model, 'coef_')[:, None]  # [n_c, 1]
            importance_matrix.append(np.abs(r))
        importance_matrix = np.hstack(importance_matrix)
        return importance_matrix, train_errs, test_errs

    def get_normalized_data(self, model, num_points, batch_size):

        representations = None
        factors = None
        i = 0
        while i < num_points:
            num_points_iter = min(num_points - i, batch_size)
            current_factors, current_observations = self.data.sample(num_points_iter)
            current_representations = model(torch.from_numpy(current_observations))
            current_representations = current_representations.data.cpu()
            if i == 0:
                factors = current_factors
                representations = current_representations
            else:
                factors = np.vstack((factors, current_factors))
                representations = np.vstack((representations, current_representations))
            i += num_points_iter
        return normalize(representations)[0], normalize(factors)[0]