import numpy as np
import torch
from sklearn import linear_model


class BetaVAEMetric(object):
    """
        Impementation of the metric in:
        beta-VAE: Learning Basic Visual Concepts with a Constrained Variational
        Framework
    """

    def __init__(self, dsprites, device_id, opt):
        super(BetaVAEMetric, self).__init__()

        self.data = dsprites
        self.device_id = device_id
        self.opt = opt

    def compute_beta_vae(self, model, random_state, batch_size=64, num_train=10000, num_eval=5000):

        train_points, train_labels = self._generate_training_batch(model, batch_size,
                                                                   num_train, random_state)
        eval_points, eval_labels = self._generate_training_batch(model, batch_size,
                                                                 num_eval, random_state)

        model = linear_model.LogisticRegression(random_state=random_state, n_jobs=-1)
        model.fit(train_points, train_labels)
        train_accuracy = np.mean(model.predict(train_points) == train_labels)
        eval_accuracy = model.score(eval_points, eval_labels)

        scores_dict = {"train_accuracy": train_accuracy, "eval_accuracy": eval_accuracy}
        return scores_dict

    def _generate_training_batch(self, model, batch_size, num_points, random_state):

        points = None  # Dimensionality depends on the representation function.
        labels = np.zeros(num_points, dtype=np.int64)
        for i in range(num_points):
            labels[i], feature_vector = self._generate_training_sample(model, batch_size, random_state)
            if points is None:
                points = np.zeros((num_points, feature_vector.shape[0]))
            points[i, :] = feature_vector
        return points, labels

    def _generate_training_sample(self, model, batch_size, random_state):

        # Select random coordinate to keep fixed.
        index = random_state.randint(low=0,
                                     high=self.data.num_factors)  # 2-size ,3-orientation, 4-X-position 5 - Yposition

        # Sample two mini batches of latent variables.
        factors1 = self.data.sample_latent(batch_size)
        factors2 = self.data.sample_latent(batch_size)

        # Ensure sampled coordinate is the same across pairs of samples.
        factors1[:, index] = factors2[:, index]

        # Transform latent variables to observation space.
        observation1 = self.data.sample_images_from_latent(factors1)
        observation2 = self.data.sample_images_from_latent(factors2)

        #   Compute representations based on the observations.
        representation1 = model(torch.from_numpy(observation1).cuda(self.device_id))
        representation2 = model(torch.from_numpy(observation2).cuda(self.device_id))
        representation1 = representation1.data.cpu().numpy()
        representation2 = representation2.data.cpu().numpy()

        # representation1 = factors1
        # representation2 = factors2

        # Compute the feature vector based on differences in representation.
        feature_vector = np.mean(np.abs(representation1 - representation2), axis=0)
        return index, feature_vector