import numpy as np
import torch


class FactorVAEMetric(object):
	""" Impementation of the metric in:
        factor-VAE metric: Learning Basic Visual Concepts with a Constrained Variational
        Framework
    """

	def __init__(self, dsprites, device_id, config):
		super(FactorVAEMetric, self).__init__()
		self.data = dsprites
		self.device_id = device_id
		self.config = config

	def compute_factor_vae(self, model, random_state, batch_size=64, num_train=10000, num_eval=5000,
						   num_variance_estimate=10000):

		global_variances = self._compute_variances(model, num_variance_estimate)
		active_dims = self._prune_dims(global_variances)
		if not active_dims.any():
			scores_dict = {"train_accuracy": 0., "eval_accuracy": 0., "num_active_dims": 0}
			return scores_dict

		training_votes = self._generate_training_batch(model, batch_size, num_train, random_state,
													   global_variances, active_dims)
		classifier = np.argmax(training_votes, axis=0)
		other_index = np.arange(training_votes.shape[1])
		train_accuracy = np.sum(training_votes[classifier, other_index]) * 1. / np.sum(training_votes)

		eval_votes = self._generate_training_batch(model, batch_size, num_eval, random_state,
												   global_variances, active_dims)
		eval_accuracy = np.sum(eval_votes[classifier, other_index]) * 1. / np.sum(eval_votes)
		scores_dict = {"train_accuracy": train_accuracy, "eval_accuracy": eval_accuracy,
					   "num_active_dims": len(active_dims)}
		return scores_dict

	def _prune_dims(self, variances, threshold=0.05):
		"""Mask for dimensions collapsed to the prior."""
		scale_z = np.sqrt(variances)
		return scale_z >= threshold

	def _compute_variances(self, model, num_variance_estimate):

		latents = self.data.sample_latent(size=num_variance_estimate)
		observations = self.data.sample_images_from_latent(latents)
		train_loader = torch.utils.data.DataLoader(observations, batch_size=self.config['batch_size'], shuffle=True,drop_last = True)
		representations_list = []
		for images in train_loader:
			representations, _ = model.encoder(images.cuda(self.device_id))
			representations_list.append(representations.data.cpu())
		representations = torch.stack(representations_list)
		representations = representations.view(-1,self.config['latent_dim'])
		# assert representations.shape[0] == num_variance_estimate
		return np.var(representations.numpy(), axis=0, ddof=1)

	def _generate_training_sample(self, model, batch_size, random_state, global_variances,
								  active_dims):

		# Select random coordinate to keep fixed.
		factor_index = random_state.randint(low=self.config['low_factor_vae'], high=self.data.num_factors)

		# Sample two mini batches of latent variables.
		factors1 = self.data.sample_latent(batch_size)

		# Fix the selected factor across mini-batch.
		factors1[:, factor_index] = factors1[0, factor_index]
		observation = self.data.sample_images_from_latent(factors1)

		representations, _ = model.encoder(torch.from_numpy(observation))
		representations = representations.data.cpu().numpy()

		## Rescaling
		local_variances = np.var(representations, axis=0, ddof=1)
		argmin = np.argmin(local_variances[active_dims] / global_variances[active_dims])
		return factor_index, argmin

	def _generate_training_batch(self, model,
								 batch_size, num_points, random_state,
								 global_variances, active_dims):

		votes = np.zeros((self.data.num_factors, global_variances.shape[0]), dtype=np.int64)
		for _ in range(num_points):
			factor_index, argmin = self._generate_training_sample(model, batch_size, random_state,
																  global_variances, active_dims)
			votes[factor_index, argmin] += 1
		return votes