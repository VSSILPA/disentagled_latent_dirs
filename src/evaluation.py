import torch
import logging
from utils import make_noise
import models.latent_regressor as latent_regressor
import numpy as np
import torch.nn as nn
import time
from metrics.betavae_metric import BetaVAEMetric
from metrics.factor_vae_metric import FactorVAEMetric
from metrics.mig import MIG
from metrics.dci_metric import DCIMetric
from latent_dataset import LatentDataset
import os
from config import BB_KWARGS

log = logging.getLogger(__name__)


class Evaluator(object):
    def __init__(self, config, opt):
        self.config = config
        self.device = torch.device('cuda:' + str(opt.device_id))
        self.opt = opt
        self.encoder = latent_regressor.Encoder(latent_dimension=self.opt.encoder.latent_dimension,
                                                backbone="cnn_encoder",
                                                **BB_KWARGS[self.opt.dataset])
        self.metric_eval = {'beta_vae': [], 'factor_vae': [], 'mig': []}

    def compute_metrics(self, generator, directions, data, epoch):
        start_time = time.time()
        encoder = self._train_encoder(generator, directions)

        beta_vae = BetaVAEMetric(data, self.device, self.opt)
        factor_vae = FactorVAEMetric(data, self.device, self.opt)
        mig = MIG(data, self.device, self.opt)
        DCI_metric = DCIMetric(data, self.device)
        beta_vae_metric = beta_vae.compute_beta_vae(encoder, np.random.RandomState(self.opt.random_seed),
                                                    batch_size=64,
                                                    num_train=5000, num_eval=5000)
        logging.info("Computed beta vae metric")
        factor_vae_metric = factor_vae.compute_factor_vae(encoder, np.random.RandomState(self.opt.random_seed),
                                                          batch_size=64, num_train=5000, num_eval=5000,
                                                          num_variance_estimate=5000)
        logging.info("Computed factor vae metric")
        mutual_info_gap = mig.compute_mig(encoder, num_train=10000, batch_size=128)

        logging.info("Computed mig metric")
        dci = DCI_metric.compute_dci(encoder)
        logging.info("Computed dci metric")

        dci_average = (dci['disentanglement'] + dci['completeness'] + dci['informativeness']) / 3
        metrics = {'beta_vae': beta_vae_metric, 'factor_vae': factor_vae_metric, 'mig': mutual_info_gap,
                   'dci_metric': dci_average}
        self.metric_eval['beta_vae'].append(metrics['beta_vae']["eval_accuracy"])
        self.metric_eval['factor_vae'].append(metrics['factor_vae']["eval_accuracy"])
        self.metric_eval['mig'].append(metrics['mig'])
        logging.info('Disentanglement Vector')
        logging.info(dci['disentanglement_vector'])
        logging.info('completeness_vector')
        logging.info(dci['completeness_vector'])
        logging.info('informativeness_vector')
        logging.info(dci['informativeness_vector'])
        logging.info(
            "Time taken %d sec B-VAE: %.3f, F-VAE %.3F, MIG : %.3f Disentanglement: %.3f "
            "Completeness: "
            "%.3f Informativeness: %.3f " % (
                time.time() - start_time,
                metrics['beta_vae'][
                    "eval_accuracy"],
                metrics['factor_vae'][
                    "eval_accuracy"],
                metrics['mig'], dci['disentanglement'],
                dci['completeness'], dci['informativeness']
            ))
        return self.metric_eval

    def _train_encoder(self, generator, directions):
        model = self.encoder.to(self.device)
        model = nn.DataParallel(model)
        loader = self._get_encoder_train_data(generator, directions)
        trained_model = latent_regressor._train(model, loader, self.opt)
        return trained_model

    def _get_encoder_train_data(self, generator, directions):
        save_dir = os.path.join(self.opt.result_dir, self.opt.encoder.root)
        os.makedirs(save_dir, exist_ok=True)
        train_dataset = LatentDataset(generator, directions, self.opt, save_dir, create_new_data=False)

        LABEL_MEAN = np.mean(train_dataset.labels, 0)
        LABEL_STD = np.std(train_dataset.labels, 0) + 1e-5

        train_dataset.labels = (train_dataset.labels - LABEL_MEAN) / LABEL_STD

        test_dataset = LatentDataset(generator, directions, self.opt, save_dir, create_new_data=False)

        test_dataset.labels = (test_dataset.labels - LABEL_MEAN) / LABEL_STD

        val_dataset = LatentDataset(generator, directions, self.opt, save_dir, create_new_data=False)

        val_dataset.labels = (val_dataset.labels - LABEL_MEAN) / LABEL_STD

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.opt.encoder.batch_size,
                                                   pin_memory=True, shuffle=True)

        valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.opt.encoder.batch_size,
                                                   pin_memory=True, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.opt.encoder.batch_size, shuffle=False)

        return {"train": train_loader, "valid": valid_loader, "test": test_loader}

    @torch.no_grad()
    def evaluate_model(self, generator, deformator, shift_predictor, trainer):
        n_steps = 100

        percents = torch.empty([n_steps])
        for step in range(n_steps):
            z = make_noise(128, generator.latent_size, truncation=self.opt.algo.ld.truncation).cuda()

            target_indices, shifts, basis_shift = trainer.make_shifts(deformator.input_dim)
            shift = deformator(basis_shift)

            imgs, _ = generator(z, self.opt.depth, self.opt.alpha)
            imgs_shifted, _ = generator(z + shift, self.opt.depth, self.opt.alpha)

            logits, _ = shift_predictor(imgs, imgs_shifted)
            percents[step] = (torch.argmax(logits, dim=1) == target_indices.cuda()).to(torch.float32).mean()

        return percents.mean()
