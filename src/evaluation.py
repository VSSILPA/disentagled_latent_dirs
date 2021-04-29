import torch
import torch.nn as nn
import logging
from utils import make_noise
from train import Trainer
from models.latent_regressor import Encoder,LatentRunner
import numpy as np
import time
from metrics.betavae_metric import BetaVAEMetric
from metrics.factor_vae_metric import FactorVAEMetric
from metrics.mig import MIG
from metrics.dci_metric import DCIMetric
from latent_dataset import LatentDataset
import json
import os
import shutil


log = logging.getLogger(__name__)


class Evaluator(object):
    def __init__(self,data, config, opt):
        self.config = config
        self.device = torch.device('cuda:' + str(opt.device_id))
        self.opt = opt
        self.encoder = Encoder(latent_dimension=self.opt.encoder.num_directions, backbone="cnn_encoder",
                               **self.opt.dataset.params[self.opt.dataset.name])
        self.data = data

    def compute_metrics(self,generator, directions):
        start_time = time.time()
        encoder = self._train_encoder(generator,directions)

        beta_vae = BetaVAEMetric(self.data, self.device)
        factor_vae = FactorVAEMetric(self.data, self.device, self.config)
        mig = MIG(self.data, self.device)
        beta_vae_metric = beta_vae.compute_beta_vae(encoder, np.random.RandomState(self.opt.random_seed),
                                                    batch_size=64,
                                                    num_train=10000, num_eval=5000)
        factor_vae_metric = factor_vae.compute_factor_vae(encoder, np.random.RandomState(self.opt.random_seed),
                                                          batch_size=64, num_train=10000, num_eval=5000,
                                                          num_variance_estimate=10000)
        mutual_info_gap = mig.compute_mig(encoder, num_train=10000, batch_size=128)

        metrics = {'beta_vae': beta_vae_metric, 'factor_vae': factor_vae_metric, 'mig': mutual_info_gap[
            "discrete_mig"]}
        self.metric_eval['beta_vae'].append(metrics['beta_vae']["eval_accuracy"])
        self.metric_eval['factor_vae'].append(metrics['factor_vae']["eval_accuracy"])
        self.metric_eval['mig'].append(metrics['mig'])
        logging.info(
            "Epochs  %d / %d Time taken %d sec B-VAE: %.3f, F-VAE %.3F, MIG : %.3f" % (epoch, self.config['epochs'],
                                                                                       time.time() - start_time,
                                                                                       metrics['beta_vae'][
                                                                                           "eval_accuracy"],
                                                                                       metrics['factor_vae'][
                                                                                           "eval_accuracy"],
                                                                                       metrics['mig']))
        return self.metric_eval


    def _train_encoder(self, generator, directions):
        model = self.encoder.to(self.device)
        model = nn.DataParallel(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.opt.encoder.latent_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.opt.encoder.latent_step_size,
            gamma=self.opt.encoder.latent_gamma,
        )

        exp_name = self.config['exp_name']
        cwd = os.path.dirname(os.getcwd()) + f'/results/{exp_name}'
        self.catalyst_logdir = os.path.join(cwd, "encoder_train")
        os.makedirs(self.catalyst_logdir, exist_ok=True)

        loader = self._get_encoder_train_data(generator, directions)

        runner = LatentRunner()
        logdir = os.path.join(self.catalyst_logdir, "tmp")

        runner.train(model=model, optimizer=optimizer, loaders=loader, logdir=logdir, num_epochs=self.opt.encoder.latent_nb_epochs,
                     scheduler=scheduler, verbose=True, load_best_on_end=True,)

        state_dict = torch.load(os.path.join(logdir, "checkpoints/best.pth"))[
            "model_state_dict"
        ]

        with open(os.path.join(logdir, "checkpoints/_metrics.json"), "r") as f:
            metrics = json.load(f)

        shutil.rmtree(logdir)

        del model

        return state_dict, metrics


    def _get_encoder_train_data(self, generator, directions):
        train_dataset = LatentDataset(generator, directions, self.opt.dataset, self.opt.encoder.num_samples,
                                      self.opt.encoder.generator_bs, create_new_data=False, root=self.opt.encoder.root)

        LABEL_MEAN = np.mean(train_dataset.labels, 0)
        LABEL_STD = np.std(train_dataset.labels, 0) + 1e-5

        train_dataset.labels = (train_dataset.labels - LABEL_MEAN) / LABEL_STD

        test_dataset = LatentDataset(generator, directions, self.opt.dataset, 5000, self.opt.encoder.generator_bs, create_new_data=True,
                                     root=self.opt.encoder.root)

        test_dataset.labels = (test_dataset.labels - LABEL_MEAN) / LABEL_STD

        val_dataset = LatentDataset(generator, directions, self.opt.dataset, 5000, self.opt.encoder.generator_bs, create_new_data=True,
                                    root=self.opt.encoder.root)

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
            z = make_noise(self.config['batch_size'], generator.dim_z).to(self.device)
            target_indices, shifts, basis_shift = trainer.make_shifts(deformator.input_dim)
            shift = deformator(basis_shift)

            imgs = generator(z)
            imgs_shifted = generator.gen_shifted(z, shift)

            logits, _ = shift_predictor(imgs, imgs_shifted)
            percents[step] = (torch.argmax(logits, dim=1) == target_indices.cuda()).to(torch.float32).mean()

        return percents.mean()
