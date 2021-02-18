import torch
import logging
from utils import make_noise

log = logging.getLogger(__name__)


class Evaluator(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda:' + str(config['device_id']))

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
