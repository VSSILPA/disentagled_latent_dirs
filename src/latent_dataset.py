import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
from config import generator_kwargs
import random
import logging


class LatentDataset(Dataset):
    def __init__(self, generator, latent_directions, opt,root, create_new_data=False):
        super().__init__()
        assert opt.encoder.num_samples % opt.encoder.generator_bs == 0
        self.device = list(generator.parameters())[0].device
        self.opt = opt
        self.root = root
        self.set_seed(self.opt.random_seed)

        if create_new_data:
            self._generate_data(generator, self.opt.encoder.generator_bs, self.opt.dataset, N=self.opt.encoder.num_samples, save=False)
        else:
            exist = self._try_load_cached(self.opt.dataset)
            if not exist:
                print("Building dataset from scratch.")
                self._generate_data(generator=generator, generator_bs=self.opt.encoder.generator_bs, dataset=self.opt.dataset,
                                    N=self.opt.encoder.num_samples, save=True)

        self.labels = self.labels @ latent_directions.weight.detach().cpu().numpy()

    def _try_load_cached(self, dataset):
        path = os.path.join(self.root, dataset + ".npz")
        if os.path.exists(path):
            "Loading cached dataset."
            arr = np.load(path)
            self.images, self.labels = arr["images"], arr["labels"]
            return True
        else:
            return False

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    @torch.no_grad()
    def _generate_data(self, generator, generator_bs, dataset, N, save=False):
        images = []
        labels = []

        for _ in range(N // generator_bs):

            z = torch.randn(generator_bs, generator.style_dim).to(self.device)
            w = generator.style(z)
            x = generator([w], **generator_kwargs)[0]
            x = torch.clamp(x, -1, 1)
            x = (((x.detach().cpu().numpy() + 1) / 2) * 255).astype(np.uint8)
            images.append(x)
            labels.append(w.detach().cpu().numpy())

        self.images = np.concatenate(images, 0)
        logging.info('max value in image :' + str(self.images.max()))
        self.labels = np.concatenate(labels, 0)
        if save:
            path = os.path.join(self.root, dataset + ".npz")
            np.savez(path, images=self.images, labels=self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = self.images[item]
        img = 2 * (img/255) - 1
        return torch.from_numpy(img).float(), torch.from_numpy(self.labels[item]).float()