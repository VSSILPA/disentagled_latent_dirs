import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
from train import Trainer

class LatentDataset(Dataset):
    def __init__(self, generator, latent_directions, opt,root, create_new_data=False):
        super().__init__()
        assert opt.encoder.num_samples % opt.encoder.generator_bs == 0
        self.device = list(generator.parameters())[0].device
        self.opt = opt
        self.root = root

        if create_new_data:
            self._generate_data(generator, self.opt.encoder.generator_bs, self.opt.dataset, N=self.opt.encoder.num_samples, save=False)
        else:
            exist = self._try_load_cached(self.opt.dataset)
            if not exist:
                print("Building dataset from scratch.")
                self._generate_data(generator=generator, generator_bs=self.opt.encoder.generator_bs, dataset=self.opt.dataset,
                                    N=self.opt.encoder.num_samples, save=True)

        self.labels = self.labels @ latent_directions.linear.weight.detach().cpu().numpy()

    def _try_load_cached(self, dataset):
        path = os.path.join(self.root, dataset + ".npz")
        if os.path.exists(path):
            "Loading cached dataset."
            arr = np.load(path)
            self.images, self.labels = arr["images"], arr["labels"]
            return True
        else:
            return False

    @torch.no_grad()
    def _generate_data(self, generator, generator_bs, dataset, N, save=False):
        images = []
        labels = []
        for _ in range(N // generator_bs):
            z = Trainer.make_noise(generator_bs, generator.latent_size,
                                   truncation=True).to(self.device)
            image, w = generator(z,self.opt.alpha,self.opt.depth )
            x = torch.clamp(image, -1, 1)
            x = (((x.detach().cpu().numpy() + 1) / 2) * 255).astype(np.uint8)
            images.append(x)
            labels.append(w.detach().cpu().numpy())

        self.images = np.concatenate(images, 0)
        self.labels = np.concatenate(labels, 0)
        if save:
            path = os.path.join(self.root, dataset + ".npz")
            np.savez(path, images=self.images, labels=self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = self.images[item]
        img = 2 * (img / 255.0) - 1
        return (torch.from_numpy(img).float(), torch.from_numpy(self.labels[item]).float(),)