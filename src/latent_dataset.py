import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
from train import Trainer

class LatentDataset(Dataset):
    def __init__(self, generator, latent_directions, dataset="dsprites", N=50000, generator_bs=50,
                create_new_data=False, root="generated_data"):
        super().__init__()
        assert N % generator_bs == 0
        self.device = list(generator.parameters())[0].device
        self.root = root

        if create_new_data:
            self._generate_data(generator, generator_bs, dataset, N=N, save=False)
        else:
            exist = self._try_load_cached(dataset)
            if not exist:
                print("Building dataset from scratch.")
                self._generate_data(generator=generator, generator_bs=generator_bs, dataset=dataset, N=N, save=True)

        self.labels = self.labels @ latent_directions

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
            image, w = generator(z)
            x = (((image.detach().cpu().numpy() + 1) / 2) * 255).astype(np.uint8)
            x = torch.clamp(x, -1, 1)
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