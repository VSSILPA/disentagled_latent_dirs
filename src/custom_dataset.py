from torch.utils.data import Dataset
import torch


class NewDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]

        if self.transform is not None:
            img = self.transform(img)
        return torch.FloatTensor(img)

    def __len__(self):
        return len(self.data)
