from torch.utils.data import Dataset


class NewDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index].clone(), self.targets[index].clone()

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.targets)

    def _assign_targets_(self, index, assign_val):
        self.targets[index] = assign_val