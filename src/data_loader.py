import torchvision
from torch.utils.data import DataLoader


def get_data_loader(config, opt):
    if opt.dataset == 'dsprites':
        from data.dsprites import DSprites
        data = DSprites(config, opt)
        return data
    elif opt.dataset == 'mnist':

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        train_data = torchvision.datasets.MNIST('../data/mnist/', train=True, download=True,
                                          transform=transform)
        test_data = torchvision.datasets.MNIST('../data/mnist/', train=False, download=True,
                                          transform=transform)
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True,drop_last=True)
        test_loader = DataLoader(test_data, batch_size=128, shuffle=True,drop_last=True)

        return (train_loader, test_loader)
    elif opt.dataset == 'celebA':
        raise NotImplementedError
    else:
        raise NotImplementedError
