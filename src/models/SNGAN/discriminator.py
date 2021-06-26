import torch.nn as nn
import torch


class Discriminator(nn.Module):
    leak = 0.1
    l_dim = 10

    class Embed(nn.Module):
        def __init__(self):
            super(Discriminator.Embed, self).__init__()
            self.layer = nn.utils.spectral_norm(
                nn.Linear(Discriminator.l_dim, 128 * 7 * 7))

        def forward(self, l):
            return self.layer(l)

    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.utils.spectral_norm(nn.Conv2d(1, 32, 3, stride=1, padding=1))
        self.layer2 = nn.utils.spectral_norm(nn.Conv2d(32, 32, 4, stride=2, padding=1))
        self.layer3 = nn.utils.spectral_norm(nn.Conv2d(32, 64, 3, stride=1, padding=1))
        self.layer4 = nn.utils.spectral_norm(nn.Conv2d(64, 64, 4, stride=2, padding=1))
        self.layer5 = nn.utils.spectral_norm(nn.Conv2d(64, 128, 3, stride=1, padding=1))
        self.fc = nn.utils.spectral_norm(nn.Linear(128 * 7 * 7, 1))
        self.embed = self.Embed()

    def forward(self, x, l):
        m = x
        m = nn.LeakyReLU(Discriminator.leak)(self.layer1(m))
        m = nn.LeakyReLU(Discriminator.leak)(self.layer2(m))
        m = nn.LeakyReLU(Discriminator.leak)(self.layer3(m))
        m = nn.LeakyReLU(Discriminator.leak)(self.layer4(m))
        m = nn.LeakyReLU(Discriminator.leak)(self.layer5(m))
        m = m.view(-1, 128 * 7 * 7)
        e = self.embed(l)
        return self.fc(m) + torch.sum(m * e)
