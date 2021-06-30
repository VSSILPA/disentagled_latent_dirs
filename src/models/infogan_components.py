import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

c1_len = 10  # Multinomial
c2_len = 0  # Gaussian
c3_len = 0  # Bernoulli
z_len = 64  # Noise vector length
embedding_len = 128


class Conv2d(nn.Conv2d):
    def reset_parameters(self):
        stdv = np.sqrt(6 / ((self.in_channels + self.out_channels) * np.prod(self.kernel_size)))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()


class ConvTranspose2d(nn.ConvTranspose2d):
    def reset_parameters(self):
        stdv = np.sqrt(6 / ((self.in_channels + self.out_channels) * np.prod(self.kernel_size)))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()


class Linear(nn.Linear):
    def reset_parameters(self):
        stdv = np.sqrt(6 / (self.in_features + self.out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = Linear(z_len + c1_len + c2_len + c3_len, 1024)
        self.fc2 = Linear(1024, 7 * 7 * 128)

        self.convt1 = ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.convt2 = ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(7 * 7 * 128)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x))).view(-1, 128, 7, 7)

        x = F.relu(self.bn3(self.convt1(x)))
        x = self.convt2(x)

        return F.sigmoid(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = Conv2d(1, 64, kernel_size=4, stride=2, padding=1)  # 28 x 28 -> 14 x 14
        self.conv2 = Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 14 x 14 -> 7 x 7

        self.fc1 = Linear(128 * 8 ** 2, 1024)
        self.fc2 = Linear(1024, 1)
        self.fc1_q = Linear(1024, c1_len)
        self.module_S = nn.Sequential(nn.Linear(in_features=1024, out_features=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.latent_similar = nn.Linear(in_features=128, out_features=10)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.bn1(self.conv2(x))).view(-1, 8 ** 2 * 128)
        x = F.leaky_relu(self.bn2(self.fc1(x)))
        similarity_vec = self.module_S(x)
        latent_vec = self.latent_similar(similarity_vec)
        return F.sigmoid(self.fc2(x)), self.fc1_q(x) ,latent_vec
