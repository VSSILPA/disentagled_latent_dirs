import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

'''
Generator Model Definition
'''


class Generator(nn.Module):
    def __init__(self, dim_z=5, dim_c_cont=5):
        super(Generator, self).__init__()
        self.dim_latent = dim_z + dim_c_cont
        self.fc1 = nn.Linear(in_features=self.dim_latent, out_features=128)
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=4 * 4 * 64)
        self.bn2 = nn.BatchNorm1d(4 * 4 * 64)
        self.upconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.upconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.upconv5 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.upconv6 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        # Layer 1: [-1, dim_latent] -> [-1, 128]
        z = self.bn1(F.relu(self.fc1(z)))

        # Layer 2: [-1, 1024] -> [-1, 4*4*64]
        z = self.bn2(F.relu(self.fc2(z)))

        # Shape Change: [-1, 4*4*64] -> [-1, 64, 4, 4]
        z = z.view(-1, 64, 4, 4)

        # Layer 3: [-1, 64, 4, 4] -> [-1, 64, 8, 8]
        z = self.bn3(F.leaky_relu(self.upconv3(z), negative_slope=0.2))

        # Layer 4: [-1, 64, 8, 8] -> [-1, 32, 16, 16]
        z = self.bn4(F.leaky_relu(self.upconv4(z), negative_slope=0.2))

        # Layer 5: [-1, 32, 16, 16] -> [-1, 32, 32,32]
        z = self.bn5(F.leaky_relu(self.upconv5(z), negative_slope=0.2))

        # Layer 6: [-1, 32, 32, 32] -> [-1, 1, 64, 64]
        img = torch.sigmoid(self.upconv6(z))

        return img


class Discriminator(nn.Module):

    def __init__(self, dim_c_cont=5):
        super(Discriminator, self).__init__()
        self.dim_c_cont = dim_c_cont
        # Shared layers
        self.module_shared = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=1,
                                    out_channels=32,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_channels=32,
                                    out_channels=64,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            Reshape(-1, 64 * 4 * 4),
            spectral_norm(nn.Linear(in_features=64 * 4 * 4, out_features=128)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # Layer for Disciminating
        self.module_D = nn.Sequential(
            spectral_norm(nn.Linear(in_features=128, out_features=1)),
            nn.Sigmoid()
        )

        self.module_Q = nn.Sequential(
            spectral_norm(nn.Linear(in_features=128, out_features=128)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.latent_cont = spectral_norm(nn.Linear(
            in_features=128, out_features=self.dim_c_cont))


    def forward(self, z):
        z = z.type(torch.cuda.FloatTensor)
        out = self.module_shared(z.view(-1,1,64,64))
        probability = self.module_D(out)
        probability = probability.squeeze()
        internal_Q = self.module_Q(out)
        c_cont = self.latent_cont(internal_Q)
        return c_cont,probability


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class InfoGan(object):
    def __init__(self, config):
        super(InfoGan, self).__init__()

        self.decoder = Generator(dim_z=config['noise_dim'], dim_c_cont=config['latent_dim'])
        self.encoder = Discriminator(dim_c_cont=config['latent_dim'])

    def dummy(self):
        print('This is a dummy function')
